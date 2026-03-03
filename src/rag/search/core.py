"""
RAG 检索（Search）核心逻辑。

职责：
- 为 RagService 提供“单库检索”的封装，输出结构化的 SearchResult；
- 后续可以在不修改 RagService 的前提下扩展更多检索策略（仅调整本模块）。
"""

from __future__ import annotations

import time
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from core.settings import settings
from rag.schema.schema_search import SearchRequest, SearchResult, SearchHit
from rag.search.fusion import reciprocal_rank_fusion
from rag.search.filters import apply_search_filters_to_hits
from utils.log_utils import get_logger

logger = get_logger(__name__)


async def hybrid_search_single_kb(
    request: SearchRequest,
    *,
    kb_id: str,
    vector_store,
    embed_model,
    bm25_retriever: BM25Retriever | None,
    corpus_size: int,
) -> SearchResult:
    """
    对单个知识库执行一次混合检索（向量 + 可选 BM25 + RRF + 可选 rerank 外层控制）。

    注意：
    - 本函数不直接依赖 QdrantClient，只使用已经构造好的 vector_store 与 bm25_retriever；
    - 不负责 rerank，rerank 由调用方在得到 SearchResult 后统一处理。
    """
    query = request.query
    similarity_top_k = request.top_k

    t0 = time.perf_counter()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # 第一阶段召回：放大 top_k，提高跨文档命中率
    base_multiplier = 3
    recall_top_k = min(similarity_top_k * base_multiplier, 80)

    vector_retriever = index.as_retriever(similarity_top_k=recall_top_k)
    vector_nodes: List[NodeWithScore] = await vector_retriever.aretrieve(query)
    vector_ms = (time.perf_counter() - t0) * 1000

    final_nodes = vector_nodes
    bm25_ms = 0.0

    if bm25_retriever is not None and settings.RAG_HYBRID_SEARCH:
        # 动态调整 BM25 top_k，确保不超过语料库大小
        effective_top_k = min(recall_top_k, corpus_size)
        bm25_retriever._similarity_top_k = effective_top_k  # type: ignore[attr-defined]

        t1 = time.perf_counter()
        bm25_nodes = bm25_retriever.retrieve(query)
        bm25_ms = (time.perf_counter() - t1) * 1000

        bm25_weight = settings.RAG_BM25_WEIGHT
        fused_nodes = reciprocal_rank_fusion(
            vector_nodes,
            bm25_nodes,
            recall_top_k,
            bm25_weight=bm25_weight,
        )
        final_nodes = fused_nodes[:similarity_top_k]

        logger.info(
            "Search(single_kb): kb=%s, vector=%d(%.0fms), bm25=%d(%.0fms), fused=%d",
            kb_id,
            len(vector_nodes),
            vector_ms,
            len(bm25_nodes),
            bm25_ms,
            len(final_nodes),
        )
    else:
        final_nodes = vector_nodes[:similarity_top_k]
        logger.info(
            "Search(single_kb, vector-only): kb=%s, vector=%d(%.0fms)",
            kb_id,
            len(final_nodes),
            vector_ms,
        )

    # 转换为 SearchResult
    hits: List[SearchHit] = []
    for nws in final_nodes:
        text = (nws.text or "").strip()
        meta = getattr(nws.node, "metadata", {}) or {}
        hits.append(
            SearchHit(
                text=text,
                score=float(nws.score or 0.0),
                metadata=meta,
            )
        )

    # 基于 metadata 的后过滤（例如 doc_title 包含某个子串）
    hits = apply_search_filters_to_hits(hits, request.filters)

    return SearchResult(hits=hits)


__all__ = [
    "hybrid_search_single_kb",
]

