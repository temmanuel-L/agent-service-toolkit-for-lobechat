"""
RAG 重排（Rerank）核心逻辑。

从 RagService 中抽取的通用 rerank 实现，统一通过 core.llm.get_rerank
调用外部 Rerank API，并结合 settings 做时间/分数等保护。
"""

from __future__ import annotations

import time
from typing import List

from llama_index.core.schema import NodeWithScore, QueryBundle

from core.llm import get_rerank
from core.settings import settings
from utils.log_utils import get_logger

logger = get_logger(__name__)


def rerank_nodes(
    nodes: List[NodeWithScore],
    query_str: str,
    top_k: int,
) -> List[NodeWithScore]:
    """
    使用 Rerank 模型对候选节点进行精排，返回按相关性重排后的 top_k。
    若未启用或模型不可用，直接按原分数截断返回。
    """
    postprocessor = get_rerank()
    if not postprocessor or not nodes:
        return nodes[:top_k]
    try:
        t0 = time.perf_counter()
        query_bundle = QueryBundle(query_str=query_str)
        reranked = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        # ---- Rerank 时间限制（兜底）----
        limit_s = float(getattr(settings, "RAG_RERANK_TIME_LIMIT", 0.0) or 0.0)
        elapsed_s = time.perf_counter() - t0
        if limit_s > 0 and elapsed_s > limit_s:
            logger.warning(
                "Rerank 超时(%.2fs>%.2fs)，降级为原序截断: kb_nodes=%d",
                elapsed_s,
                limit_s,
                len(nodes),
            )
            return nodes[:top_k]

        reranked = reranked[:top_k]

        # ---- Rerank 最低分过滤（可选）----
        min_score = float(getattr(settings, "RAG_RERANK_MIN_SCORE", 0.0) or 0.0)
        if min_score > 0 and reranked:
            kept = [
                n for n in reranked
                if float(getattr(n, "score", 0.0) or 0.0) >= min_score
            ]
            if not kept:
                logger.warning(
                    "Rerank 结果全部低于阈值(min_score=%.4f)，降级为原序截断: top_k=%d",
                    min_score,
                    top_k,
                )
                return nodes[:top_k]
            return kept

        return reranked
    except Exception as e:
        logger.warning(f"Rerank 执行失败，降级为原序截断: {e}")
        return nodes[:top_k]


__all__ = [
    "rerank_nodes",
]

