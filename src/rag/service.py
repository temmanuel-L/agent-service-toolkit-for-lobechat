"""
RAG (Retrieval-Augmented Generation) 服务模块

该模块提供知识库的文档摄入、向量化存储和检索功能。

架构：
- 使用 LlamaIndex 作为文档处理和索引框架
- 使用 Qdrant 作为向量数据库后端
- 支持多种文档格式（PDF、DOCX、PPTX、HTML 等）
- 支持 BM25 + 向量的混合检索，通过 RRF 融合提升检索质量

核心改进（相比纯向量检索）：
1. 文档元数据增强：提取文档标题注入每个 chunk，使标题/作者等信息参与 embedding
2. BM25 关键词检索：精确命中标题、人名、术语等向量检索可能遗漏的内容
3. Reciprocal Rank Fusion：无需分数归一化即可合并异构检索结果
"""
import os
import re
import time
import httpx
import tempfile
import threading
import tiktoken
import logging
from typing import Optional, List

# 屏蔽第三方库冗长的调试日志
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("bm25s").setLevel(logging.WARNING)

from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from urllib.parse import urlparse

from core.settings import settings
from core.llm import get_embedding_model, get_rerank
from utils.log_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# RAG 配置参数
# ============================================================================
# chunk_size: 每个文本块的目标大小（字符数）
#   - 较小值（512-1024）: 适合简单问答、FAQ 类文档
#   - 较大值（2048-4096）: 适合技术文档、学术论文等需要完整上下文的场景
#   - 当前设置 2048: 平衡了上下文完整性和检索精确性
#
# chunk_overlap: 相邻块之间的重叠字符数
#   - 目的是确保跨块边界的信息不会丢失
#   - 建议为 chunk_size 的 10-20%
#   - 当前设置 256: 约 12.5%，确保句子级别的连续性
#
# similarity_top_k: 检索返回的最相关文本块数量
#   - 较小值（3-5）: 精确匹配，适合简单直接的问题
#   - 较大值（8-15）: 覆盖更广，适合概述性问题或复杂查询
#   - 当前默认 8: 为复杂问题提供足够上下文
#
# 注意：chunk_size/overlap、top_k 已移至 core.settings（RAG_CHUNK_SIZE、RAG_CHUNK_OVERLAP、RAG_DEFAULT_TOP_K）

# BM25 缓存：从 Qdrant scroll 加载节点的单页大小
_BM25_SCROLL_PAGE_SIZE = 500


# ============================================================================
# 中英文混合分词器（供 BM25 使用）
# ============================================================================
def _hybrid_tokenize(text: str) -> list[str]:
    """
    中英文混合分词器，供 BM25 检索使用。

    分词策略：
    - CJK 字符（中/日/韩）：按单字拆分（中文没有空格分隔）
    - 英文 / 数字：按完整单词拆分
    - 所有 token 统一小写化

    示例:
        "A NUMERICAL MODEL 蒸汽管道" → ["a", "numerical", "model", "蒸", "汽", "管", "道"]

    这种策略确保：
    1. 英文标题/人名可以被精确匹配（如 "DAWID TALER"）
    2. 中文关键词也能被逐字匹配（如 "薪酬"、"报告"）
    """
    # Preserve CJK characters and alphanumeric sequences
    # Add support for hyphens in English words (e.g. "state-of-the-art")
    # Add support for hyphens in English words (e.g. "state-of-the-art")
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9-]+', text.lower())


def _looks_like_english_title_query(text: str) -> bool:
    """
    简单启发式：判断查询是否更像「英文标题 / 论文题目」。

    使用场景：
    - 当用户询问英文论文标题或报告名称（如 "A NUMERICAL MODEL OF ..."），
      这类查询往往在文档中是精确出现的短语，BM25 更适合负责召回；
      语义向量有时会被其他英文长文档「吸偏」。

    规则（尽量保守，避免误伤中文场景）：
    - 至少包含 4 个长度 >=3 的英文单词；
    - 合并后的英文部分长度 > 20；
    - 中文字符数量较少（< 6）。
    """
    if not text:
        return False

    english_tokens = re.findall(r"[a-zA-Z]{3,}", text)
    if len(english_tokens) < 4:
        return False

    english_concat = " ".join(english_tokens)
    if len(english_concat) <= 20:
        return False

    return True


# ============================================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================================
def _reciprocal_rank_fusion(
    vector_results: list[NodeWithScore],
    bm25_results: list[NodeWithScore],
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 0.4,  # Configurable weight
) -> list[NodeWithScore]:
    """
    Weighted Reciprocal Rank Fusion (W-RRF).
    
    改进：添加权重参数，允许调节 Vector vs BM25 的影响力。
    Score = (1 - w) * RR_vector + w * RR_bm25
    RR = 1 / (k + rank)
    """
    # node_id → (累积 RRF 分数, NodeWithScore 对象)
    score_map: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}
    
    # Vector results (Weight: 1.0 - bm25_weight)
    vec_w = 1.0 - bm25_weight
    for rank, nws in enumerate(vector_results):
        nid = nws.node.node_id
        score = vec_w * (1.0 / (rrf_k + rank + 1))
        score_map[nid] = score_map.get(nid, 0.0) + score
        if nid not in node_map:
            node_map[nid] = nws

    # BM25 results (Weight: bm25_weight)
    for rank, nws in enumerate(bm25_results):
        nid = nws.node.node_id
        score = bm25_weight * (1.0 / (rrf_k + rank + 1))
        score_map[nid] = score_map.get(nid, 0.0) + score
        if nid not in node_map:
            node_map[nid] = nws

    # 按 RRF 分数降序排列，取 top_k
    sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
    fused = []
    for nid in sorted_ids[:top_k]:
        nws = node_map[nid]
        fused.append(NodeWithScore(node=nws.node, score=score_map[nid]))

    return fused


# ============================================================================
# 文档标题提取
# ============================================================================
def _extract_doc_title(file_path: str, documents: list, file_name: str | None) -> str:
    """
    从文档中提取标题，用于注入到每个 chunk 的元数据。

    提取优先级：
    1. PDF 元数据中的 title 字段（pypdf）
    2. 文档首页前 5 行中最适合做标题的行（短、非空、非页码）
    3. 文件名去扩展名（兜底）

    标题会被注入到每个 chunk 的 metadata["doc_title"] 中，
    使得 embedding 包含文档标题信息。这解决了标题页文本短小、
    在纯向量检索中排名低的问题。

    Args:
        file_path: 临时文件的本地路径
        documents: SimpleDirectoryReader 加载的文档列表
        file_name: 原始文件名（可选）

    Returns:
        提取到的文档标题字符串，提取失败时返回空字符串
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ---- 策略 1: PDF 元数据 ----
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            meta = reader.metadata
            if meta and meta.title and meta.title.strip():
                title = meta.title.strip()
                logger.debug(f"从 PDF 元数据提取标题: '{title[:80]}'")
                return title
        except Exception:
            pass  # PDF 元数据不可用，继续尝试其他策略

    # ---- 策略 2: 从首页内容中启发式提取 ----
    if documents:
        first_text = documents[0].text[:1000]
        lines = [line.strip() for line in first_text.split('\n') if line.strip()]
        # 跳过纯数字行（页码）、过长行（正文段落）
        for line in lines[:8]:
            # 好的标题特征：长度适中（10-300 字符），不以数字开头（排除页码）
            if 5 < len(line) < 300 and not line[0].isdigit():
                logger.debug(f"从首页内容提取标题: '{line[:80]}'")
                return line

    # ---- 策略 3: 从文件名推断 ----
    if file_name:
        name_without_ext = os.path.splitext(file_name)[0]
        # 替换常见分隔符为空格
        title = re.sub(r'[_\-]+', ' ', name_without_ext).strip()
        if title:
            logger.debug(f"从文件名推断标题: '{title[:80]}'")
            return title

    return ""


class RagService:
    """
    RAG 服务：文档摄入 + 混合检索。

    混合检索架构：
    - 向量检索（Qdrant）：捕获语义相似性
    - BM25 关键词检索：精确匹配标题、人名、术语
    - RRF 融合：合并两种检索结果，取长补短

    BM25 缓存策略：
    - 首次查询某 collection 时，通过 Qdrant scroll API 加载全部节点文本
    - 构建 BM25 索引并缓存在内存中（线程安全）
    - 新文档摄入时自动失效对应 collection 的缓存
    - 后续查询直接使用缓存，开销仅 5-15ms
    """

    def __init__(self) -> None:
        # ---- Qdrant 客户端 ----
        self.api_key = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
        self.url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.aclient = AsyncQdrantClient(url=self.url, api_key=self.api_key)

        # ---- Embedding 模型 ----
        # 将 LangChain 嵌入模型封装为 LlamaIndex 格式
        lc_embeddings = get_embedding_model()
        # 1) Token 批处理：防止批量 embedding 超过模型上下文限制（如 8192）
        from rag.embedding_batcher import TokenAwareEmbedding, CacheAwareEmbedding
        token_aware = TokenAwareEmbedding(
            lc_embeddings,
            max_batch_tokens=6000,
        )
        # 2) 复用 memory 的 EmbeddingCache，与长期记忆共用同一缓存，减少重复 API 调用
        self.token_aware_embed = CacheAwareEmbedding(token_aware)
        # embed_batch_size 仅控制 LlamaIndex 每批传入条数，实际拆分由 TokenAwareEmbedding 控制
        self.embed_model = LangchainEmbedding(self.token_aware_embed, embed_batch_size=100)

        # ---- BM25 缓存 ----
        # key=collection_name, value=(BM25Retriever, corpus_size)
        self._bm25_cache: dict[str, tuple] = {}
        self._bm25_lock = threading.Lock()

    # ================================================================
    # URL 映射（Docker 内网）
    # ================================================================
    def _map_url_internally(self, url: str) -> tuple[str, dict]:
        """
        在需要时将 URL 从 'localhost' 映射为 Docker 可访问的主机。
        返回 (映射后的 url, 带原始 Host 的请求头)。
        """
        parsed = urlparse(url)
        original_host = parsed.netloc

        # 允许用户通过环境变量覆盖，以指定特定的 minio 服务名
        internal_host = os.getenv("S3_INTERNAL_HOST", "host.docker.internal")

        headers: dict[str, str] = {}
        new_url = url

        if "localhost" in original_host or "127.0.0.1" in original_host:
            new_url = url.replace(original_host.split(':')[0], internal_host)
            # 关键：必须保留原始 Host 请求头，因为 S3 预签名 URL 的签名中包含 'host'
            headers["Host"] = original_host
            logger.info(f"Mapping external URL to internal: {url} -> {new_url} (Preserving Host: {original_host})")

        return new_url, headers

    # ================================================================
    # BM25 缓存管理
    # ================================================================
    def _get_or_build_bm25(self, kb_id: str, top_k: int):
        """
        获取或构建指定 collection 的 BM25 检索器（线程安全）。

        流程：
        1. 检查内存缓存是否命中
        2. 未命中则通过 Qdrant scroll API 分页加载全部节点文本
        3. 构建 BM25 索引并写入缓存

        性能：
        - 首次构建：100-300ms（取决于 collection 大小）
        - 后续查询：直接返回缓存对象，开销 < 1ms

        Args:
            kb_id: Qdrant collection 名称
            top_k: BM25 返回的结果数量

        Returns:
            (BM25Retriever, corpus_size) 元组，构建失败时返回 (None, 0)
        """
        # 快速路径：缓存命中
        cached = self._bm25_cache.get(kb_id)
        if cached is not None:
            bm25, corpus_size = cached
            # 更新 top_k（BM25Retriever 的 similarity_top_k 可能变化）
            bm25._similarity_top_k = top_k
            return bm25, corpus_size

        with self._bm25_lock:
            # 双重检查（另一个线程可能已经构建完成）
            cached = self._bm25_cache.get(kb_id)
            if cached is not None:
                bm25, corpus_size = cached
                bm25._similarity_top_k = top_k
                return bm25, corpus_size

            t0 = time.perf_counter()
            try:
                all_nodes: list[TextNode] = []
                offset = None  # Qdrant scroll API 的分页偏移

                while True:
                    records, next_offset = self.client.scroll(
                        collection_name=kb_id,
                        limit=_BM25_SCROLL_PAGE_SIZE,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for record in records:
                        # LlamaIndex 的 QdrantVectorStore 将文本存储在 payload 的
                        # "text" 字段或 "_node_content" JSON 中
                        text = ""
                        metadata: dict = {}
                        payload = record.payload or {}
                        if "text" in payload:
                            text = payload["text"]
                            metadata = payload.get("metadata", {}) or {}
                        elif "_node_content" in payload:
                            import json
                            try:
                                nc = json.loads(payload["_node_content"])
                                text = nc.get("text", "")
                                # 将节点元数据一并取出，用于构造前缀（doc_title / file_name 等）
                                metadata = nc.get("metadata", {}) or {}
                            except (json.JSONDecodeError, TypeError):
                                pass
                        if text:
                            # 在 BM25 侧按需为文本加上轻量级元数据前缀，而不修改原始存储
                            meta_prefix_parts: list[str] = []
                            doc_title = str(metadata.get("doc_title", "")).strip()
                            file_name = str(metadata.get("file_name", "")).strip()
                            if doc_title:
                                meta_prefix_parts.append(f"[TITLE] {doc_title}")
                            if file_name:
                                meta_prefix_parts.append(f"[FILE] {file_name}")
                            if meta_prefix_parts:
                                meta_prefix = " ".join(meta_prefix_parts)
                                text_with_prefix = f"{meta_prefix}\n\n{text}"
                            else:
                                text_with_prefix = text

                            all_nodes.append(TextNode(
                                text=text_with_prefix,
                                id_=str(record.id),
                            ))

                    if next_offset is None or len(records) < _BM25_SCROLL_PAGE_SIZE:
                        break
                    offset = next_offset

                if not all_nodes:
                    logger.warning(f"BM25: collection '{kb_id}' 中无可用节点")
                    return None, 0

                corpus_size = len(all_nodes)

                bm25 = BM25Retriever.from_defaults(
                    nodes=all_nodes,
                    similarity_top_k=top_k,
                    tokenizer=_hybrid_tokenize,
                )
                self._bm25_cache[kb_id] = (bm25, corpus_size)

                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"BM25 索引构建完成: collection='{kb_id}', "
                    f"nodes={corpus_size}, elapsed={elapsed:.1f}ms"
                )
                return bm25, corpus_size

            except Exception as e:
                logger.error(f"BM25 索引构建失败: collection='{kb_id}': {e}")
                return None, 0

    def _rerank_nodes(
        self,
        nodes: list[NodeWithScore],
        query_str: str,
        top_k: int,
    ) -> list[NodeWithScore]:
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
            # 主要限制由 httpx timeout 保证（在 get_rerank 中注入），这里再做一次兜底：
            # 如果调用“返回得太慢”（例如服务端处理超时但仍返回），则放弃结果并降级。
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
            # 注意：不同供应商的分数尺度可能不同；仅在显式配置阈值时启用过滤。
            min_score = float(getattr(settings, "RAG_RERANK_MIN_SCORE", 0.0) or 0.0)
            if min_score > 0 and reranked:
                kept = [n for n in reranked if float(getattr(n, "score", 0.0) or 0.0) >= min_score]
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

    def invalidate_bm25_cache(self, kb_id: str) -> None:
        """
        使指定 collection 的 BM25 缓存失效。

        在以下场景调用：
        - 新文档摄入后（ingest_file）
        - 知识库删除后（delete_knowledge_base）

        下一次查询将重新从 Qdrant 加载节点并重建索引。
        """
        with self._bm25_lock:
            removed = self._bm25_cache.pop(kb_id, None)
            if removed is not None:
                logger.info(f"BM25 缓存已失效: collection='{kb_id}'")

    # ================================================================
    # 文档摄入
    # ================================================================
    async def ingest_file(self, file_url: str, kb_id: str, file_name: Optional[str] = None) -> int:
        """
        从 URL 下载文件，用 LlamaIndex 解析、分块，并存入 Qdrant。
        集合名即为 kb_id。

        改进点：
        - 自动提取文档标题并注入每个 chunk 的元数据
        - 摄入完成后自动失效 BM25 缓存
        """
        # 如需要则将 URL 映射到 Docker 内网并获取必要请求头
        internal_url, headers = self._map_url_internally(file_url)

        collection_name = kb_id
        logger.info(f"Starting ingestion: file={file_name or internal_url}, kb_id={kb_id}")

        # 1. 下载文件
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(internal_url, headers=headers)
            response.raise_for_status()
            file_content = response.content

        # 2. 将内容保存到临时文件（LlamaIndex 的 reader 通常需要文件路径）
        suffix = os.path.splitext(file_name or file_url.split('?')[0])[1].lower()
        if not suffix:
            suffix = ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            logger.info(f"Loading document from temp file: {tmp_path}")
            # 3. 使用 SimpleDirectoryReader 加载数据（自动支持 PDF、Docx、PPTX、HTML 等）
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            documents = reader.load_data()

            if not documents:
                logger.warning(f"No content extracted from {file_name or file_url}")
                return 0

            # 4. 提取文档标题，并注入到元数据中（文本本身保持只读，避免依赖具体实现）
            #    - 元数据用于后续 BM25 前缀拼接与日志展示
            #    - 向量侧仍按 LlamaIndex 默认方式处理文本与元数据
            doc_title = _extract_doc_title(tmp_path, documents, file_name)

            for doc in documents:
                # 元数据写入 metadata，便于后续引用和调试
                if doc_title:
                    doc.metadata["doc_title"] = doc_title
                if file_name:
                    doc.metadata["file_name"] = file_name

                # 确保所有元数据参与 embedding 和 LLM 上下文（不排除任何 key）
                doc.excluded_embed_metadata_keys = []
                doc.excluded_llm_metadata_keys = []

            if doc_title:
                logger.info(f"文档标题: '{doc_title[:80]}'")

            logger.info(
                f"Extracted {len(documents)} document pages/segments. "
                f"Starting indexing into Qdrant collection: {collection_name}"
            )

            # 5. 配置 Qdrant 向量存储与分块转换
            # 注意：LlamaIndex 在写入（from_documents → add）阶段会调用同步 client.create_collection，
            # 因此此处必须提供同步 QdrantClient；异步 AsyncQdrantClient 仅用于查询。
            vector_store = QdrantVectorStore(
                collection_name=collection_name,
                client=self.client,
                aclient=self.aclient,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 使用 tiktoken 进行基于 Token 的精准分块
            # 相比默认的基于字符分块，这能确保每个 chunk 严格适配 Embedding 模型的上下文窗口
            splitter = SentenceSplitter(
                chunk_size=settings.RAG_CHUNK_SIZE,
                chunk_overlap=settings.RAG_CHUNK_OVERLAP,
                tokenizer=tiktoken.get_encoding("cl100k_base").encode
            )
            transformations = [splitter]
            logger.info(
                f"使用基于Token的分块: chunk_size={settings.RAG_CHUNK_SIZE}, "
                f"chunk_overlap={settings.RAG_CHUNK_OVERLAP} (cl100k_base)"
            )

            # 6. 创建索引（解析 + 嵌入 + 写入）
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                transformations=transformations,
                show_progress=False,  # 生产环境设为 False 以保持日志简洁
            )

            # 7. 失效 BM25 缓存，下次查询时自动重建
            self.invalidate_bm25_cache(kb_id)

            logger.info(f"Successfully ingested {len(documents)} pages/nodes into collection '{collection_name}'")
            return len(documents)

        except Exception as e:
            logger.error(
                f"Failed to ingest file {file_name or file_url} into "
                f"collection {collection_name}: {str(e)}",
                exc_info=True,
            )
            raise e
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"Removed temp file: {tmp_path}")

    # ================================================================
    # 知识库删除
    # ================================================================
    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """从 Qdrant 中删除整个集合（知识库），同时清除 BM25 缓存。"""
        try:
            if self.client.collection_exists(kb_id):
                logger.info(f"Deleting collection: {kb_id}")
                self.client.delete_collection(kb_id)
                self.invalidate_bm25_cache(kb_id)
                return True
            else:
                logger.warning(f"Collection {kb_id} does not exist, nothing to delete.")
                return False
        except Exception as e:
            logger.error(f"Error deleting collection {kb_id}: {str(e)}")
            return False

    # ================================================================
    # 知识库查询（混合检索）
    # ================================================================
    async def query_knowledge(
        self,
        query_str: str,
        kb_ids: List[str],
        similarity_top_k: int | None = None,
    ) -> str:
        """
        跨多个知识库（collection）进行混合检索。

        当 settings.RAG_HYBRID_SEARCH 为 True 时：
        1. 对每个 collection 同时运行向量检索 + BM25 检索
        2. 用 Reciprocal Rank Fusion (RRF) 合并结果
        3. 返回去重后的 top-k 节点

        当 settings.RAG_HYBRID_SEARCH 为 False 时：
        退化为纯向量检索（与旧行为完全一致）

        Args:
            query_str: 查询字符串
            kb_ids: 要查询的知识库 ID 列表
            similarity_top_k: 返回的最相关文本块数量，默认使用 settings.RAG_DEFAULT_TOP_K

        Returns:
            str: 格式化的检索结果，每个段落带有 [Knowledge Segment N] 标记
        """
        if similarity_top_k is None:
            similarity_top_k = settings.RAG_DEFAULT_TOP_K
        if not kb_ids:
            return ""

        hybrid_enabled = settings.RAG_HYBRID_SEARCH
        mode_label = "hybrid(vector+BM25)" if hybrid_enabled else "vector-only"
        logger.info(
            f"知识库检索: query='{query_str[:50]}...', kb_ids={kb_ids}, "
            f"top_k={similarity_top_k}, mode={mode_label}"
        )

        all_segments: list[str] = []
        segment_count = 1

        for kb_id in kb_ids:
            try:
                # 检查 collection 是否存在
                if not self.client.collection_exists(kb_id):
                    logger.debug(f"Collection {kb_id} does not exist, skipping query.")
                    continue

                t0 = time.perf_counter()

                # ---- 向量检索（始终执行）----
                vector_store = QdrantVectorStore(
                    collection_name=kb_id,
                    aclient=self.aclient,
                    client=None,
                )
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=self.embed_model,
                )
                # 第一阶段召回使用「放大的 top_k」：
                # - 召回阶段适度放宽（如 3～5 倍），提高跨文档命中率；
                # - 最终返回给 LLM 的仍然是 similarity_top_k 条，保证上下文长度可控。
                base_multiplier = 3
                recall_top_k = similarity_top_k * base_multiplier
                # 对英文论文/报告标题类查询，进一步放大召回范围，避免被其它英文长文档「淹没」。
                if _looks_like_english_title_query(query_str):
                    recall_top_k = max(recall_top_k, similarity_top_k * 6)
                # 安全上限，避免在极大语料库上一次性召回过多候选
                recall_top_k = min(recall_top_k, 80)
                vector_retriever = index.as_retriever(similarity_top_k=recall_top_k)
                vector_nodes = await vector_retriever.aretrieve(query_str)

                vector_ms = (time.perf_counter() - t0) * 1000

                # ---- BM25 检索（仅混合模式）或 Rerank（纯向量时也可用）----
                final_nodes = vector_nodes  # 默认使用纯向量结果

                if hybrid_enabled:
                    t1 = time.perf_counter()
                    bm25, corpus_size = self._get_or_build_bm25(kb_id, recall_top_k)
                    if bm25 is not None:
                        try:
                            # 动态调整 top_k，确保不超过语料库大小
                            effective_top_k = min(recall_top_k, corpus_size)
                            if effective_top_k < recall_top_k:
                                bm25._similarity_top_k = effective_top_k
                                logger.info(
                                    f"BM25 top_k 动态调整: {recall_top_k} -> {effective_top_k} "
                                    f"(corpus_size={corpus_size})"
                                )
                            
                            bm25_nodes = bm25.retrieve(query_str)
                            bm25_ms = (time.perf_counter() - t1) * 1000

                            # ---- RRF 融合 ----
                            # 从配置读取 BM25 权重，允许动态调整关键词匹配的重要性，
                            # 并在特定查询（如英文论文标题）时临时提升 BM25 的相对权重。
                            bm25_weight = settings.RAG_BM25_WEIGHT
                            if _looks_like_english_title_query(query_str):
                                # 对英文标题/论文题目类查询，更依赖精确关键词匹配，
                                # 适度放大 BM25 的影响力，增强对标题/抬头的命中率。
                                bm25_weight = min(0.7, max(bm25_weight, 0.4))

                            fused_nodes = _reciprocal_rank_fusion(
                                vector_nodes, bm25_nodes, recall_top_k,
                                bm25_weight=bm25_weight,
                            )

                            # ---- Rerank 精排（若启用）或启发式加权重排 ----
                            # 说明：不在此处额外调用 get_rerank() 做布尔判断，避免同一次查询重复构造/重复日志；
                            # 实际是否可用由 _rerank_nodes 内部处理（未配置/不可用会自动降级为原序截断）。
                            if settings.RAG_RERANK_ENABLED:
                                rerank_top_k = min(settings.RAG_RERANK_TOP_K, similarity_top_k)
                                t_rerank = time.perf_counter()
                                final_nodes = self._rerank_nodes(
                                    fused_nodes, query_str, top_k=rerank_top_k
                                )
                                rerank_ms = (time.perf_counter() - t_rerank) * 1000
                                top_scores = [f"{n.score:.4f}" for n in final_nodes[:3]]
                                total_ms = (time.perf_counter() - t0) * 1000
                                logger.info(
                                    f"混合检索+Rerank: kb={kb_id}, "
                                    f"vector={len(vector_nodes)}({vector_ms:.0f}ms), "
                                    f"bm25={len(bm25_nodes)}({bm25_ms:.0f}ms), "
                                    f"rerank={len(final_nodes)}({rerank_ms:.0f}ms), "
                                    f"TopScores(Rerank)={top_scores}, total={total_ms:.0f}ms"
                                )
                            else:
                                # 启发式元数据/实体加权重排（Rerank 未启用时）
                                q_lower = query_str.lower()
                                boosted_nodes: list[NodeWithScore] = []
                                for nws in fused_nodes:
                                    boost = 0.0
                                    meta = getattr(nws.node, "metadata", {}) or {}
                                    title = str(meta.get("doc_title", "")).lower()
                                    fname = str(meta.get("file_name", "")).lower()

                                    if title and len(q_lower) > 10 and q_lower in title:
                                        boost += 0.5 * nws.score
                                    if fname and len(q_lower) > 10 and q_lower.replace(" ", "") in fname.replace(" ", ""):
                                        boost += 0.3 * nws.score

                                    content_sample = (getattr(nws.node, "text", "") or "").strip()[:200]
                                    if any(kw in q_lower for kw in ["谁", "名称", "是谁", "叫什么", "哪家"]):
                                        if any(pattern in content_sample for pattern in ["甲方:", "甲方：", "乙方:", "乙方：", "委托方", "受托方", "（甲方）", "（乙方）"]):
                                            boost += 0.4 * nws.score

                                    boosted_nodes.append(NodeWithScore(
                                        node=nws.node,
                                        score=nws.score + boost,
                                    ))

                                boosted_nodes.sort(key=lambda x: x.score, reverse=True)
                                final_nodes = boosted_nodes[:similarity_top_k]
                                top_scores = [f"{n.score:.4f}" for n in final_nodes[:3]]
                                total_ms = (time.perf_counter() - t0) * 1000
                                logger.info(
                                    f"混合检索: kb={kb_id}, "
                                    f"vector={len(vector_nodes)}({vector_ms:.0f}ms), "
                                    f"bm25={len(bm25_nodes)}({bm25_ms:.0f}ms), "
                                    f"fused={len(final_nodes)}, TopScores(RRF)={top_scores}, total={total_ms:.0f}ms"
                                )
                        except Exception as bm25_err:
                            # BM25 检索失败，降级为纯向量检索
                            logger.warning(
                                f"BM25 检索失败，降级为纯向量检索: kb={kb_id}, error={bm25_err}"
                            )
                    else:
                        # BM25 构建失败，降级为纯向量检索并截断/可选 Rerank
                        logger.warning(f"BM25 不可用，降级为纯向量检索: kb={kb_id}")
                        if settings.RAG_RERANK_ENABLED:
                            rerank_top_k = min(settings.RAG_RERANK_TOP_K, similarity_top_k)
                            final_nodes = self._rerank_nodes(
                                vector_nodes, query_str, top_k=rerank_top_k
                            )
                        else:
                            final_nodes = vector_nodes[:similarity_top_k]
                else:
                    # 纯向量模式：仍可启用 Rerank 精排
                    if settings.RAG_RERANK_ENABLED:
                        rerank_top_k = min(settings.RAG_RERANK_TOP_K, similarity_top_k)
                        t_rerank = time.perf_counter()
                        final_nodes = self._rerank_nodes(
                            vector_nodes, query_str, top_k=rerank_top_k
                        )
                        rerank_ms = (time.perf_counter() - t_rerank) * 1000
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            f"向量检索+Rerank: kb={kb_id}, "
                            f"vector={len(vector_nodes)}({vector_ms:.0f}ms), "
                            f"rerank={len(final_nodes)}({rerank_ms:.0f}ms), total={total_ms:.0f}ms"
                        )
                    else:
                        final_nodes = vector_nodes[:similarity_top_k]
                    logger.debug(f"向量检索: kb={kb_id}, results={len(final_nodes)}, elapsed={vector_ms:.0f}ms")

                # ---- 最低相关性阈值（可选）：知识库与问题域不符时避免返回无关片段 ----
                min_score = getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0) or 0.0
                if min_score > 0 and final_nodes:
                    best = max(nws.score for nws in final_nodes)
                    if best < min_score:
                        logger.info(
                            f"知识库 {kb_id} 最高相关分 {best:.4f} 低于阈值 {min_score}，跳过返回片段"
                        )
                        continue

                # ---- 格式化结果 ----
                # Log detailed results for debugging (preview first 100 chars)
                from memory.utils import is_low_quality_text
                
                for i, nws in enumerate(final_nodes):
                    content = nws.text.strip()
                    
                    # [CRITICAL] 知识库内容安检
                    if is_low_quality_text(content):
                        logger.warning(f"检测到 RAG 检索结果包含脏数据 (Score: {nws.score:.4f}, 已剔除): {content[:50]}...")
                        continue

                    # 获取元数据用于日志追踪
                    meta = getattr(nws.node, "metadata", {}) or {}
                    source_label = meta.get("doc_title") or meta.get("file_name") or "unknown_source"
                    
                    # Preview for logs: first 100 chars
                    preview = content[:100].replace('\n', ' ') + "..." if len(content) > 100 else content.replace('\n', ' ')
                    
                    logger.debug(f"  [Segment {segment_count}] Score: {nws.score:.4f} | Source: {source_label} | {preview}")

                    segment_header = f"[Knowledge Segment {segment_count}] Source: {source_label} | Score: {nws.score:.4f}"
                    all_segments.append(f"{segment_header}\n{content}")
                    segment_count += 1

            except Exception as e:
                logger.error(f"Error querying collection {kb_id}: {str(e)}")
                continue

        if not all_segments:
            # 有指定知识库但无任何片段（可能因阈值过滤或内容与问题域不符）时返回友好提示
            if kb_ids:
                return (
                    "未在知识库中找到与当前问题直接相关的内容，请确认知识库已上传相关文档或换一种问法。"
                )
            return ""

        return "\n\n---\n\n".join(all_segments)


# 单例实例
rag_service = RagService()
