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
from typing import Optional, List

from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from urllib.parse import urlparse

from core.settings import settings
from core.llm import get_embedding_model
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
# 注意：增大这些参数会增加 LLM 的输入 token 消耗
# 默认值 512/64 适配大多数 Ollama embedding 模型（如 nomic-embed-text 上下文约 8192 tokens）
# 中文字符每个约 1-2 tokens，512 字符 ~= 500-1000 tokens，留足余量
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
RAG_DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "8"))

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
                        payload = record.payload or {}
                        if "text" in payload:
                            text = payload["text"]
                        elif "_node_content" in payload:
                            import json
                            try:
                                nc = json.loads(payload["_node_content"])
                                text = nc.get("text", "")
                            except (json.JSONDecodeError, TypeError):
                                pass
                        if text:
                            all_nodes.append(TextNode(
                                text=text,
                                id_=str(record.id),
                            ))

                    if next_offset is None or len(records) < _BM25_SCROLL_PAGE_SIZE:
                        break
                    offset = next_offset

                if not all_nodes:
                    logger.warning(f"BM25: collection '{kb_id}' 中无可用节点")
                    return None, 0

                from llama_index.retrievers.bm25 import BM25Retriever

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

            # 4. 提取文档标题并注入所有 chunk 的元数据
            #    这使得每个 chunk 的 embedding 都包含文档标题，
            #    解决标题页文本短小在向量检索中排名低的问题
            doc_title = _extract_doc_title(tmp_path, documents, file_name)

            for doc in documents:
                if doc_title:
                    doc.metadata["doc_title"] = doc_title
                if file_name:
                    doc.metadata["file_name"] = file_name
                # 确保元数据参与 embedding 和 LLM 上下文（不排除任何 key）
                doc.excluded_embed_metadata_keys = []
                doc.excluded_llm_metadata_keys = []

            if doc_title:
                logger.info(f"文档标题: '{doc_title[:80]}'")

            logger.info(
                f"Extracted {len(documents)} document pages/segments. "
                f"Starting indexing into Qdrant collection: {collection_name}"
            )

            # 5. 配置 Qdrant 向量存储与分块转换
            vector_store = QdrantVectorStore(
                collection_name=collection_name,
                client=self.client,
                aclient=self.aclient
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 使用 tiktoken 进行基于 Token 的精准分块
            # 相比默认的基于字符分块，这能确保每个 chunk 严格适配 Embedding 模型的上下文窗口
            splitter = SentenceSplitter(
                chunk_size=RAG_CHUNK_SIZE, 
                chunk_overlap=RAG_CHUNK_OVERLAP,
                tokenizer=tiktoken.get_encoding("cl100k_base").encode
            )
            transformations = [splitter]
            logger.info(
                f"使用基于Token的分块: chunk_size={RAG_CHUNK_SIZE}, "
                f"chunk_overlap={RAG_CHUNK_OVERLAP} (cl100k_base)"
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
            similarity_top_k: 返回的最相关文本块数量，默认使用 RAG_DEFAULT_TOP_K

        Returns:
            str: 格式化的检索结果，每个段落带有 [Knowledge Segment N] 标记
        """
        if similarity_top_k is None:
            similarity_top_k = RAG_DEFAULT_TOP_K
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
                    client=self.client,
                    aclient=self.aclient,
                )
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=self.embed_model,
                )
                vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
                vector_nodes = await vector_retriever.aretrieve(query_str)

                vector_ms = (time.perf_counter() - t0) * 1000

                # ---- BM25 检索（仅混合模式）----
                final_nodes = vector_nodes  # 默认使用纯向量结果

                if hybrid_enabled:
                    t1 = time.perf_counter()
                    bm25, corpus_size = self._get_or_build_bm25(kb_id, similarity_top_k)
                    if bm25 is not None:
                        try:
                            # 动态调整 top_k，确保不超过语料库大小
                            effective_top_k = min(similarity_top_k, corpus_size)
                            if effective_top_k < similarity_top_k:
                                bm25._similarity_top_k = effective_top_k
                                logger.info(
                                    f"BM25 top_k 动态调整: {similarity_top_k} -> {effective_top_k} "
                                    f"(corpus_size={corpus_size})"
                                )
                            
                            bm25_nodes = bm25.retrieve(query_str)
                            bm25_ms = (time.perf_counter() - t1) * 1000

                            # ---- RRF 融合 ----
                            # 从配置读取 BM25 权重，允许动态调整关键词匹配的重要性
                            bm25_weight = settings.RAG_BM25_WEIGHT
                            final_nodes = _reciprocal_rank_fusion(
                                vector_nodes, bm25_nodes, similarity_top_k, 
                                bm25_weight=bm25_weight
                            )
                            total_ms = (time.perf_counter() - t0) * 1000
                            logger.info(
                                f"混合检索: kb={kb_id}, "
                                f"vector={len(vector_nodes)}({vector_ms:.0f}ms), "
                                f"bm25={len(bm25_nodes)}({bm25_ms:.0f}ms), "
                                f"fused={len(final_nodes)}, total={total_ms:.0f}ms"
                            )
                        except Exception as bm25_err:
                            # BM25 检索失败，降级为纯向量检索
                            logger.warning(
                                f"BM25 检索失败，降级为纯向量检索: kb={kb_id}, error={bm25_err}"
                            )
                    else:
                        # BM25 构建失败，降级为纯向量检索
                        logger.warning(f"BM25 不可用，降级为纯向量检索: kb={kb_id}")
                else:
                    logger.debug(f"向量检索: kb={kb_id}, results={len(vector_nodes)}, elapsed={vector_ms:.0f}ms")

                # ---- 格式化结果 ----
                # Log detailed results for debugging (preview first 100 chars)
                from memory.utils import is_low_quality_text
                
                for i, nws in enumerate(final_nodes):
                    content = nws.text.strip()
                    
                    # [CRITICAL] 知识库内容安检
                    # 防止检索回来的文档块本身包含死循环或乱码（如 corrupted author list）
                    if is_low_quality_text(content):
                        logger.warning(f"检测到 RAG 检索结果包含脏数据 (Score: {nws.score:.4f}, 已剔除): {content[:50]}...")
                        continue

                    # Preview for logs: first 100 chars, replace newlines
                    preview = content[:100].replace('\n', ' ') + "..." if len(content) > 100 else content.replace('\n', ' ')
                    
                    logger.info(f"  [Segment {segment_count}] Score: {nws.score:.4f} | {preview}")

                    segment_header = f"[Knowledge Segment {segment_count}]"
                    all_segments.append(f"{segment_header}\n{content}")
                    segment_count += 1

            except Exception as e:
                logger.error(f"Error querying collection {kb_id}: {str(e)}")
                continue

        if not all_segments:
            return ""

        return "\n\n---\n\n".join(all_segments)


# 单例实例
rag_service = RagService()
