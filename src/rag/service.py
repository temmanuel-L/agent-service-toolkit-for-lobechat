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
from typing import Optional, List, Any

# 屏蔽第三方库冗长的调试日志
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("bm25s").setLevel(logging.WARNING)

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import models as qdrant_models
from urllib.parse import urlparse

from core.settings import settings
from core.llm import get_embedding_model
from utils.log_utils import get_logger

# RAG 子模块：解析与知识库级元数据
from rag.parsing import parse_file_to_documents
from rag import kb_metadata
from rag.rerank import rerank_nodes
from rag.search.fusion import reciprocal_rank_fusion
from rag.search import hybrid_search_single_kb
from rag.schema.schema_search import SearchRequest

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
        from rag.chunking.embedding_batcher import TokenAwareEmbedding, CacheAwareEmbedding
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

        # ---- 向量索引 / docstore 缓存（按知识库维度）----
        # 说明：
        # - simple 策略下可选地重用 VectorStoreIndex，减少每次查询的 index 构建开销；
        # - parent_child 策略下，需要依赖 docstore 中的父子节点关系，
        #   才能在检索阶段将叶子命中提升为父节点上下文。
        # key=kb_id, value=VectorStoreIndex
        self._vector_indexes: dict[str, VectorStoreIndex] = {}

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

    # _rerank_nodes 方法已迁移到 rag.rerank.rerank_nodes，为保持兼容保留一个薄封装。
    def _rerank_nodes(
        self,
        nodes: list[NodeWithScore],
        query_str: str,
        top_k: int,
    ) -> list[NodeWithScore]:
        from rag.rerank import rerank_nodes as _rr

        return _rr(nodes, query_str, top_k)

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
            # 3. 使用解析子模块加载数据并注入统一文档级元数据（在分块之前完成）
            from rag.schema import DocumentMetadata  # 仅用于类型提示与日志，无运行时依赖

            documents, doc_meta = parse_file_to_documents(
                tmp_path,
                kb_id=kb_id,
                file_name=file_name,
                file_url=file_url,
            )

            if not documents:
                logger.warning(f"No content extracted from {file_name or file_url}")
                return 0

            if isinstance(doc_meta, DocumentMetadata) and doc_meta.doc_title:
                logger.info(f"文档标题: '{doc_meta.doc_title[:80]}'")

            logger.info(
                f"Extracted {len(documents)} document pages/segments. "
                f"Starting indexing into Qdrant collection: {collection_name}"
            )

            # 5. 配置 Qdrant 向量存储与分块转换
            # 注意：LlamaIndex 在写入（from_documents / VectorStoreIndex(...)）阶段会调用
            # 同步 client.create_collection，因此此处必须提供同步 QdrantClient；
            # 异步 AsyncQdrantClient 仅用于查询。
            vector_store = QdrantVectorStore(
                collection_name=collection_name,
                client=self.client,
                aclient=self.aclient,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            from rag.chunking import (
                build_chunking_transformations,
                build_parent_child_nodes,
            )

            strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()
            logger.info(
                "使用分块策略: strategy=%s, chunk_size=%d, chunk_overlap=%d",
                strategy,
                settings.RAG_CHUNK_SIZE,
                settings.RAG_CHUNK_OVERLAP,
            )

            # 6. 创建索引（解析 + 嵌入 + 写入）
            if strategy == "parent_child":
                # 教科书级 parent-child：
                # - 使用 HierarchicalNodeParser 生成父子两层节点；
                # - 将所有节点写入 docstore，便于检索阶段通过父子关系展开上下文；
                # - 仅对叶子节点建立向量索引，用于高精度语义检索。
                leaf_nodes, all_nodes = build_parent_child_nodes(documents)
                # 将父子节点全部注册到 docstore 中，保留完整层级关系
                storage_context.docstore.add_documents(all_nodes)

                index = VectorStoreIndex(
                    nodes=leaf_nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=False,
                )
            else:
                # simple 策略：沿用原有 SentenceSplitter 固定窗口分块行为
                transformations = build_chunking_transformations()
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    transformations=transformations,
                    show_progress=False,  # 生产环境设为 False 以保持日志简洁
                )

            # 缓存索引，便于检索阶段在 parent_child 模式下访问 docstore
            self._vector_indexes[collection_name] = index

            # 7. 失效 BM25 缓存，下次查询时自动重建
            self.invalidate_bm25_cache(kb_id)

            # 8. 保存知识库元数据到 PostgreSQL
            try:
                # 获取当前已存在的文件列表
                existing_metadata = await kb_metadata.get_kb_metadata(kb_id)
                if existing_metadata:
                    # 追加新文件
                    await kb_metadata.add_file_to_kb_metadata(
                        kb_id, file_name or "unknown", file_url
                    )
                else:
                    # 新建元数据记录
                    await kb_metadata.save_kb_metadata(
                        kb_id, [file_name or "unknown"], [file_url]
                    )
            except Exception as meta_err:
                # 元数据保存失败不应阻塞主流程，仅记录警告
                logger.warning(f"Failed to save KB metadata (non-fatal): {meta_err}")

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
        """
        删除知识库：同步删除 Qdrant Collection、BM25缓存和 PostgreSQL 元数据。
        """
        try:
            logger.info(f"Starting delete_knowledge_base: kb_id={kb_id}")
            
            # 1. 删除 Qdrant Collection
            if self.client.collection_exists(kb_id):
                logger.info(f"Deleting Qdrant collection: {kb_id}")
                self.client.delete_collection(kb_id)
                logger.info(f"Successfully deleted Qdrant collection: {kb_id}")
            else:
                logger.warning(f"Collection {kb_id} does not exist in Qdrant")

            # 2. 清除 BM25 缓存
            logger.info(f"Invalidating BM25 cache for kb_id={kb_id}")
            self.invalidate_bm25_cache(kb_id)

            # 3. 删除 PostgreSQL 元数据
            try:
                logger.info(f"Deleting KB metadata from PostgreSQL: kb_id={kb_id}")
                await kb_metadata.delete_kb_metadata(kb_id)
                logger.info(f"Successfully deleted KB metadata from PostgreSQL: kb_id={kb_id}")
            except Exception as meta_err:
                # 元数据删除失败不应阻塞主流程，仅记录警告
                logger.warning(f"Failed to delete KB metadata in PG (non-fatal): {meta_err}")

            logger.info(f"delete_knowledge_base completed successfully: kb_id={kb_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting knowledge base {kb_id}: {str(e)}", exc_info=True)
            return False

    # ================================================================
    # 知识库文件删除
    # ================================================================
    async def delete_file_from_knowledge_base(self, kb_id: str, file_name: str) -> bool:
        """
        删除知识库中的指定文件：删除 Qdrant 中该文件的 chunks、BM25缓存失效、更新 PostgreSQL 元数据。

        Args:
            kb_id: 知识库ID
            file_name: 要删除的文件名

        Returns:
            bool: 操作是否成功
        """
        try:
            logger.info(f"Starting delete_file_from_knowledge_base: kb_id={kb_id}, file_name={file_name}")
            
            # 1. 从 Qdrant 中删除该文件的 points (通过 metadata filter)
            if self.client.collection_exists(kb_id):
                logger.info(f"Deleting file from Qdrant collection: kb_id={kb_id}, file_name={file_name}")
                # 使用 filter 直接删除特定 file_name 的 points
                try:
                    self.client.delete(
                        collection_name=kb_id,
                        points_selector=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="file_name",
                                    match=qdrant_models.MatchValue(value=file_name)
                                )
                            ]
                        )
                    )
                    logger.info(f"Successfully deleted points from Qdrant for file: {file_name}")
                except Exception as delete_err:
                    logger.warning(f"Failed to delete points from Qdrant: {delete_err}")
            else:
                logger.warning(f"Collection {kb_id} does not exist in Qdrant")

            # 2. 清除 BM25 缓存
            logger.info(f"Invalidating BM25 cache for kb_id={kb_id}")
            self.invalidate_bm25_cache(kb_id)

            # 3. 从 PostgreSQL 元数据中删除文件记录
            try:
                logger.info(f"Deleting file from KB metadata in PostgreSQL: kb_id={kb_id}, file_name={file_name}")
                await kb_metadata.delete_file_from_kb_metadata(kb_id, file_name)
                logger.info(f"Successfully deleted file from KB metadata in PostgreSQL: {file_name}")
            except Exception as meta_err:
                logger.warning(f"Failed to delete file from KB metadata in PG (non-fatal): {meta_err}")

            logger.info(f"delete_file_from_knowledge_base completed: kb_id={kb_id}, file_name={file_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from knowledge base: kb_id={kb_id}, file_name={file_name}, error={str(e)}", exc_info=True)
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
        chunk_strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()
        mode_label = "hybrid(vector+BM25)" if hybrid_enabled else "vector-only"
        logger.info(
            f"知识库检索: query='{query_str[:50]}...', kb_ids={kb_ids}, "
            f"top_k={similarity_top_k}, mode={mode_label}"
        )

        # ---- 可选 HyDE Query 改写（仅影响向量检索，不改变原始 query 日志与 BM25 查询）----
        effective_query = query_str
        if getattr(settings, "RAG_HYDE_ENABLED", False) and getattr(settings, "RAG_HYDE_NUM_VARIANTS", 0) > 0:
            try:
                from rag.HyDE import generate_hyde_variants

                hyde_result = await generate_hyde_variants(
                    query_str,
                    num_variants=settings.RAG_HYDE_NUM_VARIANTS,
                )
                if hyde_result.variants:
                    # 暂时采用首条改写作为向量检索查询，后续可在 search 模块内部支持多路改写融合
                    effective_query = hyde_result.variants[0].text
                    logger.info(
                        "HyDE 已启用，使用首条改写作为向量检索查询，示例: %s",
                        effective_query[:80].replace("\n", " "),
                    )
            except Exception as hyde_err:
                logger.warning(f"HyDE 改写失败，降级为原始查询: {hyde_err}")

        # ---- 可选 Query → filters 推断（仅用于 metadata 级过滤）----
        inferred_filters: dict[str, Any] = {}
        if getattr(settings, "RAG_QUERY_FILTER_INFERENCE_ENABLED", False):
            try:
                from rag.search.filters import infer_filters_from_query

                inferred_filters = infer_filters_from_query(query_str)
                if inferred_filters:
                    logger.info("Query filter inference enabled, inferred filters: %s", inferred_filters)
            except Exception as filter_err:
                logger.warning(f"Query filter inference failed, skip filters: {filter_err}")

        all_segments: list[str] = []
        segment_count = 1

        for kb_id in kb_ids:
            try:
                # 检查 collection 是否存在
                if not self.client.collection_exists(kb_id):
                    logger.debug(f"Collection {kb_id} does not exist, skipping query.")
                    continue

                t0 = time.perf_counter()

                if chunk_strategy == "parent_child":
                    # ---- parent_child：基于父子分块的检索，返回父节点上下文 ----
                    # 1) 获取或构建向量索引（需包含 docstore，以保留父子关系）
                    index = self._vector_indexes.get(kb_id)
                    if index is None:
                        # 回退：仅从向量存储构建索引（可能缺失父子关系信息）
                        vector_store = QdrantVectorStore(
                            collection_name=kb_id,
                            aclient=self.aclient,
                            client=None,
                        )
                        index = VectorStoreIndex.from_vector_store(
                            vector_store=vector_store,
                            embed_model=self.embed_model,
                        )
                        self._vector_indexes[kb_id] = index

                    # 第一阶段召回使用「放大的 top_k」
                    base_multiplier = 3
                    recall_top_k = similarity_top_k * base_multiplier
                    if _looks_like_english_title_query(effective_query):
                        recall_top_k = max(recall_top_k, similarity_top_k * 6)
                    recall_top_k = min(recall_top_k, 80)

                    # 2) 向量检索（叶子级别）
                    retriever = index.as_retriever(similarity_top_k=recall_top_k)
                    vector_nodes = await retriever.aretrieve(effective_query)

                    # 3) 可选 BM25 检索（仍在叶子级别）
                    bm25_nodes: list[NodeWithScore] | None = None
                    if hybrid_enabled:
                        bm25, corpus_size = self._get_or_build_bm25(kb_id, recall_top_k)
                        if bm25 is not None and corpus_size > 0:
                            bm25_nodes = bm25.retrieve(effective_query)

                    # 4) 向量 + BM25 融合（叶子级别），保持与 simple 策略一致的加权 RRF 逻辑
                    fused_nodes = vector_nodes
                    if hybrid_enabled and bm25_nodes is not None:
                        bm25_weight = settings.RAG_BM25_WEIGHT
                        fused_nodes = reciprocal_rank_fusion(
                            vector_nodes,
                            bm25_nodes,
                            recall_top_k,
                            bm25_weight=bm25_weight,
                        )
                        fused_nodes = fused_nodes[:similarity_top_k]
                    else:
                        fused_nodes = vector_nodes[:similarity_top_k]

                    vector_ms = (time.perf_counter() - t0) * 1000

                    # 5) Rerank 或启发式重排（仍在叶子级别）
                    final_leaf_nodes = fused_nodes
                    if settings.RAG_RERANK_ENABLED:
                        rerank_top_k = min(settings.RAG_RERANK_TOP_K, similarity_top_k)
                        t_rerank = time.perf_counter()
                        final_leaf_nodes = self._rerank_nodes(
                            fused_nodes, query_str, top_k=rerank_top_k
                        )
                        rerank_ms = (time.perf_counter() - t_rerank) * 1000
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索+Rerank(parent_child): kb=%s, vector=%d(%.0fms), rerank=%d(%.0fms), total=%.0fms",
                            kb_id,
                            len(fused_nodes),
                            vector_ms,
                            len(final_leaf_nodes),
                            rerank_ms,
                            total_ms,
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
                        final_leaf_nodes = boosted_nodes[:similarity_top_k]
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索(启发式重排, parent_child): kb=%s, results=%d, total=%.0fms",
                            kb_id,
                            len(final_leaf_nodes),
                            total_ms,
                        )

                    # 6) 依据最低相关性阈值过滤
                    min_score = getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0) or 0.0
                    if min_score > 0 and final_leaf_nodes:
                        best = max(nws.score for nws in final_leaf_nodes)
                        if best < min_score:
                            logger.info(
                                f"知识库 {kb_id} 最高相关分 {best:.4f} 低于阈值 {min_score}，跳过返回片段"
                            )
                            continue

                    # 7) 将叶子命中提升为父节点上下文
                    try:
                        from llama_index.core.schema import NodeRelationship
                    except Exception:
                        NodeRelationship = None  # type: ignore

                    parent_nodes_map: dict[str, NodeWithScore] = {}
                    docstore = getattr(index, "storage_context", None)
                    docstore = getattr(docstore, "docstore", None)

                    for nws in final_leaf_nodes:
                        node = nws.node
                        parent_id = None
                        parent_node = None

                        if NodeRelationship is not None and docstore is not None:
                            # 不同版本的 LlamaIndex 中 relationships 结构略有差异：
                            # - 有的返回单个 RelatedNodeInfo
                            # - 有的返回 RelatedNodeInfo 列表
                            rels = getattr(node, "relationships", {}) or {}
                            parent_rel = rels.get(NodeRelationship.PARENT) if rels else None
                            if parent_rel:
                                # 如果是列表，取第一个；否则直接使用对象本身
                                if isinstance(parent_rel, list):
                                    parent_rel = parent_rel[0] if parent_rel else None
                                candidate_id = getattr(parent_rel, "node_id", None)
                                if candidate_id:
                                    try:
                                        parent_node = docstore.get_node(candidate_id)
                                        parent_id = candidate_id
                                    except Exception:
                                        parent_node = None

                        if parent_node is None:
                            # 找不到父节点时退化为使用自身
                            parent_node = node
                            parent_id = getattr(node, "node_id", None) or str(id(node))

                        existing = parent_nodes_map.get(parent_id)
                        score = float(nws.score or 0.0)
                        if existing is None or score > existing.score:
                            parent_nodes_map[parent_id] = NodeWithScore(
                                node=parent_node,
                                score=score,
                            )

                    parent_nodes = sorted(
                        parent_nodes_map.values(),
                        key=lambda x: x.score,
                        reverse=True,
                    )

                    # 8) 格式化结果（此时每个节点已经是父级上下文）
                    from rag.postprocess.segments import build_segments_from_nodes

                    segments, segment_count = build_segments_from_nodes(
                        parent_nodes,
                        start_index=segment_count,
                    )
                    all_segments.extend(segments)
                else:
                    # ---- simple：沿用原有单层分块 + hybrid_search_single_kb 逻辑 ----
                    # 构造向量存储
                    vector_store = QdrantVectorStore(
                        collection_name=kb_id,
                        aclient=self.aclient,
                        client=None,
                    )

                    # 第一阶段召回使用「放大的 top_k」
                    base_multiplier = 3
                    recall_top_k = similarity_top_k * base_multiplier
                    if _looks_like_english_title_query(effective_query):
                        recall_top_k = max(recall_top_k, similarity_top_k * 6)
                    recall_top_k = min(recall_top_k, 80)

                    # ---- BM25 构造（可选）----
                    bm25 = None
                    corpus_size = 0
                    if hybrid_enabled:
                        bm25, corpus_size = self._get_or_build_bm25(kb_id, recall_top_k)

                    # ---- 调用 search 子模块执行单库检索 ----
                    search_req = SearchRequest(
                        query=effective_query,
                        kb_ids=[kb_id],
                        top_k=recall_top_k,
                        filters=inferred_filters,
                    )

                    search_result = await hybrid_search_single_kb(
                        search_req,
                        kb_id=kb_id,
                        vector_store=vector_store,
                        embed_model=self.embed_model,
                        bm25_retriever=bm25 if hybrid_enabled and bm25 is not None else None,
                        corpus_size=corpus_size,
                    )

                    # 转换回 NodeWithScore 列表，后续沿用原有 rerank / 阈值 / 拼装逻辑
                    from llama_index.core.schema import TextNode

                    vector_nodes: list[NodeWithScore] = []
                    for hit in search_result.hits:
                        node = TextNode(text=hit.text, metadata=hit.metadata)
                        vector_nodes.append(NodeWithScore(node=node, score=hit.score))

                    vector_ms = (time.perf_counter() - t0) * 1000

                    # ---- Rerank 或启发式重排 ----
                    final_nodes = vector_nodes
                    if settings.RAG_RERANK_ENABLED:
                        rerank_top_k = min(settings.RAG_RERANK_TOP_K, similarity_top_k)
                        t_rerank = time.perf_counter()
                        final_nodes = self._rerank_nodes(
                            vector_nodes, query_str, top_k=rerank_top_k
                        )
                        rerank_ms = (time.perf_counter() - t_rerank) * 1000
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索+Rerank: kb=%s, vector=%d(%.0fms), rerank=%d(%.0fms), total=%.0fms",
                            kb_id,
                            len(vector_nodes),
                            vector_ms,
                            len(final_nodes),
                            rerank_ms,
                            total_ms,
                        )
                    else:
                        # 启发式元数据/实体加权重排（Rerank 未启用时）
                        q_lower = query_str.lower()
                        boosted_nodes: list[NodeWithScore] = []
                        for nws in vector_nodes:
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
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索(启发式重排): kb=%s, results=%d, total=%.0fms",
                            kb_id,
                            len(final_nodes),
                            total_ms,
                        )

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
                    from rag.postprocess.segments import build_segments_from_nodes

                    segments, segment_count = build_segments_from_nodes(
                        final_nodes,
                        start_index=segment_count,
                    )
                    all_segments.extend(segments)

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
