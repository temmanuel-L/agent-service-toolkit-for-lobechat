"""Vector storage manager for agent conversations and RAG."""

import asyncio
import time
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from qdrant_client import models as qdrant_models

from memory.qdrant import get_qdrant_store
from memory.embedding_cache import get_embedding_cache
from memory.utils import is_low_quality_text
from core.llm import get_embedding_model
from core import settings
from utils.log_utils import get_logger

logger = get_logger(__name__)

# LangChain Qdrant 写入的 payload 键名，与 asimilarity_search 读回一致
_CONTENT_PAYLOAD_KEY = "page_content"
_METADATA_PAYLOAD_KEY = "metadata"

# Embedding batch configuration
EMBEDDING_BATCH_MAX_TOKENS = 8000  # Match openclaw's default
EMBEDDING_RETRY_MAX_ATTEMPTS = 3
EMBEDDING_RETRY_BASE_DELAY_MS = 500
EMBEDDING_RETRY_MAX_DELAY_MS = 8000


class _SyncToAsyncEmbedding:
    """为仅支持同步的 embedding 模型提供 aembed_documents，便于统一走 _embed_with_cache_and_retry。"""

    def __init__(self, sync_embedding):
        self._sync = sync_embedding

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._sync.embed_documents, texts)


class VectorManager:
    """
    Manages vector storage for agent conversations and RAG.
    
    改进：
    - Embedding缓存：避免重复embedding同样内容
    - 自适应批处理：基于token限制动态调整batch大小
    - 重试机制：指数退避处理临时失败
    """
    
    def __init__(self, collection_name: str = "agent_conversations"):
        self.collection_name = collection_name
        self.qdrant_store = None
        
        # 与 RAG 共用同一 EmbeddingCache 单例
        self.embedding_cache = get_embedding_cache()
        if self.embedding_cache:
            logger.info("Embedding 缓存已启用（与 RAG 共用）")
        
    async def ainitialize(self):
        """Initialize the vector manager. 使用带缓存的 embedder，检索 query 走缓存以提速。"""
        from memory.embedding_cache import get_cache_aware_embedding
        self.qdrant_store = await get_qdrant_store(
            collection_name=self.collection_name,
            embeddings=get_cache_aware_embedding(),
        )
        
    async def _embed_with_cache_and_retry(self, texts: list[str], embed_model) -> list[list[float]]:
        """
        带缓存和重试的embedding。
        
        流程：
        1. 检查缓存
        2. 未缓存的文本调用embedding API（带重试）
        3. 将结果写入缓存
        """
        results: list[Optional[list[float]]] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)
        
        # Step 1: Check cache
        if self.embedding_cache:
            for i, text in enumerate(texts):
                cached = await self.embedding_cache.get(text)
                if cached:
                    results[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = [(i, t) for i, t in enumerate(texts)]
        
        if not texts_to_embed:
            logger.debug(f"Embedding全部命中缓存: {len(texts)} 条")
            return results
        
        # Step 2: Embed uncached texts with retry
        indices, uncached_texts = zip(*texts_to_embed)
        
        for attempt in range(EMBEDDING_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.perf_counter()
                embeddings = await embed_model.aembed_documents(uncached_texts)
                elapsed = (time.perf_counter() - t0) * 1000
                
                logger.debug(
                    f"Embedding完成: {len(uncached_texts)} 条, "
                    f"elapsed={elapsed:.0f}ms, cached={len(texts) - len(uncached_texts)}"
                )
                
                # Step 3: Update cache and results
                for idx, emb, text in zip(indices, embeddings, uncached_texts):
                    results[idx] = emb
                    if self.embedding_cache:
                        await self.embedding_cache.put(text, emb)
                
                return results
            
            except Exception as exc:
                if attempt < EMBEDDING_RETRY_MAX_ATTEMPTS - 1:
                    # Exponential backoff
                    delay = min(
                        EMBEDDING_RETRY_BASE_DELAY_MS * (2 ** attempt),
                        EMBEDDING_RETRY_MAX_DELAY_MS
                    ) / 1000.0
                    logger.warning(
                        f"Embedding失败 (attempt {attempt + 1}/{EMBEDDING_RETRY_MAX_ATTEMPTS}): {exc}, "
                        f"retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Embedding最终失败: {exc}")
                    raise
        
        return results

    def _ensure_collection_and_upsert(
        self,
        client,
        collection_name: str,
        points: list,
        vector_size: int,
    ) -> None:
        """
        同步：若 collection 不存在则创建，再 upsert points。
        供在 asyncio.to_thread 中调用，避免阻塞事件循环。
        """
        try:
            client.get_collection(collection_name)
        except Exception as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logger.info("已创建 Qdrant 集合：%s (维度: %d)", collection_name, vector_size)
            else:
                raise
        client.upsert(collection_name=collection_name, points=points)

    async def _add_vectors_to_qdrant(
        self, documents: List[Document], vectors: List[List[float]]
    ) -> List[str]:
        """
        将已算好向量的文档写入 Qdrant，payload 与 LangChain 约定一致，便于 asimilarity_search 读回。
        """
        if not documents or not vectors or len(documents) != len(vectors):
            return []
        client = getattr(self.qdrant_store, "client", None)
        if not client:
            # 降级：无 client 时仍走 store 的 aadd_documents（内部 embed，无缓存）
            return await self.qdrant_store.aadd_documents(documents)
        points = []
        for doc, vec in zip(documents, vectors):
            payload = {
                _CONTENT_PAYLOAD_KEY: doc.page_content,
                _METADATA_PAYLOAD_KEY: doc.metadata,
            }
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                )
            )
        vector_size = len(vectors[0])
        await asyncio.to_thread(
            self._ensure_collection_and_upsert,
            client,
            self.collection_name,
            points,
            vector_size,
        )
        return [p.id for p in points]
    
    async def aadd_messages(
        self, 
        messages: List[BaseMessage], 
        user_id: Optional[str] = None, 
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Add messages to vector store with metadata.
        先经 EmbeddingCache 做 embedding，再写入 Qdrant，与 RAG 共用缓存以提速。
        """
        if not self.qdrant_store:
            await self.ainitialize()
            
        documents = []
        for i, message in enumerate(messages):
            content = str(message.content)
            if not content or is_low_quality_text(content):
                continue
            metadata = {
                "user_id": user_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "message_type": message.type,
                "message_index": i,
            }
            metadata["timestamp"] = (
                message.additional_kwargs.get("timestamp")
                if hasattr(message, "additional_kwargs")
                else None
            )
            documents.append(Document(page_content=content, metadata=metadata))
            
        if not documents:
            return []

        texts = [d.page_content for d in documents]
        embed_model = get_embedding_model()
        if not hasattr(embed_model, "aembed_documents"):
            embed_model = _SyncToAsyncEmbedding(embed_model)
        vectors = await self._embed_with_cache_and_retry(texts, embed_model)
        return await self._add_vectors_to_qdrant(documents, vectors)
        
    async def aadd_documents(
        self,
        documents: List[Document],
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """Add documents to vector store with metadata. 先经缓存 embedding 再写入 Qdrant。"""
        if not self.qdrant_store:
            await self.ainitialize()
        for doc in documents:
            if user_id:
                doc.metadata["user_id"] = user_id
            if thread_id:
                doc.metadata["thread_id"] = thread_id
            if agent_id:
                doc.metadata["agent_id"] = agent_id
        if not documents:
            return []
        texts = [d.page_content for d in documents]
        embed_model = get_embedding_model()
        if not hasattr(embed_model, "aembed_documents"):
            embed_model = _SyncToAsyncEmbedding(embed_model)
        vectors = await self._embed_with_cache_and_retry(texts, embed_model)
        return await self._add_vectors_to_qdrant(documents, vectors)
        
    async def asimilarity_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        k: int = 4,
    ) -> List[Document]:
        """Perform similarity search with optional filters."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        # 构建 Qdrant 过滤器（metadata 前缀）
        conditions = []
        if user_id:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.user_id",
                    match=qdrant_models.MatchValue(value=user_id),
                )
            )
        if thread_id:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.thread_id",
                    match=qdrant_models.MatchValue(value=thread_id),
                )
            )
        if agent_id:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.agent_id",
                    match=qdrant_models.MatchValue(value=agent_id),
                )
            )

        qdrant_filter = qdrant_models.Filter(must=conditions) if conditions else None
        if qdrant_filter is not None:
            return await self.qdrant_store.asimilarity_search(
                query=query,
                k=k,
                filter=qdrant_filter,
            )
        return await self.qdrant_store.asimilarity_search(query=query, k=k)

    async def asimilarity_search_by_vector(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        k: int = 4,
    ) -> List[Document]:
        """
        按已算好的 query 向量检索，不再调用 embedding API。供长期记忆复用 query embedding 以省一次请求。
        """
        if not self.qdrant_store:
            await self.ainitialize()
        client = getattr(self.qdrant_store, "client", None)
        if not client:
            return []
        conditions = []
        if user_id:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.user_id",
                    match=qdrant_models.MatchValue(value=user_id),
                )
            )
        if thread_id:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.thread_id",
                    match=qdrant_models.MatchValue(value=thread_id),
                )
            )
        if agent_id:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.agent_id",
                    match=qdrant_models.MatchValue(value=agent_id),
                )
            )
        qdrant_filter = qdrant_models.Filter(must=conditions) if conditions else None

        def _search() -> List[Document]:
            res = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            docs = []
            for pt in res:
                payload = pt.payload or {}
                content = payload.get(_CONTENT_PAYLOAD_KEY, "")
                meta = payload.get(_METADATA_PAYLOAD_KEY, {})
                docs.append(Document(page_content=content, metadata=meta))
            return docs

        return await asyncio.to_thread(_search)

    async def acleanup_user_data(self, user_id: str):
        """Clean up all vector data for a specific user."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        await self.qdrant_store.adelete_points_by_metadata({"user_id": user_id})
        logger.info(f"Cleaned up vector data for user: {user_id}")
        
    async def acleanup_thread_data(self, thread_id: str):
        """Clean up all vector data for a specific thread."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        await self.qdrant_store.adelete_points_by_metadata({"thread_id": thread_id})
        logger.info(f"Cleaned up vector data for thread: {thread_id}")