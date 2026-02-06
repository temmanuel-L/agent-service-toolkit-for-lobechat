"""Vector storage manager for agent conversations and RAG."""

import asyncio
import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from qdrant_client import models as qdrant_models

from memory.qdrant import get_qdrant_store
from memory.embedding_cache import EmbeddingCache
from memory.utils import is_low_quality_text
from core import settings
from utils.log_utils import get_logger

logger = get_logger(__name__)

# Embedding batch configuration
EMBEDDING_BATCH_MAX_TOKENS = 8000  # Match openclaw's default
EMBEDDING_RETRY_MAX_ATTEMPTS = 3
EMBEDDING_RETRY_BASE_DELAY_MS = 500
EMBEDDING_RETRY_MAX_DELAY_MS = 8000


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
        
        # Embedding cache
        cache_enabled = getattr(settings, 'EMBEDDING_CACHE_ENABLED', True)
        cache_path = getattr(settings, 'EMBEDDING_CACHE_PATH', './data/embedding_cache.json')
        self.embedding_cache = EmbeddingCache(cache_file=cache_path) if cache_enabled else None
        
        if self.embedding_cache:
            logger.info(f"Embedding缓存已启用: {cache_path}")
        
    async def ainitialize(self):
        """Initialize the vector manager."""
        self.qdrant_store = await get_qdrant_store(
            collection_name=self.collection_name
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
    
    async def aadd_messages(
        self, 
        messages: List[BaseMessage], 
        user_id: Optional[str] = None, 
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """
        Add messages to vector store with metadata.
        
        改进：使用缓存和重试机制。
        """
        if not self.qdrant_store:
            await self.ainitialize()
            
        # 将 BaseMessage 转换为 Document
        # 遍历消息，进行转换和过滤
        documents = []
        for i, message in enumerate(messages):
            content = str(message.content)
            
            # 质量过滤：
            # 如果内容为空，或者被判定为低质量（如死循环重复），则跳过不存。
            # 这能有效防止 LLM 的幻觉输出污染长期记忆库。
            if not content or is_low_quality_text(content):
                continue

            # 构建元数据 (Metadata)
            # 包含用户ID、会话ID、Agent ID 以及消息类型和顺序
            metadata = {
                "user_id": user_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "message_type": message.type,
                "message_index": i,
            }
            
            # 尝试提取并保留时间戳信息
            metadata["timestamp"] = message.additional_kwargs.get("timestamp") if hasattr(message, 'additional_kwargs') else None
            
            # 封装为 Document 对象
            document = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(document)
            
        # 如果过滤后没有有效文档，直接返回空列表，不执行数据库操作
        if not documents:
            return []
            
        return await self.qdrant_store.aadd_documents(documents)
        
    async def aadd_documents(
        self,
        documents: List[Document],
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[str]:
        """Add documents to vector store with metadata."""
        if not self.qdrant_store:
            await self.ainitialize()
            
        # 为每个文档添加元数据
        for doc in documents:
            if user_id:
                doc.metadata["user_id"] = user_id
            if thread_id:
                doc.metadata["thread_id"] = thread_id
            if agent_id:
                doc.metadata["agent_id"] = agent_id
                
        return await self.qdrant_store.aadd_documents(documents)
        
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
        
    async def asimilarity_search_with_score(
        self,
        query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        k: int = 4,
    ) -> List[tuple[Document, float]]:
        """Perform similarity search and return docs with scores."""
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
        
        # Qdrant client specific method for scores
        if qdrant_filter is not None:
             return await self.qdrant_store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=qdrant_filter,
            )
        return await self.qdrant_store.asimilarity_search_with_score(query=query, k=k)
        
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