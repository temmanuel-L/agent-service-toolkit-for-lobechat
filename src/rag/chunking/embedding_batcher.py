"""
Embedding 包装器：Token 批处理 + 缓存

注意：
- 虽然主要职责是优化 embedding 调用，但它与分块策略紧密相关，
  因为需要根据 token 数控制单批大小，避免长文档在向量化时溢出。
- 因此将其放在 rag.chunking 子模块中，作为 ingest/indexing 流水线的一部分。
"""

from typing import List, Any, Optional

import tiktoken
from langchain_core.embeddings import Embeddings

from utils.log_utils import get_logger

logger = get_logger(__name__)


class CacheAwareEmbedding(Embeddings):
    """
    在现有 EmbeddingCache 之上的包装器：先查缓存，未命中再调用底层 embedding 并回写缓存。
    与 memory 的 VectorManager 共用 get_embedding_cache() 单例，RAG 与长期记忆共享同一缓存。
    """

    def __init__(self, base_embedding: Embeddings):
        self.base_embedding = base_embedding
        try:
            from memory.embedding_cache import get_embedding_cache

            self._cache = get_embedding_cache()
        except Exception:
            self._cache = None
        if not self._cache:
            logger.debug("Embedding 缓存未启用，CacheAwareEmbedding 仅透传")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._cache:
            return self.base_embedding.embed_documents(texts)
        results: List[Optional[List[float]]] = [None] * len(texts)
        missed_indices: List[int] = []
        missed_texts: List[str] = []
        for i, text in enumerate(texts):
            cached = self._cache.get_sync(text)
            if cached is not None:
                results[i] = cached
            else:
                missed_indices.append(i)
                missed_texts.append(text)
        if not missed_texts:
            logger.debug("Embedding 全部命中缓存: %d 条", len(texts))
            return results  # type: ignore[return-value]
        embeddings = self.base_embedding.embed_documents(missed_texts)
        for idx, emb, text in zip(missed_indices, embeddings, missed_texts):
            results[idx] = emb
            self._cache.put_sync(text, emb)
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        if self._cache:
            cached = self._cache.get_sync(text)
            if cached is not None:
                return cached
        emb = self.base_embedding.embed_query(text)
        if self._cache:
            self._cache.put_sync(text, emb)
        return emb

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._cache:
            if hasattr(self.base_embedding, "aembed_documents"):
                return await self.base_embedding.aembed_documents(texts)
            return self.base_embedding.embed_documents(texts)
        results: List[Optional[List[float]]] = [None] * len(texts)
        missed_indices: List[int] = []
        missed_texts: List[str] = []
        for i, text in enumerate(texts):
            cached = await self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                missed_indices.append(i)
                missed_texts.append(text)
        if not missed_texts:
            return results  # type: ignore[return-value]
        if hasattr(self.base_embedding, "aembed_documents"):
            embeddings = await self.base_embedding.aembed_documents(missed_texts)
        else:
            embeddings = self.base_embedding.embed_documents(missed_texts)
        for idx, emb, text in zip(missed_indices, embeddings, missed_texts):
            results[idx] = emb
            await self._cache.put(text, emb)
        return results  # type: ignore[return-value]

    async def aembed_query(self, text: str) -> List[float]:
        if self._cache:
            cached = await self._cache.get(text)
            if cached is not None:
                return cached
        if hasattr(self.base_embedding, "aembed_query"):
            emb = await self.base_embedding.aembed_query(text)
        else:
            emb = self.base_embedding.embed_query(text)
        if self._cache:
            await self._cache.put(text, emb)
        return emb


# 保守的默认 Token 限制（适用于大多数 8k 上下文的模型，如 nomic-embed-text）
DEFAULT_MAX_BATCH_TOKENS = 6000


class TokenAwareEmbedding(Embeddings):
    """
    LangChain Embeddings 模型的 Wrapper。

    自动根据 Token 计数而非仅仅是 Item 数量来对输入进行分批。
    """

    def __init__(
        self,
        base_embedding: Embeddings,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        model_name: str = "cl100k_base",
    ):
        self.base_embedding = base_embedding
        self.max_batch_tokens = max_batch_tokens
        try:
            self.encoding = tiktoken.get_encoding(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 3

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = self._estimate_tokens(text)
            if text_tokens > self.max_batch_tokens:
                logger.warning(
                    "单条文档 Token 数超过限制 (%d > %d)。它可能会被 Embedding 模型截断。",
                    text_tokens,
                    self.max_batch_tokens,
                )

            if current_tokens + text_tokens > self.max_batch_tokens and current_batch:
                logger.debug(
                    "处理 Embedding 批次: %d 文档, ~%d tokens",
                    len(current_batch),
                    current_tokens,
                )
                batch_embeddings = self.base_embedding.embed_documents(current_batch)
                all_embeddings.extend(batch_embeddings)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += text_tokens

        if current_batch:
            logger.debug(
                "处理最终 Embedding 批次: %d 文档, ~%d tokens",
                len(current_batch),
                current_tokens,
            )
            batch_embeddings = self.base_embedding.embed_documents(current_batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.base_embedding.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.base_embedding, "aembed_documents"):
            all_embeddings: List[List[float]] = []
            current_batch: List[str] = []
            current_tokens = 0

            for text in texts:
                text_tokens = self._estimate_tokens(text)

                if current_tokens + text_tokens > self.max_batch_tokens and current_batch:
                    batch_embeddings = await self.base_embedding.aembed_documents(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    current_batch = []
                    current_tokens = 0

                current_batch.append(text)
                current_tokens += text_tokens

            if current_batch:
                batch_embeddings = await self.base_embedding.aembed_documents(current_batch)
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        if hasattr(self.base_embedding, "aembed_query"):
            return await self.base_embedding.aembed_query(text)
        return self.embed_query(text)

