"""
Embedding 包装器：Token 批处理 + 缓存

- TokenAwareEmbedding: 自适应 Token 批处理，避免上下文溢出
- CacheAwareEmbedding: 复用 memory 的 EmbeddingCache，避免重复调用 API（与 VectorManager 共用同一缓存）
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
            logger.debug(f"Embedding 全部命中缓存: {len(texts)} 条")
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
        model_name: str = "cl100k_base" # 默认使用 OpenAI encoding
    ):
        self.base_embedding = base_embedding
        self.max_batch_tokens = max_batch_tokens
        try:
            self.encoding = tiktoken.get_encoding(model_name)
        except Exception:
            # 如果找不到指定模型的 encoding，回退到 cl100k_base
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 tokens 数量"""
        # 优先使用 tiktoken 进行准确估算
        # 如果追求极致性能，可以用 len(text) // 3 作为启发式估算（不推荐用于生产）
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text) // 3

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行 embedding，包含自适应批处理逻辑。
        """
        all_embeddings = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = self._estimate_tokens(text)
            
            # 边界情况：如果单条文本本身就超过限制，记录警告但仍尝试处理
            # （可能会被模型截断）
            if text_tokens > self.max_batch_tokens:
                logger.warning(
                    f"单条文档 Token 数超过限制 ({text_tokens} > {self.max_batch_tokens})。 "
                    "它可能会被 Embedding 模型截断。"
                )
            
            # 检查加入当前文本是否会超出 Token 限制
            if current_tokens + text_tokens > self.max_batch_tokens and current_batch:
                # 处理当前批次
                logger.debug(f"处理 Embedding 批次: {len(current_batch)} 文档, ~{current_tokens} tokens")
                batch_embeddings = self.base_embedding.embed_documents(current_batch)
                all_embeddings.extend(batch_embeddings)
                
                # 重置批次
                current_batch = []
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
            
        # 处理剩余的批次
        if current_batch:
            logger.debug(f"处理最终 Embedding 批次: {len(current_batch)} 文档, ~{current_tokens} tokens")
            batch_embeddings = self.base_embedding.embed_documents(current_batch)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """对单个查询进行 embedding"""
        return self.base_embedding.embed_query(text)
    
    # 异步方法（如果基础模型支持，否则回退到同步）
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步：对文档列表进行 embedding，包含自适应批处理逻辑。"""
        if hasattr(self.base_embedding, "aembed_documents"):
            all_embeddings = []
            current_batch = []
            current_tokens = 0
            
            for text in texts:
                text_tokens = self._estimate_tokens(text)
                
                if current_tokens + text_tokens > self.max_batch_tokens and current_batch:
                    #Process current batch
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
        else:
            return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """异步：对单个查询进行 embedding"""
        if hasattr(self.base_embedding, "aembed_query"):
            return await self.base_embedding.aembed_query(text)
        else:
            return self.embed_query(text)
