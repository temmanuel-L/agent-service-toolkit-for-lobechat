"""
Embedding cache for vector operations.

Caches embeddings by content hash to avoid redundant API calls,
significantly speeding up re-indexing and deduplication operations.
Memory + 磁盘 LRU；支持 async get/put 与 sync get_sync/put_sync，供 RAG / memory 共用。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from utils.log_utils import get_logger

logger = get_logger(__name__)

# 进程内单例，供 VectorManager 与 RAG 共用同一缓存
_embedding_cache_instance: Optional["EmbeddingCache"] = None
_lock = threading.Lock()


class EmbeddingCache:
    """
    内存 + 磁盘缓存embedding，避免重复调用embedding API。
    
    设计：
    - 内存缓存：快速访问，存储最近使用的embedding
    - 磁盘缓存：持久化存储，跨会话复用
    - LRU淘汰：内存缓存达到上限时淘汰最旧的条目
    """
    
    def __init__(self, cache_file: str | None = None, max_memory_items: int = 1000):
        """
        Args:
            cache_file: 磁盘缓存文件路径（JSON格式）
            max_memory_items: 内存缓存最大条目数
        """
        self.cache_file = Path(cache_file) if cache_file else None
        self.max_memory_items = max_memory_items
        
        # 内存缓存: {content_hash: {"embedding": [...], "timestamp": ...}}
        self._memory_cache: dict[str, dict] = {}
        
        # 加载磁盘缓存到内存
        if self.cache_file and self.cache_file.exists():
            self._load_from_disk()
    
    def _compute_hash(self, content: str) -> str:
        """计算内容的哈希值。"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _load_from_disk(self) -> None:
        """从磁盘加载缓存。"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                disk_cache = json.load(f)
            
            # 按时间戳排序，只加载最新的 max_memory_items 条
            items = sorted(
                disk_cache.items(),
                key=lambda x: x[1].get('timestamp', 0),
                reverse=True
            )
            self._memory_cache = dict(items[:self.max_memory_items])
            
            logger.info(f"从磁盘加载embedding缓存: {len(self._memory_cache)} 条")
        except Exception as e:
            logger.warning(f"加载embedding缓存失败: {e}")
            self._memory_cache = {}
    
    def _save_to_disk(self) -> None:
        """保存缓存到磁盘。"""
        if not self.cache_file:
            return
        
        try:
            # 确保目录存在
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._memory_cache, f, ensure_ascii=False)
            
            logger.debug(f"保存embedding缓存到磁盘: {len(self._memory_cache)} 条")
        except Exception as e:
            logger.warning(f"保存embedding缓存失败: {e}")
    
    def get_sync(self, content: str) -> Optional[list[float]]:
        """
        同步从内存缓存获取 embedding，供 RAG 等同步调用路径使用。
        仅读内存，不触发磁盘 I/O。
        """
        content_hash = self._compute_hash(content)
        entry = self._memory_cache.get(content_hash)
        if entry:
            entry["timestamp"] = time.time()
            return entry["embedding"]
        return None

    def put_sync(self, content: str, embedding: list[float]) -> None:
        """
        同步写入内存缓存；达到 LRU 上限时淘汰；不阻塞写磁盘（由后台或定期保存）。
        """
        content_hash = self._compute_hash(content)
        self._memory_cache[content_hash] = {
            "embedding": embedding,
            "timestamp": time.time(),
        }
        if len(self._memory_cache) > self.max_memory_items:
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True,
            )
            self._memory_cache = dict(sorted_items[: self.max_memory_items])
        if len(self._memory_cache) % 100 == 0:
            self._save_to_disk()

    async def get(self, content: str) -> Optional[list[float]]:
        """
        从缓存中获取embedding。
        
        Returns:
            embedding向量，或None（未缓存）
        """
        content_hash = self._compute_hash(content)
        entry = self._memory_cache.get(content_hash)
        
        if entry:
            # 更新访问时间（LRU）
            entry['timestamp'] = time.time()
            return entry['embedding']
        
        return None
    
    async def put(self, content: str, embedding: list[float]) -> None:
        """
        将embedding写入缓存。
        
        Args:
            content: 文本内容
            embedding: embedding向量
        """
        content_hash = self._compute_hash(content)
        
        self._memory_cache[content_hash] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
        
        # 内存缓存达到上限，淘汰最旧的条目
        if len(self._memory_cache) > self.max_memory_items:
            # 按时间戳排序，保留最新的
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )
            self._memory_cache = dict(sorted_items[:self.max_memory_items])
            logger.debug(f"LRU淘汰: 缓存大小 {len(sorted_items)} -> {len(self._memory_cache)}")
        
        # 定期保存到磁盘（每100次写入）
        if len(self._memory_cache) % 100 == 0:
            self._save_to_disk()
    
    def flush(self) -> None:
        """立即保存缓存到磁盘。"""
        self._save_to_disk()
    
    def clear(self) -> None:
        """清空缓存。"""
        self._memory_cache = {}
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding缓存已清空")


class CacheAwareEmbeddingWrapper(Embeddings):
    """
    仅做缓存的 Embeddings 包装器，供 memory / vector store 使用，与 RAG 共用同一 EmbeddingCache。
    不包含 token 批处理，仅先查缓存、未命中再调底层并回写。
    """

    def __init__(self, base: Embeddings):
        self._base = base
        self._cache = get_embedding_cache()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._cache:
            return self._base.embed_documents(texts)
        out: List[Optional[List[float]]] = [None] * len(texts)
        miss_idx, miss_txt = [], []
        for i, t in enumerate(texts):
            hit = self._cache.get_sync(t)
            if hit is not None:
                out[i] = hit
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if not miss_txt:
            return out  # type: ignore[return-value]
        embs = self._base.embed_documents(miss_txt)
        for i, e, t in zip(miss_idx, embs, miss_txt):
            out[i] = e
            self._cache.put_sync(t, e)
        return out  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        if self._cache:
            hit = self._cache.get_sync(text)
            if hit is not None:
                return hit
        e = self._base.embed_query(text)
        if self._cache:
            self._cache.put_sync(text, e)
        return e

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._cache:
            if hasattr(self._base, "aembed_documents"):
                return await self._base.aembed_documents(texts)
            return self._base.embed_documents(texts)
        out: List[Optional[List[float]]] = [None] * len(texts)
        miss_idx, miss_txt = [], []
        for i, t in enumerate(texts):
            hit = await self._cache.get(t)
            if hit is not None:
                out[i] = hit
            else:
                miss_idx.append(i)
                miss_txt.append(t)
        if not miss_txt:
            return out  # type: ignore[return-value]
        if hasattr(self._base, "aembed_documents"):
            embs = await self._base.aembed_documents(miss_txt)
        else:
            embs = self._base.embed_documents(miss_txt)
        for i, e, t in zip(miss_idx, embs, miss_txt):
            out[i] = e
            await self._cache.put(t, e)
        return out  # type: ignore[return-value]

    async def aembed_query(self, text: str) -> List[float]:
        if self._cache:
            hit = await self._cache.get(text)
            if hit is not None:
                return hit
        if hasattr(self._base, "aembed_query"):
            e = await self._base.aembed_query(text)
        else:
            e = self._base.embed_query(text)
        if self._cache:
            await self._cache.put(text, e)
        return e


def get_embedding_cache() -> Optional[EmbeddingCache]:
    """
    返回进程内单例 EmbeddingCache，与 VectorManager 使用相同配置，供 RAG 与 memory 共用。
    若 EMBEDDING_CACHE_ENABLED 为 False 则返回 None。
    """
    global _embedding_cache_instance
    if _embedding_cache_instance is not None:
        return _embedding_cache_instance
    with _lock:
        if _embedding_cache_instance is not None:
            return _embedding_cache_instance
        from core import settings
        if not getattr(settings, "EMBEDDING_CACHE_ENABLED", True):
            return None
        path = getattr(settings, "EMBEDDING_CACHE_PATH", "./data/embedding_cache.json")
        _embedding_cache_instance = EmbeddingCache(cache_file=path)
        return _embedding_cache_instance


def get_cache_aware_embedding() -> Embeddings:
    """
    返回带缓存的 embedding 模型，供长期记忆检索/去重与 VectorManager 的 Qdrant 检索使用，
    与 RAG 共用同一 EmbeddingCache。若未启用缓存则返回裸 get_embedding_model()。
    """
    from core.llm import get_embedding_model
    base = get_embedding_model()
    cache = get_embedding_cache()
    return CacheAwareEmbeddingWrapper(base) if cache else base
