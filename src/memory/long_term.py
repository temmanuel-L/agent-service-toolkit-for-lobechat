"""
长期记忆管理模块。

架构：
                    ┌──────────────────────┐
                    │    MemoryManager      │
                    │  （长期记忆编排器）     │
                    └────┬────────────┬─────┘
                         │            │
              ┌──────────▼──┐   ┌─────▼──────────┐
              │ Postgres    │   │ Qdrant          │
              │ store       │   │ VectorManager   │
              │ (摘要读写)   │   │ (片段向量检索)  │
              └─────────────┘   └────────────────┘

关键设计：
- MemoryManager 持有自己的 store 引用（由 lifespan 注入共享 store）
- VectorManager 同样由 lifespan 注入（与 vector_search_tool 共用同一实例）
- 所有公共方法只需 user_id / query，外部无需关心底层存储
- 读写均有超时保护 + 结构化日志，确保不阻塞主请求路径
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import time
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from core import get_model, settings
from utils.log_utils import get_logger

# For deduplication and relevance scoring
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memory.utils import is_low_quality_text

logger = get_logger(__name__)

# Postgres store 中的 namespace / key 约定
SUMMARY_NAMESPACE_SUFFIX = "long_term_summary"
SUMMARY_KEY = "summary"
PENDING_NAMESPACE_SUFFIX = "long_term_pending"
PENDING_KEY = "inputs"


# ===== 工具函数 ============================================================

def _utc_now_iso() -> str:
    """UTC 时间 ISO 字符串，用于时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    """安全截断文本。"""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _normalize_store_item(item: Any) -> Optional[Any]:
    """兼容 store.aget 返回 Item 或 list[Item] 两种格式。"""
    if item is None:
        return None
    if isinstance(item, list):
        return item[0] if item else None
    return item


def _backend_uses_vector() -> bool:
    return settings.LONG_TERM_MEMORY_BACKEND in ("pg_plus_qdrant", "qdrant_only")


def _backend_uses_summary() -> bool:
    return settings.LONG_TERM_MEMORY_BACKEND in ("pg_plus_qdrant", "postgres_only")


def _hash_text(text: str) -> str:
    """计算文本的哈希值（用于去重）。"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _estimate_tokens(text: str) -> int:
    """
    估算文本的token数量。
    
    使用简单启发式：中文字符 ~1.5 tokens，英文单词 ~1 token。
    这是保守估计，避免超出限制。
    """
    if not text:
        return 0
    # 统计中文字符（CJK）
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 统计英文单词（空格分隔）
    english_words = len([w for w in text.split() if w and any(c.isalpha() for c in w)])
    return int(chinese_chars * 1.5 + english_words)


def _validate_content(text: str) -> bool:
    """
    验证记忆内容是否有效。
    
    检查项：
    1. 非空且去空白后有一定长度
    2. 数据质量检查：调用 `is_low_quality_text` 识别无意义重复或死循环
    3. 长度安全上限
    """
    if not text or not text.strip():
        return False
    
    # 质量检查：过滤掉 LLM 幻觉产生的循环文本（如 "A A A..."）或乱码
    # 这是防止脏数据污染上下文的关键防线
    if is_low_quality_text(text):
        return False
    
    # 长度硬限制：防止极长文本导致的处理超时
    # 虽然 TokenAwareEmbedding 会处理批次，但作为单条记忆，过长通常意味着切割失败或异常
    if len(text) > 10000:
        logger.warning(f"内容过长({len(text)}字符)，可能是损坏数据，已丢弃")
        return False
    
    return True


# ===== 数据结构 ============================================================

@dataclass
class MemoryContext:
    """长期记忆上下文：摘要 + 向量检索片段。"""

    summary: str | None
    snippets: list[str]

    def is_empty(self) -> bool:
        return not self.summary and not self.snippets

    def format_for_prompt(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append("长期记忆摘要：\n" + self.summary.strip())
        if self.snippets:
            items = "\n".join(f"- {s}" for s in self.snippets if s)
            if items:
                parts.append("相关记忆片段：\n" + items)
        return "\n\n".join(p for p in parts if p).strip()


# ===== 核心类 ==============================================================

class MemoryManager:
    """
    长期记忆管理器。

    生命周期：
    1. 模块级创建空实例 `memory_manager = MemoryManager()`
    2. lifespan 注入依赖：
       - memory_manager.vector_manager = <全局 VectorManager>
       - memory_manager.set_store(<共享 store>)
    3. handlers.py 在每次请求时调用 abuild_system_message / arecord_turn

    所有公共方法仅接受业务参数（user_id / query），底层资源完全自治。
    """

    def __init__(self) -> None:
        self._vector_manager = None          # 由 lifespan 注入
        self._store: BaseStore | None = None  # 由 lifespan 注入
        self._vm_initialized = False          # VectorManager 是否已 ainitialize

    # ---- 依赖注入 ----

    @property
    def vector_manager(self):
        """获取 VectorManager（未注入时返回 None）。"""
        return self._vector_manager

    @vector_manager.setter
    def vector_manager(self, vm) -> None:
        self._vector_manager = vm
        self._vm_initialized = False  # 新实例需要重新 ainitialize

    def set_store(self, store: BaseStore) -> None:
        """由 lifespan 注入共享 store。"""
        self._store = store
        logger.info("MemoryManager: store 已注入")

    @property
    def store(self) -> BaseStore | None:
        return self._store

    async def _ensure_vm_ready(self) -> bool:
        """确保 VectorManager 已初始化，返回是否可用。"""
        if self._vector_manager is None:
            return False
        if not self._vm_initialized:
            await self._vector_manager.ainitialize()
            self._vm_initialized = True
        return True

    # ---- Postgres store 操作（摘要读写）----

    async def _aget_summary(self, user_id: str) -> str | None:
        """
        读取用户长期记忆摘要。

        正常耗时 5-50ms；超时阈值由 LONG_TERM_MEMORY_STORE_TIMEOUT_MS 控制。
        """
        store = self._store
        if not store or not user_id or not _backend_uses_summary():
            return None

        start = time.perf_counter()
        timeout_s = settings.LONG_TERM_MEMORY_STORE_TIMEOUT_MS / 1000
        namespace = (user_id, SUMMARY_NAMESPACE_SUFFIX)

        try:
            item = await asyncio.wait_for(
                store.aget(namespace, key=SUMMARY_KEY), timeout=timeout_s,
            )
            elapsed = (time.perf_counter() - start) * 1000
            item = _normalize_store_item(item)
            if not item:
                logger.info("摘要读取: user=%s found=False elapsed=%.1fms", user_id, elapsed)
                return None
            value = getattr(item, "value", None) or {}
            summary = value.get("summary")
            summary = summary.strip() if isinstance(summary, str) and summary.strip() else None
            
            # 增加质量校验：防止被污染的摘要再次注入上下文
            if summary and is_low_quality_text(summary):
                logger.warning(f"检测到低质量摘要 (已丢弃): {summary[:50]}...")
                return None
                
            logger.info("摘要读取: user=%s found=%s elapsed=%.1fms", user_id, bool(summary), elapsed)
            return summary

        except asyncio.TimeoutError:
            logger.warning(
                "摘要读取超时: user=%s elapsed=%.1fms limit=%dms",
                user_id, (time.perf_counter() - start) * 1000, settings.LONG_TERM_MEMORY_STORE_TIMEOUT_MS,
            )
            return None
        except Exception as exc:
            logger.warning(
                "摘要读取失败: user=%s elapsed=%.1fms err=%s",
                user_id, (time.perf_counter() - start) * 1000, exc,
            )
            return None

    async def _aput_summary(self, user_id: str, summary: str) -> None:
        """写入用户长期记忆摘要。"""
        store = self._store
        if not store or not user_id or not summary or not _backend_uses_summary():
            return

        start = time.perf_counter()
        namespace = (user_id, SUMMARY_NAMESPACE_SUFFIX)
        try:
            await asyncio.wait_for(
                store.aput(namespace, SUMMARY_KEY, {"summary": summary, "updated_at": _utc_now_iso()}),
                timeout=settings.LONG_TERM_MEMORY_STORE_TIMEOUT_MS / 1000,
            )
            logger.info(
                "摘要写入成功: user=%s len=%d elapsed=%.1fms",
                user_id, len(summary), (time.perf_counter() - start) * 1000,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "摘要写入超时: user=%s elapsed=%.1fms limit=%dms",
                user_id, (time.perf_counter() - start) * 1000, settings.LONG_TERM_MEMORY_STORE_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.warning(
                "摘要写入失败: user=%s elapsed=%.1fms err=%s",
                user_id, (time.perf_counter() - start) * 1000, exc,
            )

    async def _aget_pending(self, user_id: str) -> list[str]:
        """读取待压缩的摘要输入缓冲（用于按间隔压缩）。"""
        store = self._store
        if not store or not user_id or not _backend_uses_summary():
            return []
        try:
            item = await asyncio.wait_for(
                store.aget((user_id, PENDING_NAMESPACE_SUFFIX), key=PENDING_KEY),
                timeout=settings.LONG_TERM_MEMORY_STORE_TIMEOUT_MS / 1000,
            )
            item = _normalize_store_item(item)
            if not item:
                return []
            value = getattr(item, "value", None) or {}
            messages = value.get("messages")
            return list(messages) if isinstance(messages, list) else []
        except Exception:
            return []

    async def _aput_pending(self, user_id: str, messages: list[str]) -> None:
        """写入待压缩的摘要输入缓冲。"""
        store = self._store
        if not store or not user_id or not _backend_uses_summary():
            return
        try:
            await asyncio.wait_for(
                store.aput((user_id, PENDING_NAMESPACE_SUFFIX), PENDING_KEY, {"messages": messages}),
                timeout=settings.LONG_TERM_MEMORY_STORE_TIMEOUT_MS / 1000,
            )
        except Exception as exc:
            logger.warning("待压缩缓冲写入失败: user=%s err=%s", user_id, exc)

    async def _aupdate_summary(
        self, user_id: str, messages: Iterable[str], config: RunnableConfig | None = None,
    ) -> str | None:
        """
        使用 LLM 将本轮对话压缩进长期摘要（增量更新）。
        
        改进：
        - 更严格的压缩指令，避免摘要无限膨胀
        - 明确要求去重和合并相似信息
        - 优先保留最新的、可操作的信息
        """
        if not self._store or not user_id or not _backend_uses_summary():
            return None

        merged = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
        if not merged:
            return await self._aget_summary(user_id)

        # 限制单条长度，缩短摘要 LLM 输入以加快推理
        max_chars = getattr(settings, "LONG_TERM_MEMORY_MAX_ITEM_CHARS", 400)
        merged = [_truncate(m, max_chars) for m in merged]
        previous = await self._aget_summary(user_id) or "无"
        input_text = "\n".join(f"- {m}" for m in merged)

        system_prompt = (
            "你是长期记忆整理助手。你的任务是将新对话内容压缩合并到现有摘要中。\n\n"
            "压缩规则：\n"
            "1. 去重：如果新信息与已有摘要重复，则直接跳过\n"
            "2. 修正：如果新信息与已有记忆冲突（如名字、地点、偏好改变），必须**删除旧信息，只保留新信息**\n"
            "3. 合并：如果新信息是已有信息的补充，合并到已有条目\n"
            "4. 提炼：只保留关键事实、偏好、目标，删除闲聊和冗余细节\n"
            "5. 格式：最终摘要不超过800字，使用条目式列表，严禁通过'用户说'等前缀复述对话\n\n"
            "请用中文输出。"
        )
        user_prompt = (
            f"【当前摘要】\n{previous}\n\n"
            f"【新增对话】\n{input_text}\n\n"
            f"请输出压缩后的摘要（条目式，不超过800字）："
        )

        model_name = settings.LONG_TERM_MEMORY_MODEL or settings.DEFAULT_MODEL
        model = get_model(model_name)
        response = await model.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            config=config,
        )
        summary = response.content if hasattr(response, "content") else str(response)
        summary = _truncate(summary, settings.LONG_TERM_MEMORY_SUMMARY_MAX_CHARS)
        
        # 写入前二次校验：防止生成的摘要本身包含死循环
        if summary and not is_low_quality_text(summary):
            await self._aput_summary(user_id, summary)
        elif summary:
            logger.warning(f"生成了低质量摘要 (已丢弃): {summary[:50]}...")
            
        return summary or None

    # ---- Qdrant 操作（向量片段检索）----

    async def _deduplicate_snippets(self, snippets: list[str]) -> list[str]:
        """
        对记忆片段去重。
        
        策略：
        1. 基于哈希值去除完全相同的片段
        2. 基于语义相似度去除高度相似的片段（cosine > threshold）
        """
        if not snippets:
            return []
        
        # Step 1: Hash-based deduplication
        seen_hashes = set()
        unique_snippets = []
        for s in snippets:
            h = _hash_text(s)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_snippets.append(s)
        
        if len(unique_snippets) < len(snippets):
            logger.info(f"哈希去重: {len(snippets)} -> {len(unique_snippets)}")
        
        # Step 2: Semantic deduplication (only if we have embedding model)
        if len(unique_snippets) <= 1:
            return unique_snippets
        
        try:
            from memory.embedding_cache import get_cache_aware_embedding
            embed_model = get_cache_aware_embedding()
            embeddings = await embed_model.aembed_documents(unique_snippets)
            
            # Compute pairwise cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Keep track of which snippets to keep
            to_keep = [True] * len(unique_snippets)
            threshold = settings.LONG_TERM_MEMORY_DEDUP_THRESHOLD
            
            for i in range(len(unique_snippets)):
                if not to_keep[i]:
                    continue
                for j in range(i + 1, len(unique_snippets)):
                    if not to_keep[j]:
                        continue
                    if similarity_matrix[i][j] > threshold:
                        # j is very similar to i, remove j (keep the first occurrence)
                        to_keep[j] = False
                        logger.debug(f"语义去重: snippet {j} 与 {i} 相似度 {similarity_matrix[i][j]:.3f}")
            
            deduplicated = [s for i, s in enumerate(unique_snippets) if to_keep[i]]
            
            if len(deduplicated) < len(unique_snippets):
                logger.info(f"语义去重: {len(unique_snippets)} -> {len(deduplicated)}")
            
            return deduplicated
        
        except Exception as exc:
            logger.warning(f"语义去重失败: {exc}，使用哈希去重结果")
            return unique_snippets
    
    async def _filter_by_relevance(self, snippets: list[str], query: str) -> list[str]:
        """
        过滤低相关性的记忆片段。批量 embedding（1 次 query + 1 次 documents），避免 N×2 次 API 调用。
        """
        if not snippets:
            return []
        threshold = settings.LONG_TERM_MEMORY_MIN_RELEVANCE_SCORE
        try:
            from memory.embedding_cache import get_cache_aware_embedding
            embed_model = get_cache_aware_embedding()
            query_emb, snippet_embs = await asyncio.gather(
                embed_model.aembed_query(query),
                embed_model.aembed_documents(snippets),
            )
            q = np.array(query_emb).reshape(1, -1)
            S = np.array(snippet_embs)
            scores = cosine_similarity(S, q).ravel()
            scored_snippets = [(s, float(max(0.0, min(1.0, sc)))) for s, sc in zip(snippets, scores) if sc >= threshold]
            scored_snippets.sort(key=lambda x: x[1], reverse=True)
            filtered = [s for s, _ in scored_snippets]
            if len(filtered) < len(snippets):
                logger.info(
                    "相关性过滤: %d -> %d (threshold=%.2f, scores=%s)",
                    len(snippets), len(filtered), threshold,
                    [f"{sc:.2f}" for _, sc in scored_snippets[:3]],
                )
            return filtered
        except Exception as exc:
            logger.warning("相关性过滤失败: %s，保留全部片段", exc)
            return snippets
    
    async def _aretrieve_snippets(self, query: str, user_id: str, top_k: int | None = None) -> list[str]:
        """
        从 Qdrant 检索与当前问题相关的对话记忆片段。
        
        改进：
        - 内容验证：过滤损坏数据
        - 去重：移除重复和高度相似的片段
        - 相关性过滤：只保留与当前查询相关的片段
        """
        if not user_id or not query or not _backend_uses_vector():
            return []
        if not await self._ensure_vm_ready():
            return []

        k = top_k or settings.LONG_TERM_MEMORY_TOP_K
        retrieve_k = min(k * 2, 20)  # 记忆不多时减少检索量，目标构建 <1s

        from memory.embedding_cache import get_cache_aware_embedding
        embed_model = get_cache_aware_embedding()

        # 先算 query embedding 一次，再按向量检索，避免 asimilarity_search 内部再算一次（省 1 次 API）
        t0 = time.perf_counter()
        try:
            query_emb = await embed_model.aembed_query(query)
        except Exception as exc:
            logger.warning("长期记忆 query embedding 失败: %s", exc)
            return []
        t_query_embed = (time.perf_counter() - t0) * 1000

        try:
            docs = await self._vector_manager.asimilarity_search_by_vector(
                query_vector=query_emb,
                user_id=user_id,
                k=retrieve_k,
            )
        except Exception as exc:
            logger.warning("向量检索失败: user=%s elapsed=%.1fms err=%s", user_id, (time.perf_counter() - t0) * 1000, exc)
            return []
        t_qdrant = (time.perf_counter() - t0) * 1000 - t_query_embed

        snippets = []
        for doc in docs:
            content = getattr(doc, "page_content", "") or ""
            if _validate_content(content):
                snippets.append(_truncate(content, settings.LONG_TERM_MEMORY_MAX_ITEM_CHARS))

        if not snippets:
            logger.info(
                "长期记忆耗时: query_embed=%.0fms qdrant=%.0fms snippets=0 total=%.0fms",
                t_query_embed, t_qdrant, (time.perf_counter() - t0) * 1000,
            )
            return []

        # 哈希去重（无 API）
        seen_hashes = set()
        unique_snippets = []
        for s in snippets:
            h = _hash_text(s)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_snippets.append(s)
        if len(unique_snippets) < len(snippets):
            logger.info("哈希去重: %d -> %d", len(snippets), len(unique_snippets))
        snippets = unique_snippets

        # 只对 snippets 做一次 embedding，query_emb 已在上方算过并用于检索
        t_embed_start = time.perf_counter()
        try:
            snippet_embs = await embed_model.aembed_documents(snippets)
        except Exception as exc:
            logger.warning("长期记忆 snippet embedding 失败: %s，返回哈希去重结果", exc)
            logger.info(
                "长期记忆耗时: query_embed=%.0fms qdrant=%.0fms snippet_embed_fail total=%.0fms",
                t_query_embed, t_qdrant, (time.perf_counter() - t0) * 1000,
            )
            return snippets[:k]
        t_snippet_embed = (time.perf_counter() - t_embed_start) * 1000

        S = np.array(snippet_embs)
        # 语义去重：相似度矩阵，保留首个
        if len(snippets) > 1:
            sim = cosine_similarity(S)
            to_keep = [True] * len(snippets)
            dedup_th = settings.LONG_TERM_MEMORY_DEDUP_THRESHOLD
            for i in range(len(snippets)):
                if not to_keep[i]:
                    continue
                for j in range(i + 1, len(snippets)):
                    if to_keep[j] and sim[i, j] > dedup_th:
                        to_keep[j] = False
            snippets = [s for i, s in enumerate(snippets) if to_keep[i]]
            S = S[to_keep]

        # 相关性过滤并排序
        q = np.array(query_emb).reshape(1, -1)
        scores = cosine_similarity(S, q).ravel()
        rel_th = settings.LONG_TERM_MEMORY_MIN_RELEVANCE_SCORE
        scored = [(s, float(max(0.0, min(1.0, sc)))) for s, sc in zip(snippets, scores) if sc >= rel_th]
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [s for s, _ in scored][:k]
        t_dedup_filter = (time.perf_counter() - t_embed_start) * 1000 - t_snippet_embed
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "长期记忆耗时: query_embed=%.0fms qdrant=%.0fms snippet_embed=%.0fms dedup_filter=%.0fms total=%.0fms raw=%d -> %d",
            t_query_embed, t_qdrant, t_snippet_embed, max(0, t_dedup_filter), total_ms, len(docs), len(result),
        )
        return result

    # ---- 公共 API ----

    async def abuild_context(self, user_id: str, query: str) -> MemoryContext:
        """
        组合摘要 + 向量片段，形成可注入的上下文。
        
        改进：
        - 强制token限制：确保总上下文不超过 MAX_CONTEXT_TOKENS
        - 动态裁剪：优先保留摘要，然后按相关性裁剪snippets
        """
        start = time.perf_counter()
        max_tokens = settings.LONG_TERM_MEMORY_MAX_CONTEXT_TOKENS

        use_summary = _backend_uses_summary()
        use_vector = _backend_uses_vector()
        max_wait_ms = getattr(settings, "LONG_TERM_MEMORY_MAX_WAIT_MS", 0)

        if use_summary and use_vector:
            summary = await self._aget_summary(user_id)
            if max_wait_ms > 0:
                try:
                    snippets = await asyncio.wait_for(
                        self._aretrieve_snippets(query, user_id),
                        timeout=max_wait_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "长期记忆片段检索超时(limit=%dms)，仅使用摘要以保证响应速度",
                        max_wait_ms,
                    )
                    snippets = []
            else:
                snippets = await self._aretrieve_snippets(query, user_id)
        elif use_summary:
            summary = await self._aget_summary(user_id)
            snippets = []
        elif use_vector:
            summary = None
            if max_wait_ms > 0:
                try:
                    snippets = await asyncio.wait_for(
                        self._aretrieve_snippets(query, user_id),
                        timeout=max_wait_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("长期记忆片段检索超时(limit=%dms)，返回空片段", max_wait_ms)
                    snippets = []
            else:
                snippets = await self._aretrieve_snippets(query, user_id)
        else:
            summary = None
            snippets = []

        # Estimate tokens and enforce limit
        summary_tokens = _estimate_tokens(summary) if summary else 0
        remaining_tokens = max_tokens - summary_tokens
        
        if remaining_tokens < 0:
            # Summary itself exceeds limit, truncate it
            logger.warning(
                f"摘要过长({summary_tokens} tokens)，裁剪到 {max_tokens} tokens"
            )
            # Truncate summary to fit
            char_limit = int(max_tokens / 1.5)  # Conservative estimate
            summary = _truncate(summary, char_limit)
            remaining_tokens = 0
            snippets = []  # No room for snippets
        else:
            # Trim snippets to fit remaining token budget
            trimmed_snippets = []
            used_tokens = 0
            for s in snippets:
                s_tokens = _estimate_tokens(s)
                if used_tokens + s_tokens <= remaining_tokens:
                    trimmed_snippets.append(s)
                    used_tokens += s_tokens
                else:
                    break
            
            if len(trimmed_snippets) < len(snippets):
                logger.info(
                    f"Token限制裁剪: snippets {len(snippets)} -> {len(trimmed_snippets)} "
                    f"(used={used_tokens + summary_tokens}/{max_tokens})"
                )
            snippets = trimmed_snippets
        
        logger.info(
            "上下文构建: user=%s summary=%s snippets=%d total_tokens~%d elapsed=%.1fms",
            user_id, bool(summary), len(snippets), 
            summary_tokens + sum(_estimate_tokens(s) for s in snippets),
            (time.perf_counter() - start) * 1000,
        )
        return MemoryContext(summary=summary, snippets=snippets)

    async def abuild_system_message(self, user_id: str, query: str) -> SystemMessage | None:
        """构建注入到 LLM 的 SystemMessage（长期记忆）。"""
        if not settings.LONG_TERM_MEMORY_ENABLED or not user_id:
            return None
        ctx = await self.abuild_context(user_id, query)
        if ctx.is_empty():
            return None
        content = ctx.format_for_prompt()
        if not content:
            return None
        guidance = (
            "以下内容由系统自动整理的长期记忆，若与当前问题相关，请优先使用；"
            "若与用户当前表述冲突，以用户最新表述为准。"
        )
        return SystemMessage(
            content=f"[长期记忆]\n{guidance}\n\n{content}",
            additional_kwargs={"source": "long_term_memory"},
        )

    async def arecord_turn(
        self,
        user_id: str,
        thread_id: str | None,
        user_message: str | None,
        assistant_message: str | None,
        config: RunnableConfig | None = None,
    ) -> None:
        """
        回写本轮对话到长期记忆。

        - Qdrant：写入对话片段向量，供后续语义检索
        - Postgres：LLM 压缩后写入滚动摘要
        """
        if not settings.LONG_TERM_MEMORY_ENABLED or not user_id:
            return

        t0 = time.perf_counter()
        messages: list[BaseMessage] = []
        summary_inputs: list[str] = []

        if user_message and user_message.strip():
            msg = HumanMessage(content=user_message.strip())
            msg.additional_kwargs["timestamp"] = _utc_now_iso()
            messages.append(msg)
            summary_inputs.append(f"用户: {user_message.strip()}")
        if assistant_message and assistant_message.strip():
            msg = AIMessage(content=assistant_message.strip())
            msg.additional_kwargs["timestamp"] = _utc_now_iso()
            messages.append(msg)
            summary_inputs.append(f"助手: {assistant_message.strip()}")

        # 写入 Qdrant
        vector_ok = False
        if messages and _backend_uses_vector() and await self._ensure_vm_ready():
            try:
                await self._vector_manager.aadd_messages(messages=messages, user_id=user_id, thread_id=thread_id)
                vector_ok = True
            except Exception as exc:
                logger.warning("向量写入失败: user=%s err=%s", user_id, exc)

        # 写入 Postgres 摘要（按间隔压缩：仅每 N 轮调用 LLM，其余轮只缓冲，缩短回写时间）
        summary_ok = False
        if self._store and summary_inputs and _backend_uses_summary():
            interval = settings.LONG_TERM_MEMORY_COMPRESSION_INTERVAL or 1
            if interval <= 1:
                logger.info(
                    "摘要压缩配置: interval=%d（每轮压缩），本轮将直接调用摘要LLM",
                    interval,
                )
                await self._aupdate_summary(user_id, summary_inputs, config=config)
                summary_ok = True
            else:
                pending = await self._aget_pending(user_id)
                pending.extend(summary_inputs)
                pending_count = len(pending)
                remaining_to_compress = max(0, interval - pending_count)
                logger.info(
                    "摘要压缩进度: user=%s interval=%d pending=%d remaining=%d",
                    user_id,
                    interval,
                    pending_count,
                    remaining_to_compress,
                )
                if len(pending) >= interval:
                    await self._aupdate_summary(user_id, pending, config=config)
                    await self._aput_pending(user_id, [])
                    logger.info(
                        "摘要压缩触发: user=%s pending=%d >= interval=%d，已调用摘要LLM并清空缓冲",
                        user_id,
                        pending_count,
                        interval,
                    )
                    summary_ok = True
                else:
                    await self._aput_pending(user_id, pending)

        logger.info(
            "记忆回写: user=%s vector=%s summary=%s elapsed=%.1fms",
            user_id, vector_ok, summary_ok, (time.perf_counter() - t0) * 1000,
        )


# 模块级单例：空壳实例，由 lifespan 注入 store + vector_manager 后方可工作
memory_manager = MemoryManager()
