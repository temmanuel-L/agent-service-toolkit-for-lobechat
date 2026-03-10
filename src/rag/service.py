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
import asyncio
import time
import json
import httpx
import tempfile
import threading
import tiktoken
import logging
from typing import Optional, List, Any
from pathlib import Path

# 屏蔽第三方库冗长的调试日志
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("bm25s").setLevel(logging.WARNING)

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle, NodeRelationship
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


def _longest_content_phrase(query: str, min_len: int = 10) -> str | None:
    """
    从查询中提取最长的「内容短语」，用于通用内容级 boost。

    规则：取 query 中最长的连续子串，满足 (a) 长度 >= min_len，(b) 由字母/数字/空格或连续 CJK 组成。
    用途：当 node.text 包含该短语时加分，作为「查询与段落内容重叠」的通用相关性信号。
    """
    if not query or len(query.strip()) < min_len:
        return None
    q = query.strip()
    pattern = rf"[a-zA-Z0-9\s]{{{min_len},}}|[\u4e00-\u9fff]{{{min_len},}}"
    matches = re.findall(pattern, q)
    if not matches:
        return None
    best = max(matches, key=len)
    return best.strip() if len(best.strip()) >= min_len else None


def _section_title_boost(section_title: str, query_lower: str) -> float:
    """
    计算 section_title 与 query 之间的关键词重叠度，返回 boost 系数。

    策略：
    1. section_title 完整出现在 query 中 → 0.6（最强匹配）
    2. query 中 ≥3 字符的片段出现在 section_title 中 → 0.5
       （覆盖 query 含"科技" 而 section_title 为"产品营销（科技）"的场景）
    3. query 中 2 字符的片段出现在 section_title 中 → 0.3
    """
    st = section_title.lower()

    if st in query_lower:
        return 0.6

    best_match_len = 0
    max_sub = min(len(query_lower), 8)
    for kw_len in range(max_sub, 1, -1):
        for i in range(len(query_lower) - kw_len + 1):
            sub = query_lower[i:i + kw_len]
            if sub in st:
                best_match_len = kw_len
                break
        if best_match_len > 0:
            break

    if best_match_len >= 3:
        return 0.5
    if best_match_len >= 2:
        return 0.3
    return 0.0


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

        # ---- 轨道 B：doc_title 缓存（用于 metadata 预过滤）----
        # key=kb_id, value=set of doc_title
        self._doc_title_cache: dict[str, set[str]] = {}

        # ---- 向量索引 / docstore 缓存（按知识库维度）----
        # 说明：
        # - simple 策略下可选地重用 VectorStoreIndex，减少每次查询的 index 构建开销；
        # - parent_child 策略下，需要依赖 docstore 中的父子节点关系，
        #   才能在检索阶段将叶子命中提升为父节点上下文。
        # key=kb_id, value=VectorStoreIndex
        self._vector_indexes: dict[str, VectorStoreIndex] = {}
        # ---- parent 文本持久化缓存（用于 parent_child 在重启后恢复父上下文）----
        self._parent_text_cache: dict[str, dict[str, str]] = {}
        self._parent_store_dir = Path("data/rag_parent_store").absolute()
        self._parent_store_dir.mkdir(parents=True, exist_ok=True)

    def _parent_store_path(self, kb_id: str) -> Path:
        safe_kb = re.sub(r"[^a-zA-Z0-9_.-]", "_", kb_id)
        return self._parent_store_dir / f"{safe_kb}.json"

    def _parent_store_key(self, kb_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", kb_id)

    def _save_parent_text_map(self, kb_id: str, parent_text_map: dict[str, str]) -> None:
        try:
            existing = self._load_parent_text_map(kb_id)
            merged = dict(existing)
            merged.update(parent_text_map)
            self._parent_text_cache[kb_id] = merged
            path = self._parent_store_path(kb_id)
            path.write_text(
                json.dumps(merged, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "parent文本映射已持久化: kb=%s, parents=%d, file=%s",
                kb_id,
                len(merged),
                str(path),
            )
        except Exception as e:
            logger.warning("parent文本映射持久化失败（不影响主流程）: kb=%s, err=%s", kb_id, e)

    def _load_parent_text_map(self, kb_id: str) -> dict[str, str]:
        cached = self._parent_text_cache.get(kb_id)
        if cached is not None:
            return cached
        path = self._parent_store_path(kb_id)
        if not path.exists():
            self._parent_text_cache[kb_id] = {}
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                parent_map = {
                    str(k): str(v)
                    for k, v in loaded.items()
                    if str(k).strip() and str(v).strip()
                }
                self._parent_text_cache[kb_id] = parent_map
                return parent_map
        except Exception as e:
            logger.warning("parent文本映射读取失败（降级为空）: kb=%s, err=%s", kb_id, e)
        self._parent_text_cache[kb_id] = {}
        return {}

    def _clear_parent_text_map(self, kb_id: str) -> None:
        self._parent_text_cache.pop(kb_id, None)
        path = self._parent_store_path(kb_id)
        try:
            if path.exists():
                path.unlink()
                logger.info("parent文本映射已删除: kb=%s, file=%s", kb_id, str(path))
        except Exception as e:
            logger.warning("删除parent文本映射失败（不影响主流程）: kb=%s, err=%s", kb_id, e)

    @staticmethod
    def _extract_metadata_from_payload(payload: dict) -> dict:
        if not payload:
            return {}
        if "metadata" in payload and isinstance(payload.get("metadata"), dict):
            return payload.get("metadata") or {}
        if "_node_content" in payload:
            try:
                nc = json.loads(payload["_node_content"])
                metadata = nc.get("metadata", {}) or {}
                if isinstance(metadata, dict):
                    return metadata
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    @staticmethod
    def _extract_text_from_payload(payload: dict) -> str:
        if not payload:
            return ""
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        if "_node_content" in payload:
            try:
                nc = json.loads(payload["_node_content"])
                text = nc.get("text", "")
                if isinstance(text, str):
                    return text.strip()
            except (json.JSONDecodeError, TypeError):
                return ""
        return ""

    @staticmethod
    def _prepend_metadata_prefix_to_nodes(nodes: list) -> None:
        """
        将 doc_title、file_name、author、section_title 等 metadata 拼入每个节点的 text，
        使 embedding 包含文档级信息，便于检索时「参考文献」等 chunk 能匹配标题类 query。

        与 BM25 侧的 meta_prefix 格式保持一致，修改节点 text 原地。
        """
        for node in nodes:
            meta = getattr(node, "metadata", {}) or {}
            text = (getattr(node, "text", "") or "").strip()
            parts: list[str] = []
            for key, label in [
                ("doc_title", "TITLE"),
                ("file_name", "FILE"),
                ("author", "AUTHOR"),
                ("section_title", "SECTION"),
            ]:
                val = str(meta.get(key, "") or "").strip()
                if val:
                    parts.append(f"[{label}] {val}")
            if parts:
                prefix = " ".join(parts)
                setattr(node, "text", f"{prefix}\n\n{text}")

    @staticmethod
    def _merge_texts_with_overlap(texts: list[str]) -> str:
        """
        将同一 parent_id 下的多个叶子文本尽量合并为更长上下文（近似父块）。
        """
        merged = ""
        for raw in texts:
            cur = (raw or "").strip()
            if not cur:
                continue
            if not merged:
                merged = cur
                continue
            if cur in merged:
                continue
            if merged in cur:
                merged = cur
                continue
            max_overlap = min(len(merged), len(cur), 120)
            overlap = 0
            for k in range(max_overlap, 20, -1):
                if merged[-k:] == cur[:k]:
                    overlap = k
                    break
            if overlap > 0:
                merged = merged + cur[overlap:]
            else:
                merged = merged + "\n" + cur
        return merged.strip()

    def _resolve_parent_segment_token_budget(self) -> int:
        """
        计算 parent_child 返回给 LLM 的「单段父上下文」token 预算。

        优先级：
        1) 显式配置 RAG_PARENT_MAX_TOKENS_PER_SEGMENT（>0 时生效）；
        2) 自动推导为 RAG_CHUNK_SIZE * RAG_FATHER_SON_RATIO。
        """
        explicit_budget = int(getattr(settings, "RAG_PARENT_MAX_TOKENS_PER_SEGMENT", 0) or 0)
        if explicit_budget > 0:
            return explicit_budget
        ratio = max(1, int(getattr(settings, "RAG_FATHER_SON_RATIO", 3) or 3))
        return max(1, int(settings.RAG_CHUNK_SIZE * ratio))

    @staticmethod
    def _clip_parent_text_for_budget(
        parent_text: str,
        leaf_text: str,
        max_tokens: int,
    ) -> tuple[str, bool]:
        """
        对父上下文做预算裁剪，围绕命中的 leaf 文本保留局部窗口。

        采用字符级近似（2.5 chars/token），避免对每个节点进行 tiktoken 全文编码，
        性能约为原 tiktoken 版本的 10 倍。精度损失在 ±20% 以内，可接受，因后续
        postprocess/truncate.py 会做精确 token 预算兜底。
        """
        text = (parent_text or "").strip()
        if not text or max_tokens <= 0:
            return text, False

        # 中英文混合保守估算：2.5 字符 ≈ 1 token
        CHARS_PER_TOKEN = 2.5
        budget_chars = int(max_tokens * CHARS_PER_TOKEN)
        if len(text) <= budget_chars:
            return text, False

        # 用 leaf 文本前 80 字符作为锚点，定位在父文本中的位置
        anchor = (leaf_text or "").strip()
        probe = anchor[:80]
        pos = text.find(probe) if probe else -1
        if pos < 0 and len(probe) > 30:
            pos = text.find(probe[:30])

        if pos >= 0:
            half = budget_chars // 2
            start = max(0, pos - half)
            end = min(len(text), start + budget_chars)
            # 若尾部已到达末端，向前补充
            if end - start < budget_chars:
                start = max(0, end - budget_chars)
            return text[start:end].strip(), True

        # 找不到锚点：优先保留 leaf，再补充父块开头
        if anchor:
            anchor_chars = len(anchor)
            if anchor_chars >= budget_chars:
                return anchor[:budget_chars].strip(), True
            remaining = budget_chars - anchor_chars - 2  # 2 for "\n\n"
            head = text[:remaining].strip() if remaining > 0 else ""
            mixed = f"{anchor}\n\n{head}".strip() if head else anchor
            return mixed[:budget_chars].strip(), True

        return text[:budget_chars].strip(), True

    def _apply_source_level_filter(
        self,
        nodes: list,
        kb_id: str = "",
    ) -> list:
        """
        来源级别相对过滤：按来源文档分组，计算每个来源的最高分，
        只保留最高分 >= 最佳来源分 * RAG_SOURCE_SCORE_RATIO 的整个来源。

        与 segment 级过滤的区别：要么保留某来源的全部段落，要么整体排除。
        对于针对特定文档的查询（如"合同中的专利条款"），该文档 BM25+向量分数
        整体高于其他文档，其他文档会被整体过滤，避免 LLM 答非所问；
        对于合理的跨文档查询，多个来源得分相近，则都保留。
        """
        ratio = getattr(settings, "RAG_SOURCE_SCORE_RATIO", 0.0) or 0.0
        if not nodes or ratio <= 0:
            return nodes

        source_max: dict[str, float] = {}
        for n in nodes:
            src = ((getattr(n.node, "metadata", None) or {}).get("file_name") or "__unknown__")
            sc = float(n.score or 0.0)
            if sc > source_max.get(src, 0.0):
                source_max[src] = sc

        if not source_max:
            return nodes

        best = max(source_max.values())
        threshold = best * ratio
        kept_sources = {src for src, sc in source_max.items() if sc >= threshold}

        before = len(nodes)
        result = [
            n for n in nodes
            if ((getattr(n.node, "metadata", None) or {}).get("file_name") or "__unknown__") in kept_sources
        ]

        if len(result) < before:
            logger.info(
                "source级别过滤: kb=%s, best_score=%.4f, threshold=%.4f(ratio=%.2f), "
                "sources_kept=%d/%d, segs=%d→%d",
                kb_id, best, threshold, ratio,
                len(kept_sources), len(source_max), before, len(result),
            )
        return result

    @staticmethod
    def _summarize_numeric_series(values: list[int]) -> dict[str, float]:
        """
        计算数值序列的统计摘要，用于分块诊断日志。
        """
        if not values:
            return {
                "count": 0.0,
                "min": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }

        ordered = sorted(int(v) for v in values)
        n = len(ordered)

        def _pick(percent: float) -> float:
            idx = int((n - 1) * percent)
            idx = max(0, min(n - 1, idx))
            return float(ordered[idx])

        return {
            "count": float(n),
            "min": float(ordered[0]),
            "p50": _pick(0.50),
            "p90": _pick(0.90),
            "max": float(ordered[-1]),
            "avg": float(sum(ordered)) / float(n),
        }

    def _collect_parent_ids_from_collection(self, kb_id: str) -> set[str]:
        parent_ids: set[str] = set()
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=kb_id,
                limit=_BM25_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                metadata = self._extract_metadata_from_payload(payload)
                parent_id = str(metadata.get("_pc_parent_id", "") or "").strip()
                if parent_id:
                    parent_ids.add(parent_id)
            if next_offset is None:
                break
            offset = next_offset
        return parent_ids

    def _prune_parent_text_map_by_collection(self, kb_id: str) -> None:
        if not self.client.collection_exists(kb_id):
            self._clear_parent_text_map(kb_id)
            return
        current_map = self._load_parent_text_map(kb_id)
        if not current_map:
            return
        try:
            alive_parent_ids = self._collect_parent_ids_from_collection(kb_id)
        except Exception as e:
            logger.warning("收集存活parent_id失败，跳过映射裁剪: kb=%s, err=%s", kb_id, e)
            return
        pruned = {pid: txt for pid, txt in current_map.items() if pid in alive_parent_ids}
        if len(pruned) == len(current_map):
            return
        self._parent_text_cache[kb_id] = pruned
        path = self._parent_store_path(kb_id)
        try:
            path.write_text(json.dumps(pruned, ensure_ascii=False), encoding="utf-8")
            logger.info(
                "parent文本映射已裁剪: kb=%s, before=%d, after=%d",
                kb_id,
                len(current_map),
                len(pruned),
            )
        except Exception as e:
            logger.warning("写回裁剪后的parent映射失败（不影响主流程）: kb=%s, err=%s", kb_id, e)

    def _self_heal_parent_text_map(self, kb_id: str) -> None:
        """
        启动自愈：在 parent 映射缺失时，尝试从叶子 points 中按 _pc_parent_id 近似重建父上下文。
        """
        if not self.client.collection_exists(kb_id):
            return
        max_points = max(1000, int(getattr(settings, "RAG_PARENT_STORE_SELF_HEAL_MAX_POINTS", 50000) or 50000))
        scanned = 0
        offset = None
        parent_leaf_texts: dict[str, list[str]] = {}
        while True:
            records, next_offset = self.client.scroll(
                collection_name=kb_id,
                limit=_BM25_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                scanned += 1
                payload = record.payload or {}
                metadata = self._extract_metadata_from_payload(payload)
                parent_id = str(metadata.get("_pc_parent_id", "") or "").strip()
                if not parent_id:
                    continue
                text = self._extract_text_from_payload(payload)
                if not text:
                    continue
                parent_leaf_texts.setdefault(parent_id, []).append(text)
            if next_offset is None or scanned >= max_points:
                break
            offset = next_offset

        if not parent_leaf_texts:
            logger.warning("parent_store 自愈未找到可用 parent_id: kb=%s, scanned=%d", kb_id, scanned)
            return

        healed_map: dict[str, str] = {}
        for parent_id, leaf_texts in parent_leaf_texts.items():
            merged = self._merge_texts_with_overlap(leaf_texts)
            if merged:
                healed_map[parent_id] = merged

        if not healed_map:
            logger.warning("parent_store 自愈失败（无法拼装文本）: kb=%s, scanned=%d", kb_id, scanned)
            return

        self._save_parent_text_map(kb_id, healed_map)
        logger.info(
            "parent_store 自愈完成: kb=%s, scanned_points=%d, healed_parents=%d",
            kb_id,
            scanned,
            len(healed_map),
        )

    def startup_parent_store_self_check(self) -> None:
        """
        启动自检：扫描 rag_parent_store 与当前 kb_* 集合一致性。
        """
        try:
            collections = self.client.get_collections()
            kb_ids = [c.name for c in collections.collections if c.name.startswith("kb_")]
        except Exception as e:
            logger.warning("启动自检：获取集合失败，跳过 parent_store 一致性检查: %s", e)
            return

        existing_keys = {self._parent_store_key(kb_id): kb_id for kb_id in kb_ids}
        files = list(self._parent_store_dir.glob("*.json"))
        store_keys = {f.stem for f in files}

        missing_store = [existing_keys[k] for k in sorted(set(existing_keys.keys()) - store_keys)]
        orphan_store = sorted(set(store_keys) - set(existing_keys.keys()))

        empty_maps: list[str] = []
        for kb_id in kb_ids:
            parent_map = self._load_parent_text_map(kb_id)
            if not parent_map:
                empty_maps.append(kb_id)

        logger.info(
            "parent_store 启动自检: kb_collections=%d, store_files=%d, missing_store=%d, orphan_store=%d, empty_maps=%d",
            len(kb_ids),
            len(files),
            len(missing_store),
            len(orphan_store),
            len(empty_maps),
        )
        if missing_store:
            logger.warning("parent_store 缺失映射文件: %s", missing_store)
        if orphan_store:
            logger.warning("parent_store 存在孤儿映射文件（无对应kb）：%s", orphan_store)
        if empty_maps:
            logger.warning("parent_store 存在空映射（建议重建该KB）：%s", empty_maps)

        if getattr(settings, "RAG_PARENT_STORE_SELF_HEAL_ENABLED", True):
            for kb_id in missing_store + empty_maps:
                try:
                    self._self_heal_parent_text_map(kb_id)
                except Exception as e:
                    logger.warning("parent_store 自愈失败（不影响启动）: kb=%s, err=%s", kb_id, e)

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

    def _build_download_candidates(self, url: str) -> list[tuple[str, dict, str]]:
        """
        构建可回退的下载候选地址（用于预签名 URL 在容器网络差异下的鲁棒下载）。
        """
        parsed = urlparse(url)
        original_host = parsed.netloc
        host_only = original_host.split(":")[0] if original_host else ""
        port = original_host.split(":")[1] if ":" in original_host else ""
        candidates: list[tuple[str, dict, str]] = []
        seen: set[str] = set()

        def _append(candidate_url: str, candidate_headers: dict, label: str) -> None:
            key = f"{candidate_url}|{candidate_headers.get('Host', '')}"
            if key in seen:
                return
            seen.add(key)
            candidates.append((candidate_url, candidate_headers, label))

        primary_url, primary_headers = self._map_url_internally(url)
        _append(primary_url, primary_headers, "primary_mapped")

        # 可选回退（默认关闭）：仅在显式开启时尝试多内部主机，避免“隐式魔法配置”造成长期技术债。
        enable_fallback = os.getenv("S3_DOWNLOAD_FALLBACK_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        if enable_fallback and host_only in {"localhost", "127.0.0.1"}:
            raw_candidates = os.getenv(
                "S3_INTERNAL_HOST_CANDIDATES",
                "host.docker.internal,minio,lobe",
            )
            for h in [x.strip() for x in raw_candidates.split(",") if x.strip()]:
                mapped_host = f"{h}:{port}" if port else h
                mapped_url = url.replace(original_host, mapped_host)
                _append(mapped_url, {"Host": original_host}, f"fallback_host={h}")
        elif host_only in {"localhost", "127.0.0.1"}:
            logger.info("S3 下载候选回退已关闭（S3_DOWNLOAD_FALLBACK_ENABLED=false），仅使用 primary_mapped + original_url")

        # 最后回退原始 URL（用于非容器或特殊网络场景）
        _append(url, {}, "original_url")
        return candidates

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
                                metadata = nc.get("metadata", {}) or {}
                            except (json.JSONDecodeError, TypeError):
                                pass
                        if text:
                            # 轨道 A 后：摄入时已拼入 metadata 前缀，此处仅对旧数据兜底补全
                            if not (text.startswith("[TITLE]") or text.startswith("[FILE]")):
                                meta_prefix_parts: list[str] = []
                                doc_title = str(metadata.get("doc_title", "")).strip()
                                file_name = str(metadata.get("file_name", "")).strip()
                                section_title = str(metadata.get("section_title", "")).strip()
                                if doc_title:
                                    meta_prefix_parts.append(f"[TITLE] {doc_title}")
                                if file_name:
                                    meta_prefix_parts.append(f"[FILE] {file_name}")
                                if section_title:
                                    meta_prefix_parts.append(f"[SECTION] {section_title}")
                                if meta_prefix_parts:
                                    meta_prefix = " ".join(meta_prefix_parts)
                                    text = f"{meta_prefix}\n\n{text}"

                            all_nodes.append(TextNode(
                                text=text,
                                id_=str(record.id),
                                metadata=metadata,
                            ))
                            # 轨道 B：顺带收集 doc_title 供预过滤缓存
                            t = str(metadata.get("doc_title", "") or "").strip()
                            if t:
                                self._doc_title_cache.setdefault(kb_id, set()).add(t)

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

    def _get_doc_titles_for_kb(self, kb_id: str) -> set[str]:
        """
        轨道 B：获取 KB 内所有唯一的 doc_title，用于 metadata 预过滤的模糊匹配。
        优先从缓存读取，未命中时 scroll 收集并缓存。
        """
        cached = self._doc_title_cache.get(kb_id)
        if cached is not None:
            return cached
        titles: set[str] = set()
        try:
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=kb_id,
                    limit=min(500, _BM25_SCROLL_PAGE_SIZE),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    meta = self._extract_metadata_from_payload(record.payload or {})
                    t = str(meta.get("doc_title", "") or "").strip()
                    if t:
                        titles.add(t)
                if next_offset is None or len(records) < 500:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning("doc_title cache scroll failed: kb=%s, err=%s", kb_id, e)
        self._doc_title_cache[kb_id] = titles
        return titles

    def _get_matched_doc_titles(
        self, kb_id: str, doc_title_filter: str
    ) -> list[str]:
        """
        轨道 B：根据 filter 子串，从 KB 的 doc_title 集合中模糊匹配出精确值列表。
        保守策略：filter 为子串 或 doc_title 为 filter 子串 均视为匹配。
        """
        filter_lower = doc_title_filter.lower().strip()
        if not filter_lower or len(filter_lower) < 2:
            return []
        all_titles = self._get_doc_titles_for_kb(kb_id)
        if not all_titles:
            return []
        matched: list[str] = []
        for t in all_titles:
            t_lower = t.lower()
            if filter_lower in t_lower or t_lower in filter_lower:
                matched.append(t)
        return matched[:50]  # 限制数量，避免 MatchAny 过长

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
        self._doc_title_cache.pop(kb_id, None)

    def invalidate_vector_index_cache(self, kb_id: str) -> None:
        removed = self._vector_indexes.pop(kb_id, None)
        if removed is not None:
            logger.info("Vector 索引缓存已失效: collection='%s'", kb_id)

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
        collection_name = kb_id
        logger.info(f"Starting ingestion: file={file_name or file_url}, kb_id={kb_id}")

        # 1. 下载文件
        file_content: bytes | None = None
        last_error: Exception | None = None
        download_candidates = self._build_download_candidates(file_url)
        async with httpx.AsyncClient(timeout=60.0) as client:
            for candidate_url, headers, label in download_candidates:
                try:
                    response = await client.get(candidate_url, headers=headers)
                    if response.status_code >= 400:
                        response.raise_for_status()
                    file_content = response.content
                    logger.info(
                        "文件下载成功: kb=%s, source=%s, url=%s",
                        kb_id, label, candidate_url,
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "文件下载失败，尝试下一个候选: kb=%s, source=%s, url=%s, err=%s",
                        kb_id, label, candidate_url, e,
                    )
        if file_content is None:
            raise RuntimeError(f"All download candidates failed for file_url={file_url}, last_error={last_error}")

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

            from rag.chunking import build_parent_child_nodes, build_simple_nodes

            chunk_strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()
            title_aware = getattr(settings, "RAG_CHUNKING_TITLE_AWARE", False)
            father_son_ratio = max(1, int(getattr(settings, "RAG_FATHER_SON_RATIO", 3) or 3))
            splitter_type = (getattr(settings, "RAG_SPLITTER_TYPE", "token") or "token").lower()

            logger.info(
                "分块策略: strategy=%s, splitter_type=%s, title_aware=%s, chunk_size=%d, father_son_ratio=%d, num_docs=%d",
                chunk_strategy, splitter_type, title_aware, settings.RAG_CHUNK_SIZE, father_son_ratio, len(documents),
            )

            # 6. 创建索引（解析 + 嵌入 + 写入）
            ingested_count = len(documents)
            ingested_unit_label = "documents"
            if chunk_strategy == "parent_child":
                leaf_nodes, all_nodes = build_parent_child_nodes(
                    documents,
                    title_aware=title_aware,
                )
                # 记录 parent_id 到父文本的映射（单份持久化，不把父全文重复塞进每个 leaf）。
                parent_by_id = {
                    getattr(node, "node_id", ""): node
                    for node in all_nodes
                    if getattr(node, "node_id", None)
                }
                parent_text_map: dict[str, str] = {}
                enriched_leaf_count = 0
                for leaf in leaf_nodes:
                    rels = getattr(leaf, "relationships", {}) or {}
                    parent_rel = rels.get(NodeRelationship.PARENT)
                    if isinstance(parent_rel, list):
                        parent_rel = parent_rel[0] if parent_rel else None
                    parent_id = getattr(parent_rel, "node_id", None) if parent_rel else None
                    if not parent_id:
                        continue
                    parent = parent_by_id.get(parent_id)
                    if parent is None:
                        continue
                    parent_text = (getattr(parent, "text", "") or "").strip()
                    if not parent_text:
                        continue
                    parent_text_map[parent_id] = parent_text
                    leaf_meta = getattr(leaf, "metadata", {}) or {}
                    leaf_meta["_pc_parent_id"] = parent_id
                    leaf.metadata = leaf_meta
                    enriched_leaf_count += 1
                self._save_parent_text_map(collection_name, parent_text_map)

                # parent_child 诊断日志：用于分析「父子节点数量比为何偏离 father_son_ratio」
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    leaf_token_counts: list[int] = []
                    children_per_parent: dict[str, int] = {}
                    for leaf in leaf_nodes:
                        leaf_text = (getattr(leaf, "text", "") or "").strip()
                        if leaf_text:
                            leaf_token_counts.append(len(encoding.encode(leaf_text)))

                        rels = getattr(leaf, "relationships", {}) or {}
                        parent_rel = rels.get(NodeRelationship.PARENT)
                        if isinstance(parent_rel, list):
                            parent_rel = parent_rel[0] if parent_rel else None
                        pid = str(getattr(parent_rel, "node_id", "") or "").strip() if parent_rel else ""
                        if pid:
                            children_per_parent[pid] = children_per_parent.get(pid, 0) + 1

                    parent_token_counts = [
                        len(encoding.encode(text))
                        for text in parent_text_map.values()
                        if isinstance(text, str) and text.strip()
                    ]
                    child_per_parent_counts = list(children_per_parent.values())

                    leaf_stats = self._summarize_numeric_series(leaf_token_counts)
                    parent_stats = self._summarize_numeric_series(parent_token_counts)
                    cpp_stats = self._summarize_numeric_series(child_per_parent_counts)
                    observed_ratio = (
                        float(len(leaf_nodes)) / float(max(1, len(parent_text_map)))
                    )

                    logger.info(
                        "parent_child分块诊断: kb=%s, file=%s, expected_ratio=1:%d, observed_ratio=1:%.2f, leaf_nodes=%d, unique_parents=%d, leaf_tokens(avg/p50/p90/max)=%.0f/%.0f/%.0f/%.0f, parent_tokens(avg/p50/p90/max)=%.0f/%.0f/%.0f/%.0f, children_per_parent(avg/p50/p90/max)=%.2f/%.0f/%.0f/%.0f",
                        kb_id,
                        file_name or "unknown",
                        father_son_ratio,
                        observed_ratio,
                        len(leaf_nodes),
                        len(parent_text_map),
                        leaf_stats["avg"],
                        leaf_stats["p50"],
                        leaf_stats["p90"],
                        leaf_stats["max"],
                        parent_stats["avg"],
                        parent_stats["p50"],
                        parent_stats["p90"],
                        parent_stats["max"],
                        cpp_stats["avg"],
                        cpp_stats["p50"],
                        cpp_stats["p90"],
                        cpp_stats["max"],
                    )
                except Exception as diag_err:
                    logger.warning("parent_child分块诊断日志生成失败（不影响主流程）: kb=%s, err=%s", kb_id, diag_err)

                logger.info(
                    "parent_child索引父信息写入: leaf_nodes=%d, enriched_with_parent_id=%d, unique_parents=%d",
                    len(leaf_nodes),
                    enriched_leaf_count,
                    len(parent_text_map),
                )
                ingested_count = len(leaf_nodes)
                ingested_unit_label = "leaf_nodes"
                storage_context.docstore.add_documents(all_nodes)

                # 轨道 A：metadata 拼入 chunk text，使 embedding 包含 doc_title 等，便于检索
                self._prepend_metadata_prefix_to_nodes(leaf_nodes)

                index = VectorStoreIndex(
                    nodes=leaf_nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=False,
                )
            else:
                simple_nodes = build_simple_nodes(documents, title_aware=title_aware)
                ingested_count = len(simple_nodes)
                ingested_unit_label = "nodes"
                # 轨道 A：metadata 拼入 chunk text，使 embedding 包含 doc_title 等，便于检索
                self._prepend_metadata_prefix_to_nodes(simple_nodes)
                index = VectorStoreIndex(
                    nodes=simple_nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                    show_progress=False,
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

            logger.info(
                "Successfully ingested %d %s (from %d documents) into collection '%s'",
                ingested_count,
                ingested_unit_label,
                len(documents),
                collection_name,
            )
            return ingested_count

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
            self.invalidate_vector_index_cache(kb_id)
            self._clear_parent_text_map(kb_id)

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
            self.invalidate_vector_index_cache(kb_id)
            # 3. 按当前集合存活 parent_id 裁剪映射，避免父映射无限增长/陈旧
            self._prune_parent_text_map_by_collection(kb_id)

            # 4. 从 PostgreSQL 元数据中删除文件记录
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
        extra_filters: dict | None = None,
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
            similarity_top_k: 若传入则覆盖 RAG_RECALL_TOP_K/RAG_CONTEXT_TOP_K，否则使用思路一分离配置

        Returns:
            str: 格式化的检索结果，每个段落带有 [Knowledge Segment N] 标记
        """
        recall_pool_size = settings.RAG_RECALL_TOP_K or settings.RAG_DEFAULT_TOP_K
        context_top_k = settings.RAG_CONTEXT_TOP_K or settings.RAG_DEFAULT_TOP_K
        if similarity_top_k is not None:
            recall_pool_size = context_top_k = similarity_top_k
        if not kb_ids:
            return ""

        hybrid_enabled = settings.RAG_HYBRID_SEARCH
        chunk_strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()
        mode_label = "hybrid(vector+BM25)" if hybrid_enabled else "vector-only"
        logger.info(
            f"知识库检索: query='{query_str[:50]}...', kb_ids={kb_ids}, "
            f"recall_pool={recall_pool_size}, context_top_k={context_top_k}, mode={mode_label}"
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
        # 轨道 P3：合并多轮指代解析得到的 extra_filters（优先使用）
        if extra_filters:
            inferred_filters = {**inferred_filters, **extra_filters}
            logger.info("Merged extra_filters (e.g. from referent resolution): %s", extra_filters)

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
                    index = self._vector_indexes.get(kb_id)
                    if index is None:
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
                        logger.info(
                            "parent_child索引缓存未命中：kb=%s，从向量库重建索引（可能缺少父子docstore关系）",
                            kb_id,
                        )

                    base_multiplier = 3
                    recall_top_k = recall_pool_size * base_multiplier
                    if _looks_like_english_title_query(effective_query):
                        recall_top_k = max(recall_top_k, recall_pool_size * 6)
                    recall_top_k = min(recall_top_k, 80)

                    # 轨道 B：若有 doc_title filter，构建 Qdrant 预过滤
                    vector_store_kwargs: dict = {}
                    if inferred_filters:
                        dt = str(inferred_filters.get("doc_title", "") or "").strip()
                        if dt:
                            match_list = self._get_matched_doc_titles(kb_id, dt)
                            if match_list:
                                from rag.search.filters import build_qdrant_doc_title_filter
                                qf = build_qdrant_doc_title_filter(match_list)
                                if qf is not None:
                                    vector_store_kwargs["qdrant_filters"] = qf
                                    logger.info(
                                        "parent_child: applying doc_title pre-filter, match_count=%d",
                                        len(match_list),
                                    )

                    retriever = index.as_retriever(
                        similarity_top_k=recall_top_k,
                        vector_store_kwargs=vector_store_kwargs if vector_store_kwargs else {},
                    )
                    vector_nodes = await retriever.aretrieve(effective_query)

                    bm25_nodes: list[NodeWithScore] | None = None
                    if hybrid_enabled:
                        # BM25 首次构建需要同步 scroll Qdrant；用 run_in_executor
                        # 将其移入线程池，避免阻塞 asyncio event loop。
                        loop = asyncio.get_event_loop()
                        bm25, corpus_size = await loop.run_in_executor(
                            None, self._get_or_build_bm25, kb_id, recall_top_k
                        )
                        if bm25 is not None and corpus_size > 0:
                            bm25_nodes = bm25.retrieve(effective_query)

                    from rag.search.fusion import reciprocal_rank_fusion

                    fused_nodes = vector_nodes
                    if hybrid_enabled and bm25_nodes is not None:
                        bm25_weight = settings.RAG_BM25_WEIGHT
                        fused_nodes = reciprocal_rank_fusion(
                            vector_nodes,
                            bm25_nodes,
                            recall_top_k,
                            bm25_weight=bm25_weight,
                        )
                        fused_nodes = fused_nodes[:recall_pool_size]
                    else:
                        fused_nodes = vector_nodes[:recall_pool_size]

                    vector_ms = (time.perf_counter() - t0) * 1000

                    final_leaf_nodes = fused_nodes
                    if settings.RAG_RERANK_ENABLED:
                        rerank_top_k = min(settings.RAG_RERANK_TOP_K, context_top_k)
                        t_rerank = time.perf_counter()
                        final_leaf_nodes = self._rerank_nodes(
                            fused_nodes, query_str, top_k=rerank_top_k
                        )
                        rerank_ms = (time.perf_counter() - t_rerank) * 1000
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索+Rerank(parent_child): kb=%s, vector=%d(%.0fms), rerank=%d(%.0fms), total=%.0fms",
                            kb_id, len(fused_nodes), vector_ms,
                            len(final_leaf_nodes), rerank_ms, total_ms,
                        )
                    else:
                        q_lower = query_str.lower()
                        content_phrase = _longest_content_phrase(query_str)
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

                            if content_phrase:
                                full_text = (getattr(nws.node, "text", "") or "").strip()
                                if content_phrase.lower() in full_text.lower():
                                    boost += 0.6 * (nws.score or 0.01)

                            boosted_nodes.append(NodeWithScore(
                                node=nws.node, score=nws.score + boost,
                            ))

                        boosted_nodes.sort(key=lambda x: x.score, reverse=True)
                        final_leaf_nodes = boosted_nodes[:context_top_k]
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索(启发式重排, parent_child): kb=%s, results=%d, total=%.0fms",
                            kb_id, len(final_leaf_nodes), total_ms,
                        )

                    # 轨道 C：metadata 后过滤（parent_child 接入 filters，与 simple 统一）
                    if inferred_filters:
                        from rag.search.filters import apply_search_filters_to_nodes
                        final_leaf_nodes = apply_search_filters_to_nodes(
                            final_leaf_nodes, inferred_filters
                        )

                    min_score = getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0) or 0.0
                    if min_score > 0 and final_leaf_nodes:
                        best = max(nws.score for nws in final_leaf_nodes)
                        if best < min_score:
                            logger.info(
                                f"知识库 {kb_id} 最高相关分 {best:.4f} 低于阈值 {min_score}，跳过返回片段"
                            )
                            continue

                    # 来源级别过滤：过滤掉最高分明显低于最佳来源的整个来源文档，
                    # 防止无关文档的段落混入 LLM 上下文导致答非所问。
                    final_leaf_nodes = self._apply_source_level_filter(final_leaf_nodes, kb_id)
                    if not final_leaf_nodes:
                        continue

                    try:
                        from llama_index.core.schema import NodeRelationship, TextNode
                    except Exception:
                        NodeRelationship = None  # type: ignore
                        from llama_index.core.schema import TextNode

                    parent_nodes_map: dict[str, NodeWithScore] = {}
                    docstore = getattr(index, "storage_context", None)
                    docstore = getattr(docstore, "docstore", None)
                    parent_text_map = self._load_parent_text_map(kb_id)
                    parent_token_budget = self._resolve_parent_segment_token_budget()
                    parent_resolved_count = 0
                    fallback_parent_store_count = 0
                    fallback_leaf_count = 0
                    parent_truncated_count = 0

                    for nws in final_leaf_nodes:
                        node = nws.node
                        leaf_text = (getattr(node, "text", "") or "").strip()
                        parent_id = None
                        parent_node = None

                        if NodeRelationship is not None and docstore is not None:
                            rels = getattr(node, "relationships", {}) or {}
                            parent_rel = rels.get(NodeRelationship.PARENT) if rels else None
                            if parent_rel:
                                if isinstance(parent_rel, list):
                                    parent_rel = parent_rel[0] if parent_rel else None
                                candidate_id = getattr(parent_rel, "node_id", None)
                                if candidate_id:
                                    try:
                                        parent_node = docstore.get_node(candidate_id)
                                        parent_id = candidate_id
                                        parent_resolved_count += 1
                                    except Exception:
                                        parent_node = None

                        if parent_node is None:
                            # 服务重启后 docstore 关系可能不可用；使用 parent_id 到
                            # 持久化 parent_text 映射恢复父上下文，避免退化为碎片 leaf。
                            meta = getattr(node, "metadata", {}) or {}
                            parent_id_meta = str(meta.get("_pc_parent_id", "") or "").strip()
                            parent_text = parent_text_map.get(parent_id_meta, "").strip() if parent_id_meta else ""
                            if parent_text:
                                parent_node = TextNode(text=parent_text, metadata=meta)
                                parent_id = parent_id_meta or getattr(node, "node_id", None) or str(id(node))
                                fallback_parent_store_count += 1
                            else:
                                parent_node = node
                                parent_id = getattr(node, "node_id", None) or str(id(node))
                                fallback_leaf_count += 1

                        parent_raw_text = (getattr(parent_node, "text", "") or "").strip() if parent_node is not None else ""
                        clipped_parent_text, was_truncated = self._clip_parent_text_for_budget(
                            parent_raw_text,
                            leaf_text,
                            parent_token_budget,
                        )
                        if clipped_parent_text:
                            if was_truncated:
                                parent_truncated_count += 1
                            parent_meta = getattr(parent_node, "metadata", None) or getattr(node, "metadata", {}) or {}
                            parent_node = TextNode(text=clipped_parent_text, metadata=parent_meta)
                        else:
                            parent_node = node
                            parent_id = getattr(node, "node_id", None) or str(id(node))

                        existing = parent_nodes_map.get(parent_id)
                        score = float(nws.score or 0.0)
                        if existing is None or score > existing.score:
                            parent_nodes_map[parent_id] = NodeWithScore(
                                node=parent_node, score=score,
                            )

                    parent_nodes = sorted(
                        parent_nodes_map.values(),
                        key=lambda x: x.score, reverse=True,
                    )

                    logger.info(
                        "parent_child父节点提升统计: kb=%s, leaf_in=%d, parent_resolved=%d, fallback_parent_store=%d, fallback_leaf=%d, unique_segments=%d, docstore_ready=%s, parent_store_size=%d, parent_budget_tokens=%d, parent_truncated=%d",
                        kb_id,
                        len(final_leaf_nodes),
                        parent_resolved_count,
                        fallback_parent_store_count,
                        fallback_leaf_count,
                        len(parent_nodes),
                        bool(docstore is not None),
                        len(parent_text_map),
                        parent_token_budget,
                        parent_truncated_count,
                    )

                    from rag.postprocess.segments import build_segments_from_nodes

                    segments, segment_count = build_segments_from_nodes(
                        parent_nodes, start_index=segment_count,
                    )
                    all_segments.extend(segments)
                else:
                    # ---- simple：单层分块 + hybrid_search_single_kb ----
                    vector_store = QdrantVectorStore(
                        collection_name=kb_id,
                        aclient=self.aclient,
                        client=None,
                    )

                    base_multiplier = 3
                    recall_top_k = recall_pool_size * base_multiplier
                    if _looks_like_english_title_query(effective_query):
                        recall_top_k = max(recall_top_k, recall_pool_size * 6)
                    recall_top_k = min(recall_top_k, 80)

                    bm25 = None
                    corpus_size = 0
                    if hybrid_enabled:
                        loop = asyncio.get_event_loop()
                        bm25, corpus_size = await loop.run_in_executor(
                            None, self._get_or_build_bm25, kb_id, recall_top_k
                        )

                    # 轨道 B：若有 doc_title filter，计算匹配列表用于 Qdrant 预过滤
                    doc_title_match_list: list[str] | None = None
                    if inferred_filters:
                        dt = str(inferred_filters.get("doc_title", "") or "").strip()
                        if dt:
                            doc_title_match_list = self._get_matched_doc_titles(kb_id, dt)

                    search_req = SearchRequest(
                        query=effective_query,
                        kb_ids=[kb_id],
                        top_k=recall_top_k,
                        filters=inferred_filters,
                        doc_title_match_list=doc_title_match_list,
                    )

                    search_result = await hybrid_search_single_kb(
                        search_req,
                        kb_id=kb_id,
                        vector_store=vector_store,
                        embed_model=self.embed_model,
                        bm25_retriever=bm25 if hybrid_enabled and bm25 is not None else None,
                        corpus_size=corpus_size,
                    )

                    from llama_index.core.schema import TextNode

                    vector_nodes: list[NodeWithScore] = []
                    for hit in search_result.hits:
                        node = TextNode(text=hit.text, metadata=hit.metadata)
                        vector_nodes.append(NodeWithScore(node=node, score=hit.score))

                    vector_ms = (time.perf_counter() - t0) * 1000

                    final_nodes = vector_nodes
                    if settings.RAG_RERANK_ENABLED:
                        rerank_top_k = min(settings.RAG_RERANK_TOP_K, context_top_k)
                        t_rerank = time.perf_counter()
                        final_nodes = self._rerank_nodes(
                            vector_nodes, query_str, top_k=rerank_top_k
                        )
                        rerank_ms = (time.perf_counter() - t_rerank) * 1000
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索+Rerank: kb=%s, vector=%d(%.0fms), rerank=%d(%.0fms), total=%.0fms",
                            kb_id, len(vector_nodes), vector_ms,
                            len(final_nodes), rerank_ms, total_ms,
                        )
                    else:
                        q_lower = query_str.lower()
                        content_phrase = _longest_content_phrase(query_str)
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

                            if content_phrase:
                                full_text = (getattr(nws.node, "text", "") or "").strip()
                                if content_phrase.lower() in full_text.lower():
                                    boost += 0.6 * (nws.score or 0.01)

                            boosted_nodes.append(NodeWithScore(
                                node=nws.node, score=nws.score + boost,
                            ))

                        boosted_nodes.sort(key=lambda x: x.score, reverse=True)
                        final_nodes = boosted_nodes[:context_top_k]
                        total_ms = (time.perf_counter() - t0) * 1000
                        logger.info(
                            "检索(启发式重排): kb=%s, results=%d, total=%.0fms",
                            kb_id, len(final_nodes), total_ms,
                        )

                    min_score = getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0) or 0.0
                    if min_score > 0 and final_nodes:
                        best = max(nws.score for nws in final_nodes)
                        if best < min_score:
                            logger.info(
                                f"知识库 {kb_id} 最高相关分 {best:.4f} 低于阈值 {min_score}，跳过返回片段"
                            )
                            continue

                    # 来源级别过滤：过滤掉最高分明显低于最佳来源的整个来源文档。
                    final_nodes = self._apply_source_level_filter(final_nodes, kb_id)
                    if not final_nodes:
                        continue

                    from rag.postprocess.segments import build_segments_from_nodes

                    segments, segment_count = build_segments_from_nodes(
                        final_nodes, start_index=segment_count,
                    )
                    all_segments.extend(segments)

            except Exception as e:
                logger.error(f"Error querying collection {kb_id}: {str(e)}", exc_info=True)
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
