"""
Search 级别的简单过滤与过滤推断工具。

当前目标（第一阶段）：
- 支持从自然语言问题中，粗略推断出需要按文档标题过滤的场景（例如“某某论文的作者是谁”）；
- 对 SearchResult.hits 进行基于 metadata 的后过滤（目前仅支持 doc_title 相关过滤）；
- 所有过滤逻辑都是“保守启用”：宁可不过滤，也不要误杀召回。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List

from rag.schema.schema_search import SearchHit
from utils.log_utils import get_logger

if TYPE_CHECKING:
    from llama_index.core.schema import NodeWithScore
    from qdrant_client import models as qdrant_models

logger = get_logger(__name__)


def infer_filters_from_query(query: str) -> Dict[str, Any]:
    """
    从自然语言问题中推断简单的过滤条件。

    当前仅支持：
    - 论文/报告标题类的 doc_title 过滤，例如：
      - “《xxx》这篇论文的作者是谁？”
      - “名为xxx的报告的作者是谁？”
    """
    q = (query or "").strip()
    if not q:
        return {}

    # 1) 匹配《xxx》样式的标题
    title_match = re.search(r"[《「『](.+?)[》」』]", q)
    if title_match:
        title = title_match.group(1).strip()
        if title:
            logger.info("Query filter inference: detected doc_title from 《》: %s", title)
            return {"doc_title": title}

    # 2) 匹配“名为xxx的论文/报告/文档/文件”
    name_match = re.search(r"名为(.+?)的(论文|报告|文档|文件)", q)
    if name_match:
        title = name_match.group(1).strip("《》「」『』“”\"' ").strip()
        if title:
            logger.info("Query filter inference: detected doc_title from '名为...的X': %s", title)
            return {"doc_title": title}

    # 3) 匹配"XXX的参考文献/作者/摘要/目录"（支持中英文标题，含空格）
    suffix_match = re.search(
        r"(.+?)(的参考文献|的作者|的摘要|的目录|的引用)",
        q,
        re.IGNORECASE | re.DOTALL,
    )
    if suffix_match:
        title = suffix_match.group(1).strip()
        # 过滤过短或纯标点的噪声
        if len(title) >= 2 and re.search(r"[\w\u4e00-\u9fff]", title):
            logger.info(
                "Query filter inference: detected doc_title from 'XXX的Y': %s",
                title[:60],
            )
            return {"doc_title": title}

    # 4) 匹配"XXX 参考文献"（英文标题后直接跟空格+参考文献，无"的"）
    ref_match = re.search(r"(.+?)\s+(参考文献|references)\s*$", q, re.IGNORECASE | re.DOTALL)
    if ref_match:
        title = ref_match.group(1).strip()
        if len(title) >= 2 and re.search(r"[\w\u4e00-\u9fff]", title):
            logger.info(
                "Query filter inference: detected doc_title from 'XXX 参考文献': %s",
                title[:60],
            )
            return {"doc_title": title}

    return {}


def apply_search_filters_to_hits(
    hits: List[SearchHit],
    filters: Dict[str, Any] | None,
) -> List[SearchHit]:
    """
    在 SearchResult.hits 上应用基于 metadata 的后过滤。

    当前仅支持：
    - doc_title: 命中过滤值作为子串的 doc_title 或 file_name 保留，其余丢弃。
    """
    if not hits or not filters:
        return hits

    doc_title_filter = str(filters.get("doc_title", "") or "").strip()
    if not doc_title_filter:
        return hits

    f_lower = doc_title_filter.lower()
    kept: List[SearchHit] = []
    for hit in hits:
        meta = hit.metadata or {}
        title = str(meta.get("doc_title", "") or "").lower()
        fname = str(meta.get("file_name", "") or "").lower()

        if f_lower in title or f_lower in fname:
            kept.append(hit)

    if not kept:
        logger.info(
            "Search filter(doc_title=%s) filtered out all hits (%d). "
            "Falling back to unfiltered hits.",
            doc_title_filter,
            len(hits),
        )
        return hits

    logger.info(
        "Search filter(doc_title=%s) applied: %d -> %d hits",
        doc_title_filter,
        len(hits),
        len(kept),
    )
    return kept


def apply_search_filters_to_nodes(
    nodes: List["NodeWithScore"],
    filters: Dict[str, Any] | None,
) -> List["NodeWithScore"]:
    """
    在 NodeWithScore 列表上应用基于 metadata 的后过滤（与 apply_search_filters_to_hits 逻辑一致）。
    用于 parent_child 等直接使用 NodeWithScore 的检索路径。
    """
    if not nodes or not filters:
        return nodes

    doc_title_filter = str(filters.get("doc_title", "") or "").strip()
    if not doc_title_filter:
        return nodes

    f_lower = doc_title_filter.lower()
    kept: List["NodeWithScore"] = []
    for nws in nodes:
        meta = getattr(nws.node, "metadata", {}) or {}
        title = str(meta.get("doc_title", "") or "").lower()
        fname = str(meta.get("file_name", "") or "").lower()

        if f_lower in title or f_lower in fname:
            kept.append(nws)

    if not kept:
        logger.info(
            "Search filter(doc_title=%s) filtered out all nodes (%d). "
            "Falling back to unfiltered nodes.",
            doc_title_filter,
            len(nodes),
        )
        return nodes

    logger.info(
        "Search filter(doc_title=%s) applied to nodes: %d -> %d",
        doc_title_filter,
        len(nodes),
        len(kept),
    )
    return kept


# 轨道 C/P3：指代词列表，用于多轮指代解析
_REFERENT_PATTERNS = (
    "这篇文章", "该论文", "那份合同", "那个文件", "这份文档",
    "这篇文档", "该文档", "那份报告", "这个文件", "该报告",
)


def resolve_referent_from_messages(
    query: str,
    messages: list,
    max_lookback: int = 10,
) -> str | None:
    """
    轨道 P3：从对话历史中解析「这篇文章」「那份合同」等指代词对应的 doc_title。
    规则：在最近 N 条消息中，查找最后一次出现的 doc_title（从检索结果或模型回复中提取）。
    若 query 包含指代词且解析成功，返回 doc_title；否则返回 None。
    """
    q = (query or "").strip()
    if not q or not messages:
        return None
    if not any(p in q for p in _REFERENT_PATTERNS):
        return None

    # 从最近消息中提取可能的 doc_title（启发式：检索结果中的 Source: xxx）
    import re
    for msg in reversed(messages[-max_lookback:]):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        elif isinstance(msg, dict):
            content = str(msg.get("content", "") or "")
        if not content:
            continue
        # 匹配 "Source: xxx" 或 "[Knowledge Segment N] Source: xxx"
        match = re.search(r"Source:\s*([^\n|]+)", content)
        if match:
            title = match.group(1).strip()
            if len(title) >= 2 and re.search(r"[\w\u4e00-\u9fff]", title):
                return title
    return None


def build_qdrant_doc_title_filter(doc_title_match_list: List[str]) -> Any:
    """
    轨道 B：根据 doc_title 精确值列表构建 Qdrant MatchAny filter。
    用于预过滤，仅检索匹配文档的 chunk。
    """
    if not doc_title_match_list:
        return None
    try:
        from qdrant_client import models as qdrant_models
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.doc_title",
                    match=qdrant_models.MatchAny(any=doc_title_match_list),
                )
            ]
        )
    except Exception as e:
        logger.warning("Build Qdrant doc_title filter failed: %s", e)
        return None


__all__ = [
    "infer_filters_from_query",
    "apply_search_filters_to_hits",
    "apply_search_filters_to_nodes",
    "build_qdrant_doc_title_filter",
    "resolve_referent_from_messages",
]

