"""
Search 级别的简单过滤与过滤推断工具。

当前目标（第一阶段）：
- 支持从自然语言问题中，粗略推断出需要按文档标题过滤的场景（例如“某某论文的作者是谁”）；
- 对 SearchResult.hits 进行基于 metadata 的后过滤（目前仅支持 doc_title 相关过滤）；
- 所有过滤逻辑都是“保守启用”：宁可不过滤，也不要误杀召回。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from rag.schema.schema_search import SearchHit
from utils.log_utils import get_logger

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


__all__ = [
    "infer_filters_from_query",
    "apply_search_filters_to_hits",
]

