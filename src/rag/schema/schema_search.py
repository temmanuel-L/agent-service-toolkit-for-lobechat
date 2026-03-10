"""
RAG 检索（Search）阶段的数据模型。

用于在「业务接口层」与「具体存储实现(Qdrant/Postgres 等)」之间提供一个抽象层。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional  # noqa: F401 - List used in doc_title_match_list

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """抽象的检索请求。"""

    query: str = Field(description="最终用于检索的查询文本（可能已经过 HyDE/改写）。")
    kb_ids: List[str] = Field(
        description="需要检索的知识库 ID 列表（通常对应 Qdrant collection 名称）。",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        description="期望返回的候选段落数量上限（召回阶段）。",
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="结构化过滤条件（如按作者/文件名筛选），将映射为底层 Qdrant filter。",
    )
    doc_title_match_list: List[str] | None = Field(
        default=None,
        description="轨道 B：预过滤时匹配的 doc_title 精确值列表，用于构建 Qdrant MatchAny filter。",
    )


class SearchHit(BaseModel):
    """单条检索命中记录。"""

    text: str = Field(description="命中的段落文本。")
    score: float = Field(description="检索得分（语义/关键词/融合后的分数）。")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="与该段落相关的元数据（包括文档级与 chunk 级）。",
    )


class SearchResult(BaseModel):
    """检索阶段统一的返回结构。"""

    hits: List[SearchHit] = Field(
        default_factory=list,
        description="按相关性排序的命中列表。",
    )


__all__ = [
    "SearchRequest",
    "SearchHit",
    "SearchResult",
]

