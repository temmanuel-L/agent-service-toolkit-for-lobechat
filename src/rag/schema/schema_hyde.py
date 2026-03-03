"""
RAG Query 改写 / HyDE 阶段的数据模型。

HyDE（Hypothetical Document Embeddings）通常包含：
- 原始查询；
- 若干个由 LLM 生成的「假想文档」或等价改写查询；
- 对这些假想文档的向量表示及其在检索阶段的使用方式。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class HyDEConfig(BaseModel):
    """HyDE/Query 改写相关配置。"""

    enabled: bool = Field(
        default=False,
        description="是否启用 HyDE/Query 改写。",
    )
    num_variants: int = Field(
        default=0,
        ge=0,
        description="为同一个用户问题生成多少种改写/假想文档。",
    )


class QueryVariant(BaseModel):
    """单个改写后的查询或假想文档。"""

    text: str = Field(description="改写后的查询文本或假想文档内容。")
    source: str = Field(
        default="hyde",
        description="改写来源标记，例如 original/hyde/paraphrase 等。",
    )


class HyDEResult(BaseModel):
    """HyDE 阶段输出：包含原始 query 及若干改写版本。"""

    original_query: str = Field(description="原始用户查询。")
    variants: List[QueryVariant] = Field(
        default_factory=list,
        description="用于后续检索的改写/假想文档列表（包括或不包括原始查询）。",
    )


__all__ = [
    "HyDEConfig",
    "QueryVariant",
    "HyDEResult",
]

