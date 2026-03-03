"""
RAG 检索结果后处理阶段的数据模型。

用于描述：
- 已经选定要给 LLM 的若干段落；
- 这些段落在最终上下文字符串中的组织方式与长度控制信息。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SegmentForLLM(BaseModel):
    """准备给 LLM 的单个段落（已过检索+Rerank）。"""

    header: str = Field(
        description="段落头部信息，例如 [Knowledge Segment N] Source: ... | Score: ...",
    )
    content: str = Field(description="段落正文文本。")


class PostprocessResult(BaseModel):
    """后处理阶段的统一输出。"""

    segments: List[SegmentForLLM] = Field(
        default_factory=list,
        description="按顺序组织好的段落列表。",
    )
    combined_text: str = Field(
        description="拼装后提供给 LLM 使用的完整上下文字符串。",
    )
    truncated: bool = Field(
        default=False,
        description="是否因为长度或段落数限制而发生截断。",
    )


__all__ = [
    "SegmentForLLM",
    "PostprocessResult",
]

