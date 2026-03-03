"""
RAG 检索结果截断与拼装工具（token 级别）。

目标：
- 在不改变当前段落分隔与标记格式的前提下，提供更精细的 token 级长度控制；
- 便于根据 RAG_CHUNK_SIZE 与 RAG_DEFAULT_TOP_K 推导合适的最大 token 数；
- 后续可作为统一的 postprocess 能力被 SearchKnowledgeTool 与其他 RAG 工具复用。
"""

from __future__ import annotations

from typing import Tuple

import tiktoken

from rag.utils import RAG_SEGMENT_SEPARATOR


def _split_segments(content: str) -> list[str]:
    if not content:
        return []
    return content.split(RAG_SEGMENT_SEPARATOR)


def truncate_rag_result_token_aware(
    content: str,
    max_tokens: int,
    max_segments: int,
    encoding_name: str = "cl100k_base",
) -> Tuple[str, bool]:
    """
    基于 token 的 RAG 结果截断。

    Args:
        content: 原始检索结果字符串（由若干 segment 使用 RAG_SEGMENT_SEPARATOR 拼接而成）
        max_tokens: 允许的最大 token 数（粗略上限，按整个结果计算）
        max_segments: 允许的最大段落数
        encoding_name: tiktoken 编码名称，默认 cl100k_base（兼容 OpenAI/大部分 embedding）

    Returns:
        (truncated_content, truncated_flag)
    """
    if not content:
        return content, False

    segments = _split_segments(content)
    truncated_by_segments = False
    if len(segments) > max_segments:
        segments = segments[:max_segments]
        truncated_by_segments = True

    encoding = tiktoken.get_encoding(encoding_name)

    # 先整体估算一次，如果已经在 token 限制内则直接返回
    joined = RAG_SEGMENT_SEPARATOR.join(segments)
    total_tokens = len(encoding.encode(joined))
    if total_tokens <= max_tokens:
        return joined, truncated_by_segments

    # 否则按段落逐个累积，直到接近 max_tokens
    kept_segments: list[str] = []
    current_tokens = 0
    for seg in segments:
        seg_tokens = len(encoding.encode(seg))
        if current_tokens + seg_tokens > max_tokens:
            break
        kept_segments.append(seg)
        current_tokens += seg_tokens

    truncated = truncated_by_segments or len(kept_segments) < len(segments)
    result = RAG_SEGMENT_SEPARATOR.join(kept_segments)
    return result, truncated


__all__ = [
    "truncate_rag_result_token_aware",
]

