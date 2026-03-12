"""
医疗多智能体辅助工具：上下文与 token 控制。

提供 token 估算、消息压缩、RAG 上下文截断等，供各节点在调用 LLM 前控制上下文长度。
"""

from __future__ import annotations

from typing import List, Union

import tiktoken

from langchain_core.messages import BaseMessage

from utils.log_utils import get_logger

logger = get_logger(__name__)

DEFAULT_ENCODING = "cl100k_base"
DEFAULT_MAX_TOKENS_PER_CALL = 12_000  # 为输出预留余量，假设模型 16K 上下文


def estimate_tokens(
    text: str,
    encoding_name: str = DEFAULT_ENCODING,
) -> int:
    """基于 tiktoken 估算文本 token 数。"""
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # 粗略回退


def estimate_messages_tokens(
    messages: List[BaseMessage],
    encoding_name: str = DEFAULT_ENCODING,
) -> int:
    """估算消息列表的 token 数。"""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            total += estimate_tokens(content, encoding_name)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content", "")
                    if isinstance(text, str):
                        total += estimate_tokens(text, encoding_name)
        total += 4  # 每条消息的 overhead
    return total


def should_compress(
    messages_or_context: Union[List[BaseMessage], str],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_CALL,
    encoding_name: str = DEFAULT_ENCODING,
) -> bool:
    """判断是否超出 token 限制，需要压缩。"""
    if isinstance(messages_or_context, str):
        return estimate_tokens(messages_or_context, encoding_name) > max_tokens
    return estimate_messages_tokens(messages_or_context, encoding_name) > max_tokens


def compress_messages(
    messages: List[BaseMessage],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_CALL,
    encoding_name: str = DEFAULT_ENCODING,
    llm=None,
) -> List[BaseMessage]:
    """
    当消息列表超限时，用 LLM 摘要压缩为更短的对话历史。

    若未提供 llm 或压缩失败，则截断保留最近的消息（按 token 估算）。
    """
    if not should_compress(messages, max_tokens, encoding_name):
        return messages

    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = "请将以下对话历史压缩为简洁的摘要，保留关键信息（患者主诉、诊断结论、用药建议等），便于后续对话继续。\n\n"
            for m in messages:
                role = "用户" if isinstance(m, HumanMessage) else "助手"
                content = getattr(m, "content", "").strip() or ""
                if content:
                    prompt += f"{role}: {content}\n"
            prompt += "\n压缩后的摘要："
            resp = llm.invoke(prompt)
            summary = resp.content if hasattr(resp, "content") else str(resp)
            return [SystemMessage(content=f"[对话摘要] {summary}")]
        except Exception as e:
            logger.warning("compress_messages LLM 摘要失败，回退截断: %s", e)

    # 回退：截断保留最近消息
    enc = tiktoken.get_encoding(encoding_name)
    kept: List[BaseMessage] = []
    current = 0
    for m in reversed(messages):
        content = getattr(m, "content", "") or ""
        if isinstance(content, str):
            tok = len(enc.encode(content)) + 4
        else:
            tok = 100
        if current + tok > max_tokens:
            break
        kept.insert(0, m)
        current += tok
    return kept if kept else messages[:1]


def compress_rag_context(
    context: str,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_CALL,
    encoding_name: str = DEFAULT_ENCODING,
) -> str:
    """
    对 RAG 检索结果做 token 级截断。

    按段落（\\n\\n 分隔）逐个累积，直到接近 max_tokens。
    """
    if not context or not should_compress(context, max_tokens, encoding_name):
        return context

    enc = tiktoken.get_encoding(encoding_name)
    segments = [s for s in context.split("\n\n") if s.strip()]
    kept: List[str] = []
    current = 0
    for seg in segments:
        tok = len(enc.encode(seg))
        if current + tok > max_tokens:
            break
        kept.append(seg)
        current += tok
    result = "\n\n".join(kept)
    if len(kept) < len(segments):
        logger.info(
            "RAG 上下文已截断: 原 %d 段 -> %d 段, max_tokens=%d",
            len(segments),
            len(kept),
            max_tokens,
        )
    return result


__all__ = [
    "estimate_tokens",
    "estimate_messages_tokens",
    "should_compress",
    "compress_messages",
    "compress_rag_context",
    "DEFAULT_MAX_TOKENS_PER_CALL",
]
