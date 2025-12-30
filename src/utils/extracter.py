"""
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2025-09-10 10:58:02
LastEditTime: 2025-10-24 15:58:56
LastEditors: uyplayer
Description: 消息类型相关的提取/查询工具
FilePath: /simulation-intelligent-assistant/app/toolkit/extracter.py
@copyright Copyright (c) 2025 by 3040
"""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_MESSAGE_TYPE_REGISTRY: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
    "chat": ChatMessage,
    "tool": ToolMessage,
    "function": FunctionMessage,
}


def get_message_type_registry() -> Mapping[str, type[BaseMessage]]:
    """
    返回内置支持的消息类型映射，方便根据名称拿到具体的消息类

    Returns:
        Mapping[str, Type[BaseMessage]]: 键为规范化的小写类型名，值为对应的 BaseMessage 子类
    """

    return _MESSAGE_TYPE_REGISTRY.copy()


def build_message(message_type: str, content: Any, **message_kwargs) -> BaseMessage:
    """
    根据消息类型字符串快速构造对应的 BaseMessage 实例

    Args:
        message_type (str): 消息类型标识（system/human/ai/chat/tool/function）
        content (Any): 消息内容，不同类型可传 str / list / dict
        **message_kwargs: 额外的消息字段，例如 tool_call_id、name 等

    Returns:
        BaseMessage: 初始化完成的消息对象
    """
    normalized_type = message_type.lower()
    message_cls = _MESSAGE_TYPE_REGISTRY.get(normalized_type)
    if message_cls is None:
        raise ValueError(
            f"未知的消息类型: {message_type}，必须是 {_MESSAGE_TYPE_REGISTRY.keys()} 之一"
        )

    try:
        return message_cls(content=content, **message_kwargs)
    except TypeError as exc:
        raise ValueError(f"{message_cls.__name__} 初始化失败，可能缺少必要参数: {exc}") from exc


def extract_messages(
    messages: Sequence[BaseMessage], extract_type: str | None = None
) -> list[BaseMessage]:
    """
    根据消息类型获取对应的消息列表如果不传类型，返回所有消息的副本

    Args:
        messages (Sequence[BaseMessage]): 原始消息列表
        extract_type (str | None): 指定的消息类型（如 system/human/ai/chat/tool/function）

    Returns:
        list[BaseMessage]: 过滤后的消息列表
    """

    if extract_type is None:
        return list(messages)

    normalized_type = extract_type.lower()
    if normalized_type not in _MESSAGE_TYPE_REGISTRY:
        raise ValueError(
            f"extract_type 必须是 {_MESSAGE_TYPE_REGISTRY.keys()} 之一，当前为: {extract_type}"
        )

    target_cls = _MESSAGE_TYPE_REGISTRY[normalized_type]
    return [message for message in messages if isinstance(message, target_cls)]
