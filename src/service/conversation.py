"""
会话管理模块
"""
from datetime import datetime
from typing import Any

from fastapi import HTTPException, status
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig

from agents import AgentGraph, get_agent
from memory.postgres import get_postgres_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    ConversationInput,
    ConversationsList,
    DeleteConversationInput,
    DeleteConversationOutput,
)

from utils.log_utils import get_logger
from .utils import langchain_to_chat_message

logger = get_logger(__name__)


async def history_handler(input: ChatHistoryInput) -> ChatHistory:
    """
    获取指定会话线程的历史消息记录并转换为聊天历史格式

    该函数通过线程ID获取代理的状态快照，从中提取消息历史，
    并将其转换为适用于前端展示的聊天历史格式

    Args:
        input (ChatHistoryInput): 包含代理标识和线程ID的输入参数对象
            - agent: 代理标识，用于获取对应的代理实例
            - thread_id: 会话线程唯一标识符

    Raises:
        HTTPException: 当获取会话历史过程中发生异常时抛出HTTP 500错误

    Returns:
        ChatHistory: 包含转换后聊天消息列表的聊天历史对象
            - messages: 转换后的聊天消息列表，每条消息都符合ChatMessage格式
    """
    # 获取指定代理实例
    agent: AgentGraph = get_agent(input.agent)
    try:
        # 根据线程ID获取代理状态快照
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        # 从状态中提取消息列表
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        # 将LangChain消息格式转换为聊天消息格式
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def conversations_handler(input: ConversationInput) -> list[ConversationsList]:
    """
    获取指定用户的会话列表，并按 thread_id 聚合出最新摘要

    Args:
        input (ConversationInput): 输入的参数

    Returns:
        list[ConversationsList]: 对话列表
    """

    def _extract_messages(checkpoint: Any) -> list[ChatMessage]:
        if not isinstance(checkpoint, dict):
            return []
        channel_values = checkpoint.get("channel_values") or {}
        raw_messages = channel_values.get("messages")
        if raw_messages is None:
            start_section = channel_values.get("__start__")
            if isinstance(start_section, dict):
                raw_messages = start_section.get("messages")

        messages: list[ChatMessage] = []
        if isinstance(raw_messages, list):
            for raw in raw_messages:
                try:
                    messages.append(langchain_to_chat_message(raw))
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("跳过无法解析的 checkpoint 消息：%s", exc)
        return messages

    def _short_text(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _title(messages: list[ChatMessage]) -> str:
        for msg in messages:
            if msg.type == "human" and msg.content:
                return _short_text(msg.content, 40)
        if messages and messages[0].content:
            return _short_text(messages[0].content, 40)
        return "新对话"

    def _preview(messages: list[ChatMessage]) -> str | None:
        for msg in reversed(messages):
            if msg.content:
                return _short_text(msg.content, 80)
        return None

    def _parse_ts(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        value = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            logger.debug("无法解析 checkpoint 时间戳：%s", value)
            return None

    agent: AgentGraph = get_agent(input.agent)

    checkpointer = getattr(agent, "checkpointer", None)
    if not checkpointer or not hasattr(checkpointer, "alist"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='系统没有开启"状态持久化（persistence）"，因此无法查看历史对话（conversation listing）',
        )

    # 获取存储对象用于查询话题
    store = getattr(agent, "store", None)

    grouped: dict[str, ConversationsList] = {}
    conversations_iter = checkpointer.alist(config=None, filter={"user_id": input.user_id})
    async for checkpoint_tuple in conversations_iter:
        config = checkpoint_tuple.config or {}
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            continue

        checkpoint_data = checkpoint_tuple.checkpoint
        messages = _extract_messages(checkpoint_data)
        updated_at = _parse_ts(
            checkpoint_data.get("ts") if isinstance(checkpoint_data, dict) else None
        )

        # 尝试从存储中获取话题作为标题
        title = "新对话"
        if store:
            try:
                async with get_postgres_store() as store:
                    topic_obj = await store.aget(
                        namespace=(input.user_id, "conversation_topic"), key=thread_id
                    )
                    if topic_obj and isinstance(topic_obj.value, dict):
                        title = topic_obj.value.get("title", "新对话")
                    else:
                        # 如果没有保存的话题，则使用原来的逻辑
                        title = _title(messages)
            except Exception as e:
                logger.debug(f"获取话题失败: {e}")
                # 如果获取话题失败，则使用原来的逻辑
                title = _title(messages)
        else:
            # 如果没有存储功能，则使用原来的逻辑
            title = _title(messages)

        summary = ConversationsList(
            user_id=input.user_id,
            thread_id=thread_id,
            title=title,
            last_message_preview=_preview(messages),
            updated_at=updated_at,
        )

        existing = grouped.get(thread_id)
        if existing is None or (
            (summary.updated_at or datetime.min) >= (existing.updated_at or datetime.min)
        ):
            grouped[thread_id] = summary

    return sorted(
        grouped.values(),
        key=lambda item: item.updated_at or datetime.min,
        reverse=True,
    )


async def delete_conversation_handler(
    input: DeleteConversationInput,
) -> DeleteConversationOutput:
    """
    删除指定用户的某个对话；若未指定 thread_id 则清空该用户的所有历史对话

    Args:
        input (DeleteConversationInput): 输入参数

    Raises:
        HTTPException: 异常
    Returns:
        DeleteConversationOutput: 返回结果
    """

    agent: AgentGraph = get_agent(input.agent)
    checkpointer = getattr(agent, "checkpointer", None)
    if not checkpointer or not hasattr(checkpointer, "adelete_thread"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='系统没有开启"状态持久化（persistence）"，因此无法删除历史对话',
        )

    # 获取存储对象用于删除话题
    store = getattr(agent, "store", None)

    deleted_threads: list[str] = []

    if input.thread_id:
        config = RunnableConfig(
            configurable={
                "thread_id": input.thread_id,
                "user_id": input.user_id,
            }
        )
        try:
            existing = await checkpointer.aget_tuple(config=config)
        except Exception as exc:
            logger.error("检查对话是否存在失败：%s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="校验历史对话失败",
            )

        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到指定对话，或对话不属于该用户",
            )
        deleted_threads = [input.thread_id]
    else:
        if not hasattr(checkpointer, "alist"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='系统没有开启"状态持久化（persistence）"，因此无法删除历史对话',
            )
        try:
            conversations_iter = checkpointer.alist(config=None, filter={"user_id": input.user_id})
            async for checkpoint_tuple in conversations_iter:
                cp_config = checkpoint_tuple.config or {}
                configurable = cp_config.get("configurable") or {}
                thread_id = configurable.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    deleted_threads.append(thread_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("读取用户历史对话失败：%s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="读取历史对话失败",
            )

        if not deleted_threads:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="该用户没有可删除的历史对话",
            )

    for thread_id in deleted_threads:
        try:
            await checkpointer.adelete_thread(thread_id=thread_id)

            # 同时删除存储在 store 中的话题
            if store:
                try:
                    await store.adelete(
                        namespace=(input.user_id, "conversation_topic"), key=thread_id
                    )
                except Exception as exc:
                    logger.warning(f"删除话题失败 {thread_id}: {exc}")
                    # 不中断主流程，仅记录警告日志
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("删除历史对话失败 %s：%s", thread_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="删除历史对话失败",
            )

    return DeleteConversationOutput(deleted_thread_ids=deleted_threads)