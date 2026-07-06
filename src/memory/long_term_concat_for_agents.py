"""长期记忆与 agent 对话 messages 的拼装（不写 checkpoint messages channel）。"""

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import RemoveMessage

from core.settings import settings

# 与 memory/long_term.py abuild_system_message 中 additional_kwargs.source 保持一致
LONG_TERM_MEMORY_SOURCE = "long_term_memory"
LONG_TERM_MEMORY_CONTENT_KEY = "long_term_memory_content"
LONG_TERM_MEMORY_SKIP_KEY = "long_term_memory_skip"


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not config:
        return {}
    return config.get("configurable") or {}


def is_long_term_memory_message(msg: BaseMessage) -> bool:
    if not isinstance(msg, SystemMessage):
        return False
    return (msg.additional_kwargs or {}).get("source") == LONG_TERM_MEMORY_SOURCE


def should_attach_long_term_memory(config: RunnableConfig | None) -> bool:
    if not settings.LONG_TERM_MEMORY_ENABLED:
        return False
    cfg = _configurable(config)
    if cfg.get(LONG_TERM_MEMORY_SKIP_KEY):
        return False
    content = cfg.get(LONG_TERM_MEMORY_CONTENT_KEY)
    return bool(content and str(content).strip())


def strip_legacy_long_term_messages(messages: list[BaseMessage]) -> list[RemoveMessage]:
    """返回 RemoveMessage 列表，用于清理 checkpoint 中历史长期记忆 SystemMessage。"""
    removals: list[RemoveMessage] = []
    for msg in messages:
        if is_long_term_memory_message(msg) and msg.id:
            removals.append(RemoveMessage(id=msg.id))
    return removals


def _conversation_without_long_term(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if not is_long_term_memory_message(m)]


def _conversation_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """去掉 checkpoint 遗留 LTM 与所有 SystemMessage（system 统一由 build_llm_messages 生成）。"""
    return [m for m in _conversation_without_long_term(messages) if not isinstance(m, SystemMessage)]


def _long_term_memory_text(config: RunnableConfig | None) -> str | None:
    if not should_attach_long_term_memory(config):
        return None
    text = str(_configurable(config)[LONG_TERM_MEMORY_CONTENT_KEY]).strip()
    return text or None


def _merge_system_text(memory: str | None, agent_system: str | None) -> str | None:
    """合并 system 文本：长期记忆在前，agent system 在后。"""
    parts: list[str] = []
    if memory:
        parts.append(memory)
    if agent_system and agent_system.strip():
        parts.append(agent_system.strip())
    return "\n\n".join(parts) if parts else None


def build_llm_messages(
    conversation: list[BaseMessage],
    config: RunnableConfig | None,
    *,
    agent_system: str | None = None,
) -> list[BaseMessage]:
    """LLM 调用前的唯一拼装出口：至多一条 SystemMessage，其余为对话轮次。"""
    cleaned = _conversation_for_llm(conversation)
    merged = _merge_system_text(_long_term_memory_text(config), agent_system)
    if merged:
        return [SystemMessage(content=merged), *cleaned]
    return cleaned


def concat_system_for_medical(
    base_system: str,
    config: RunnableConfig | None,
) -> str:
    """medical conversation_agent 使用 dict role invoke 时的 system 合并。"""
    return _merge_system_text(_long_term_memory_text(config), base_system) or base_system


async def prepare_long_term_entry(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """统一 START 节点：清理历史 LTM system，不写入新 system 到 messages。"""
    messages = state.get("messages") or []
    removals = strip_legacy_long_term_messages(messages)
    if removals:
        return {"messages": removals}
    return {}


def wrap_agent_with_long_term_entry(
    compiled: Any,
    state_schema: type = MessagesState,
) -> Any:
    """预构建 agent 外包：START → prepare_long_term → 原 compiled graph。"""
    builder = StateGraph(state_schema)
    builder.add_node("prepare_long_term", prepare_long_term_entry)
    builder.add_node("run", compiled)
    builder.add_edge(START, "prepare_long_term")
    builder.add_edge("prepare_long_term", "run")
    builder.add_edge("run", END)
    return builder.compile()
