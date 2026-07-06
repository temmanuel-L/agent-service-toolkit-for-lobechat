from typing import Any
import re

from pydantic import BaseModel, Field

from langchain_core.messages import ChatMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from agents.multimodal_input_processor import MultimodalInputProcessor


def coerce_optional_str(value: Any) -> str | None:
    """将 tool/state 返回值规范为 Optional[str]（供 Pydantic mode=before 或业务读取）。"""
    if value is None or value == "null" or value == "":
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, list):
        for item in value:
            s = coerce_optional_str(item)
            if s:
                return s
        return None
    if isinstance(value, dict):
        for key in ("city", "destination", "place", "name"):
            val = value.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    s = str(value).strip()
    return s or None


def coerce_state_str(value: Any) -> str:
    """从 state 安全读取标量字符串槽位（兼容误存的 list/dict）。"""
    return coerce_optional_str(value) or ""


def normalize_date_optional_str(value: Any) -> str | None:
    """将常见中文/短格式日期规范为 YYYY-MM-DD（如 2026-7-10）。"""
    s = coerce_optional_str(value)
    if not s:
        return None
    m = re.match(r"^(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return s


class CustomData(BaseModel):
    "Custom data being sent by an agent"

    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        writer(self.to_langchain())


def get_silent_config(config: RunnableConfig) -> RunnableConfig:
    """
    获取一个静默配置，用于中间节点调用 LLM 时屏蔽流式输出，
    同时保留 Langfuse 等 Trace 回调。
    """
    # 1. 浅拷贝配置对象，防止修改影响到原始 config（LangGraph 的 state 传递安全性）
    new_config = config.copy()

    # 2. 确保 tags 存在且包含 skip_stream
    if "tags" not in new_config:
        new_config["tags"] = []
    else:
        # 如果原始 tags 是不可变的 tuple，转为 list
        new_config["tags"] = list(new_config["tags"])

    if "skip_stream" not in new_config["tags"]:
        new_config["tags"].append("skip_stream")

    return new_config


def build_interrupt_text_message_update(
    user_response: Any,
    *,
    extraction_field: str = "extraction_input_text",
) -> dict[str, Any]:
    """
    规范化 interrupt 回合的用户输入：
    - messages 仅写入文本（避免 image_url 落入 checkpointer）
    - extraction 字段仅写文本，供调用方按需二次增强
    """
    text = MultimodalInputProcessor.extract_text_only(user_response)
    update: dict[str, Any] = {extraction_field: text}
    if text:
        update["messages"] = [HumanMessage(content=text)]
    return update
