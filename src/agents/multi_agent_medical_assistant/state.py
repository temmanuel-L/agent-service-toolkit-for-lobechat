# -*- coding: utf-8 -*-
"""
医疗多智能体共享状态与工具。

避免 graph.py 与 sub_agents 之间的循环导入。
"""

from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages

# ============================================================================
# 常量
# ============================================================================

CONTEXT_LIMIT = 20

# ============================================================================
# 状态类型
# ============================================================================


class MedicalAgentState(TypedDict, total=False):
    """医疗智能体状态。"""

    messages: Annotated[List[BaseMessage], add_messages]
    current_input: Optional[Union[str, dict]]
    has_image: bool
    image_type: Optional[str]
    agent_name: Optional[str]
    output: Optional[Union[str, AIMessage]]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool
    next: Optional[str]
    _check: Optional[str]


# ============================================================================
# 共享工具
# ============================================================================


def get_input_text(current_input: Optional[Union[str, dict]]) -> str:
    """从 current_input 提取文本。"""
    if isinstance(current_input, str):
        return current_input
    if isinstance(current_input, dict):
        return current_input.get("text", "") or ""
    return ""
