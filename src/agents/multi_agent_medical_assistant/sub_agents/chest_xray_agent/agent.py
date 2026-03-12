# -*- coding: utf-8 -*-
"""胸片分析子智能体（占位，未来可扩展 CV 模型）。"""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.multi_agent_medical_assistant.state import MedicalAgentState


def run_chest_xray_agent(
    state: MedicalAgentState, config: RunnableConfig | None = None
) -> dict:
    """胸片分析（占位）。"""
    return {
        "output": AIMessage(
            content="Chest X-ray analysis requires specialized CV models. Please configure the model path."
        ),
        "agent_name": "CHEST_XRAY_AGENT",
        "needs_human_validation": True,
    }
