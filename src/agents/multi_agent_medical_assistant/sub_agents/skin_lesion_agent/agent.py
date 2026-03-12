# -*- coding: utf-8 -*-
"""皮肤病变分析子智能体（占位，未来可扩展 CV 模型）。"""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.multi_agent_medical_assistant.state import MedicalAgentState


def run_skin_lesion_agent(
    state: MedicalAgentState, config: RunnableConfig | None = None
) -> dict:
    """皮肤病变分析（占位）。"""
    return {
        "output": AIMessage(
            content="Skin lesion analysis requires specialized CV models. Please configure the model path."
        ),
        "agent_name": "SKIN_LESION_AGENT",
        "needs_human_validation": True,
    }
