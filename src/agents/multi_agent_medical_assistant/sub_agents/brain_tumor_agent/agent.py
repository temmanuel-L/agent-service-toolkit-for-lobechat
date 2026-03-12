# -*- coding: utf-8 -*-
"""脑 MRI 分析子智能体（占位，未来可扩展 CV 模型）。"""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.multi_agent_medical_assistant.state import MedicalAgentState


def run_brain_tumor_agent(
    state: MedicalAgentState, config: RunnableConfig | None = None
) -> dict:
    """脑 MRI 分析（占位）。"""
    return {
        "output": AIMessage(
            content="Brain tumor analysis requires specialized CV models. Please configure the model path."
        ),
        "agent_name": "BRAIN_TUMOR_AGENT",
        "needs_human_validation": True,
    }
