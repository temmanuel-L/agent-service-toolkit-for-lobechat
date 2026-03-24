# -*- coding: utf-8 -*-
"""通用对话子智能体。"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.utils import get_silent_config
from core import get_model, settings

from agents.multi_agent_medical_assistant.state import (
    CONTEXT_LIMIT,
    MedicalAgentState,
    get_input_text,
)
from agents.multi_agent_medical_assistant.utils import compress_messages, should_compress


def run_conversation_agent(
    state: MedicalAgentState, config: RunnableConfig | None = None
) -> dict:
    """通用对话。"""
    current_input = state.get("current_input")
    messages = state.get("messages") or []
    text = get_input_text(current_input)

    recent = ""
    for m in messages[-CONTEXT_LIMIT:]:
        if isinstance(m, HumanMessage):
            recent += f"User: {getattr(m, 'content', '')}\n"
        elif isinstance(m, AIMessage):
            recent += f"Assistant: {getattr(m, 'content', '')}\n"

    sys = """You are an AI Medical Conversation Assistant. Handle general chat and medical questions.
Be professional, accurate. For serious concerns, recommend consulting a healthcare professional.
Do not provide diagnoses or prescriptions.
Always respond in the SAME LANGUAGE as the user's message (Chinese if user writes in Chinese, English if in English).

IMPORTANT: Do NOT repeat, quote, or include the [长期记忆] / long-term memory content in your response.
Use it only as background context. For simple greetings (e.g. 你好), respond briefly without listing memory."""

    model = get_model((config or {}).get("configurable", {}).get("model", settings.DEFAULT_MODEL))
    to_invoke = messages[-CONTEXT_LIMIT:] if len(messages) > CONTEXT_LIMIT else messages
    if should_compress(to_invoke):
        to_invoke = compress_messages(to_invoke, llm=model)
    silent_cfg = get_silent_config(config or {})
    resp = model.invoke(
        [{"role": "system", "content": sys}]
        + [{"role": "user", "content": f"Context:\n{recent}\n\nUser: {text}"}],
        config=silent_cfg,
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {"output": AIMessage(content=content), "agent_name": "CONVERSATION_AGENT"}
