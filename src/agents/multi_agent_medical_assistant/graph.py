# -*- coding: utf-8 -*-
"""
医疗多智能体主图编排。

流程：prepare_input -> analyze_input -> route_to_agent -> 各 agent -> check_validation
     -> human_validation (interrupt) -> apply_guardrails -> END
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

from agents.multi_agent_medical_assistant.guardrails import LocalGuardrails
from agents.multi_agent_medical_assistant.image_analysis import ImageClassifier
from agents.multi_agent_medical_assistant.state import (
    MedicalAgentState,
    get_input_text,
)
from agents.multi_agent_medical_assistant.sub_agents import (
    run_brain_tumor_agent,
    run_chest_xray_agent,
    run_conversation_agent,
    run_rag_agent,
    run_skin_lesion_agent,
    run_web_search_agent,
)
from utils.log_utils import get_logger

logger = get_logger(__name__)

# ============================================================================
# 配置
# ============================================================================

CONFIDENCE_THRESHOLD = 0.85

DECISION_PROMPT = """You are an intelligent medical triage system. Route user queries to the appropriate agent.

Available agents:
1. CONVERSATION_AGENT - General chat, greetings, non-medical questions.
2. RAG_AGENT - Medical knowledge from established literature (brain tumor, COVID chest X-ray, skin lesion, etc.).
3. WEB_SEARCH_PROCESSOR_AGENT - Recent medical developments, time-sensitive info.
4. BRAIN_TUMOR_AGENT - Brain MRI image analysis.
5. CHEST_XRAY_AGENT - Chest X-ray image analysis.
6. SKIN_LESION_AGENT - Skin lesion image analysis.

Rules:
- No image: route to CONVERSATION_AGENT.
- With medical image: route to the matching vision agent based on image_type.
- Medical knowledge question: RAG_AGENT.
- Recent/outbreak question: WEB_SEARCH_PROCESSOR_AGENT.

Respond in JSON: {{"agent": "AGENT_NAME", "reasoning": "...", "confidence": 0.95}}"""


class AgentDecision(BaseModel):
    agent: str = Field(description="Agent name")
    reasoning: str = Field(description="Reasoning")
    confidence: float = Field(description="Confidence 0-1")


# ============================================================================
# 编排节点
# ============================================================================

guardrails = LocalGuardrails()
image_classifier = ImageClassifier()


def prepare_input(state: MedicalAgentState, config: RunnableConfig | None = None) -> dict:
    """从 messages 提取 current_input，支持 agent_config.image_path。"""
    messages = state.get("messages") or []
    current_input = state.get("current_input")
    if current_input is not None:
        return {}

    text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            text = getattr(m, "content", "") or ""
            break

    configurable = (config or {}).get("configurable", {})
    image_path = configurable.get("image_path")
    if image_path:
        current_input = {"text": text or "用户上传了医学图像，请分析。", "image": image_path}
    else:
        current_input = text

    return {"current_input": current_input}


def analyze_input(state: MedicalAgentState, config: RunnableConfig | None = None) -> dict:
    """Guardrails 检查，若有图片则分类。"""
    current_input = state.get("current_input")
    if current_input is None:
        return {}

    text = get_input_text(current_input)
    if text:
        is_ok, msg = guardrails.check_input(text)
        if not is_ok:
            out_msg = msg if isinstance(msg, AIMessage) else AIMessage(content=str(msg))
            return {
                "messages": [out_msg],
                "output": out_msg,
                "agent_name": "INPUT_GUARDRAILS",
                "bypass_routing": True,
            }

    has_image = False
    image_type = None
    if isinstance(current_input, dict) and current_input.get("image"):
        has_image = True
        model = get_model(
            (config or {}).get("configurable", {}).get("model", settings.DEFAULT_MODEL)
        )
        result = image_classifier.classify_image(
            current_input["image"],
            vision_model=model,
        )
        image_type = result.get("image_type", "UNKNOWN")
        logger.info("[analyze_input] image_type=%s", image_type)

    return {
        "has_image": has_image,
        "image_type": image_type,
        "bypass_routing": False,
    }


def route_to_agent(state: MedicalAgentState, config: RunnableConfig | None = None) -> dict:
    """LLM 决策路由。"""
    if state.get("bypass_routing"):
        return {"next": "apply_guardrails", "agent_name": state.get("agent_name")}

    current_input = state.get("current_input")
    messages = state.get("messages") or []
    has_image = state.get("has_image", False)
    image_type = state.get("image_type") or "None"

    text = get_input_text(current_input)
    recent = ""
    for m in messages[-6:]:
        if isinstance(m, HumanMessage):
            recent += f"User: {getattr(m, 'content', '')}\n"
        elif isinstance(m, AIMessage):
            recent += f"Assistant: {getattr(m, 'content', '')}\n"

    prompt = f"""User query: {text}
Recent context:
{recent}
Has image: {has_image}
Image type: {image_type}

Which agent should handle this? Respond in JSON: {{\"agent\": \"AGENT_NAME\", \"reasoning\": \"...\", \"confidence\": 0.95}}"""

    model = get_model(
        (config or {}).get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    )
    parser = JsonOutputParser(pydantic_object=AgentDecision)
    chain = (
        ChatPromptTemplate.from_messages([("system", DECISION_PROMPT), ("human", "{input}")])
        | model
        | parser
    )
    decision = chain.invoke({"input": prompt})

    agent_name = decision.get("agent", "CONVERSATION_AGENT")
    confidence = float(decision.get("confidence", 0.9))
    needs_human_validation = confidence < CONFIDENCE_THRESHOLD
    logger.info(
        "[route_to_agent] agent=%s confidence=%s needs_validation=%s",
        agent_name,
        confidence,
        needs_human_validation,
    )

    return {
        "agent_name": agent_name,
        "next": agent_name,
        "needs_human_validation": needs_human_validation,
    }


def check_validation(state: MedicalAgentState) -> str:
    """是否需要人工确认。"""
    if state.get("needs_human_validation"):
        return "human_validation"
    return "apply_guardrails"


def human_validation_node(state: MedicalAgentState) -> dict:
    """人机协同：interrupt 等待用户确认。"""
    output = state.get("output")
    content = output.content if isinstance(output, AIMessage) else str(output or "")
    prompt = f"{content}\n\n**Human Validation Required:** Please confirm (Yes/No)."
    user_response = interrupt(prompt)
    return {"messages": [AIMessage(content=user_response)]}


def apply_guardrails(state: MedicalAgentState, config: RunnableConfig | None = None) -> dict:
    """输出安全过滤并写入 messages。"""
    output = state.get("output")
    current_input = state.get("current_input")
    text = get_input_text(current_input)
    if not output:
        return {}
    sanitized = guardrails.check_output(output, text)
    msg = AIMessage(content=sanitized)
    return {"messages": [msg]}


# ============================================================================
# 图构建
# ============================================================================


def build_graph():
    workflow = StateGraph(MedicalAgentState)

    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", lambda s: {"_check": check_validation(s)})
    workflow.add_node("human_validation", human_validation_node)
    workflow.add_node("apply_guardrails", apply_guardrails)

    workflow.set_entry_point("prepare_input")
    workflow.add_edge("prepare_input", "analyze_input")
    workflow.add_edge("analyze_input", "route_to_agent")

    def _route_from_decision(s: MedicalAgentState) -> str:
        if s.get("bypass_routing"):
            return "apply_guardrails"
        return s.get("next", "CONVERSATION_AGENT")

    workflow.add_conditional_edges(
        "route_to_agent",
        _route_from_decision,
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "apply_guardrails": "apply_guardrails",
        },
    )

    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_conditional_edges(
        "check_validation",
        lambda s: s.get("_check", "apply_guardrails"),
        {"human_validation": "human_validation", "apply_guardrails": "apply_guardrails"},
    )
    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)

    return workflow.compile()


multi_agent_medical_assistant = build_graph()

# try:
#     graph_obj = multi_agent_medical_assistant.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('multi_agent_medical_assistant.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")