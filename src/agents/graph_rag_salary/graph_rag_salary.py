# -*- coding: utf-8 -*-
"""
Salary Graph RAG Agent

基于 Neo4j 薪酬知识图谱的单主智能体。
流程：
1. prepare_long_term → reset_turn（清空上一轮 ephemeral 字段）
2. 一级意图：闲聊 / RAG
3. 二级意图：salary_lookup | compare | industry_overview | skill_trend | fallback
4. 实体链接 + 本体 Cypher 模板检索（含降级）→ 受控生成回答
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from agents.graph_rag_salary.intent_router import (
    VALID_INTENTS,
    INTENT_PROMPT,
    heuristic_intent_with_postprocess,
    intent_result_from_slots,
    message_content_to_str,
    parse_intent_from_llm_content,
)
from agents.graph_rag_salary.neo4j_client import get_salary_neo4j_driver
from agents.graph_rag_salary.prompts import (
    ANSWER_PROMPT,
    CHAT_SYSTEM_PROMPT,
    EMPTY_CONTEXT_REPLY,
    build_l1_intent_system_prompt,
)
from agents.graph_rag_salary.query_templates import format_context_for_llm
from agents.graph_rag_salary.retrieval import run_salary_retrieval
from core import get_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry
from utils.log_utils import get_logger

logger = get_logger(__name__)

SalaryIntentLiteral = Literal[
    "salary_lookup",
    "compare",
    "industry_overview",
    "skill_trend",
    "fallback",
]


# =============================================================================
# 状态定义（精简）
# =============================================================================
class GraphRAGSalaryState(MessagesState, total=False):
    """薪酬 Graph RAG 智能体状态。跨轮只保留 messages；业务字段每轮 reset。"""

    current_user_text: Optional[str]
    intent: Optional[Literal["chat", "rag"]]
    salary_intent: Optional[SalaryIntentLiteral]
    intent_slots: Optional[dict[str, Any]]
    retrieval_payload: Optional[dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]


# =============================================================================
# 结构化意图模型
# =============================================================================
class ChatVsRagIntent(BaseModel):
    """判断用户问题是否需要薪酬知识图谱 RAG。"""

    intent: Literal["chat", "rag"] = Field(
        ...,
        description=(
            "chat=闲聊或与薪酬图谱无关；"
            "rag=涉及行业/岗位薪资、对比、行业概览、技能或市场趋势"
        ),
    )


# =============================================================================
# 辅助
# =============================================================================
def _last_human_message_text(state: GraphRAGSalaryState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _make_node_result(answer: str, **extra: Any) -> dict:
    """写 final_answer + AIMessage；默认出口清空 retrieval_payload。"""
    out: dict[str, Any] = {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "retrieval_payload": None,
    }
    out.update(extra)
    return out


def _turn_reset_fields() -> dict[str, Any]:
    return {
        "current_user_text": None,
        "intent": None,
        "salary_intent": None,
        "intent_slots": None,
        "retrieval_payload": None,
        "final_answer": None,
        "error": None,
    }


# =============================================================================
# 节点
# =============================================================================
async def reset_turn_node(state: GraphRAGSalaryState, config: RunnableConfig) -> dict:
    """每轮入口清空上一轮 ephemeral 业务字段，避免 checkpoint 残留。"""
    return _turn_reset_fields()


async def chat_vs_rag_intent_node(state: GraphRAGSalaryState, config: RunnableConfig) -> dict:
    """一级意图：chat vs rag。"""
    user_text = _last_human_message_text(state)
    if not user_text:
        return {"intent": "chat", "current_user_text": ""}

    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([ChatVsRagIntent])

    try:
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(
            [
                SystemMessage(content=build_l1_intent_system_prompt()),
                HumanMessage(content=user_text),
            ],
            config,
        )
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = dict(tool_call.get("args") or {})
            parsed = ChatVsRagIntent(**args)
            intent = parsed.intent
        else:
            content = (
                (response.content or "").strip().lower()
                if hasattr(response, "content")
                else ""
            )
            intent = content if content in ("chat", "rag") else "chat"

        logger.info("一级意图识别结果: %s", intent)
        return {"intent": intent, "current_user_text": user_text}
    except Exception as e:
        logger.error("一级意图识别失败: %s", e)
        return {"intent": "chat", "current_user_text": user_text, "error": str(e)}


async def salary_sub_intent_node(state: GraphRAGSalaryState, config: RunnableConfig) -> dict:
    """二级意图：参考 classify_intent；LLM 调用必须带 skip_stream，避免 JSON 漏到前端。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)

    try:
        llm = get_model(model_name).with_config(tags=["skip_stream"])
        prompt = INTENT_PROMPT.format(query=user_text)
        response = await llm.ainvoke([HumanMessage(content=prompt)], config)
        content = message_content_to_str(
            response.content if hasattr(response, "content") else response
        )
        result = parse_intent_from_llm_content(content, user_text)
        salary_intent = result.intent if result.intent in VALID_INTENTS else "fallback"
        slots = result.to_slots()
        logger.info(
            "二级意图识别结果: %s | industries=%s | jobs=%s | region=%s | groups=%s",
            salary_intent,
            slots.get("industries"),
            slots.get("job_titles"),
            slots.get("region"),
            slots.get("compare_groups"),
        )
        return {
            "salary_intent": salary_intent,
            "intent_slots": slots,
        }
    except Exception as e:
        logger.error("二级意图识别失败，使用规则兜底: %s", e)
        result = heuristic_intent_with_postprocess(user_text)
        return {
            "salary_intent": result.intent if result.intent in VALID_INTENTS else "fallback",
            "intent_slots": result.to_slots(),
            "error": str(e),
        }


async def retrieve_node(state: GraphRAGSalaryState, config: RunnableConfig) -> dict:
    """实体链接 + 模板检索（含降级）。使用薪酬专用 Neo4j（默认 7691）。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    salary_intent = state.get("salary_intent")
    intent_slots = state.get("intent_slots")
    intent = intent_result_from_slots(salary_intent, intent_slots)

    try:
        driver = get_salary_neo4j_driver()
        payload = await asyncio.to_thread(
            run_salary_retrieval,
            driver,
            user_text,
            intent,
        )
        mode = payload.get("mode") if isinstance(payload, dict) else None
        data = payload.get("data") if isinstance(payload, dict) else None
        data_len = len(data) if isinstance(data, list) else 0
        # compare 时看各组 results 是否全空
        if mode == "compare" and isinstance(data, list):
            nonempty = sum(1 for g in data if (g or {}).get("results"))
            logger.info(
                "retrieval done | mode=%s | groups=%s | nonempty_groups=%s",
                mode,
                data_len,
                nonempty,
            )
        else:
            logger.info("retrieval done | mode=%s | data_rows=%s", mode, data_len)
        return {"retrieval_payload": payload}
    except Exception as e:
        logger.error("检索失败: %s", e)
        return {
            "retrieval_payload": {
                "mode": "fallback_chunk",
                "data": [],
                "extra": {"note": f"检索异常: {e}"},
            },
            "error": str(e),
        }


def _payload_has_usable_data(payload: dict | None) -> bool:
    """判断检索 payload 是否有可供生成的事实（对齐参考：有 data 就交给 LLM）。"""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False
    mode = payload.get("mode")
    if mode == "compare":
        return any((g or {}).get("results") for g in data)
    if mode == "fallback_chunk":
        return any(
            (item.get("content") if isinstance(item, dict) else str(item) or "").strip()
            for item in data
        )
    return True


async def generate_node(state: GraphRAGSalaryState, config: RunnableConfig) -> dict:
    """根据本轮 retrieval_payload 生成回答；出口清空 payload。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    payload = state.get("retrieval_payload")
    salary_intent = state.get("salary_intent") or "fallback"
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)

    if not _payload_has_usable_data(payload):
        logger.info(
            "generate skip | empty payload | mode=%s",
            (payload or {}).get("mode") if isinstance(payload, dict) else None,
        )
        return _make_node_result(EMPTY_CONTEXT_REPLY)

    context = format_context_for_llm(salary_intent, payload)
    if not (context or "").strip():
        return _make_node_result(EMPTY_CONTEXT_REPLY)

    try:
        llm = get_model(model_name).with_config(tags=["skip_stream"])
        prompt = ANSWER_PROMPT.format(query=user_text, context=context)
        resp = await llm.ainvoke([HumanMessage(content=prompt)], config)
        content = resp.content if hasattr(resp, "content") else str(resp)
        answer = content if isinstance(content, str) else str(content)
        answer = (answer or "").strip() or EMPTY_CONTEXT_REPLY
        logger.info("generate done | answer_chars=%s", len(answer))
        return _make_node_result(answer)
    except Exception as e:
        logger.error("答案生成失败: %s", e)
        return {
            **_make_node_result("检索已完成，但生成回答时出错，请稍后重试。"),
            "error": str(e),
        }


async def chat_node(state: GraphRAGSalaryState, config: RunnableConfig) -> dict:
    """闲聊分支；防御性清空 RAG 槽位。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    try:
        messages = build_llm_messages(
            [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=user_text),
            ],
            config,
        )
        resp = await llm.ainvoke(messages, config)
        answer = resp.content if hasattr(resp, "content") else str(resp)
        return _make_node_result(
            answer if isinstance(answer, str) else str(answer),
            salary_intent=None,
            intent_slots=None,
        )
    except Exception as e:
        logger.error("LLM 对话失败: %s", e)
        return _make_node_result(
            "抱歉，暂时无法回复。",
            salary_intent=None,
            intent_slots=None,
            error=str(e),
        )


# =============================================================================
# 条件路由
# =============================================================================
def route_after_first_intent(state: GraphRAGSalaryState) -> str:
    if state.get("intent") == "rag":
        return "rag"
    return "chat"


# =============================================================================
# 构建图
# =============================================================================
def build_graph_rag_salary():
    graph = StateGraph(GraphRAGSalaryState)

    graph.add_node("prepare_long_term", prepare_long_term_entry)
    graph.add_node("reset_turn", reset_turn_node)
    graph.add_node("l1_intent", chat_vs_rag_intent_node)
    graph.add_node("l2_intent", salary_sub_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "prepare_long_term")
    graph.add_edge("prepare_long_term", "reset_turn")
    graph.add_edge("reset_turn", "l1_intent")

    graph.add_conditional_edges(
        "l1_intent",
        route_after_first_intent,
        {"rag": "l2_intent", "chat": "chat"},
    )
    graph.add_edge("l2_intent", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("chat", END)

    return graph.compile()


graph_rag_salary = build_graph_rag_salary()
