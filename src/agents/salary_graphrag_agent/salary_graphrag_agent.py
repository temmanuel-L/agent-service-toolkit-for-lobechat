# -*- coding: utf-8 -*-
"""
Salary GraphRAG Agent（plan-and-execute + 并行召回/闲聊）

流程：
1. prepare_long_term → reset_turn
2. plan（多 Task 拆解，skip_stream）
3. 缺 Industry → clarify → END
4. parallel_execute（business / chitchat / chunk 并行）
5. generate（综合最终答案，可流式）
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

from agents.salary_graphrag_agent.neo4j_client import get_salary_graphrag_neo4j_driver
from agents.salary_graphrag_agent.planner import (
    PlanResult,
    Task,
    build_plan_prompt,
    check_plan_slots,
    load_industry_tree,
    plan_from_llm_content,
    plan_query,
)
from agents.salary_graphrag_agent.prompts import ANSWER_PROMPT, EMPTY_CONTEXT_REPLY
from agents.salary_graphrag_agent.query_templates import format_context_for_llm
from agents.salary_graphrag_agent.retrieval import parallel_retrieve
from core import get_model, settings
from memory.long_term_concat_for_agents import prepare_long_term_entry
from utils.log_utils import get_logger

logger = get_logger(__name__)


class SalaryGraphRAGState(MessagesState, total=False):
    """跨轮只保留 messages；业务字段每轮 reset。"""

    current_user_text: Optional[str]
    plan: Optional[dict[str, Any]]
    needs_clarify: Optional[bool]
    clarify_message: Optional[str]
    retrieval_payload: Optional[dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]


def _last_human_message_text(state: SalaryGraphRAGState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _make_node_result(answer: str, **extra: Any) -> dict:
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
        "plan": None,
        "needs_clarify": None,
        "clarify_message": None,
        "retrieval_payload": None,
        "final_answer": None,
        "error": None,
    }


def _plan_from_state(state: SalaryGraphRAGState) -> PlanResult | None:
    raw = state.get("plan")
    if not isinstance(raw, dict):
        return None
    tasks = []
    for t in raw.get("tasks") or []:
        if not isinstance(t, dict):
            continue
        tasks.append(
            Task(
                type=t.get("type") or "chitchat",
                chain=t.get("chain") or "none",
                industries=list(t.get("industries") or []),
                sub_sectors=list(t.get("sub_sectors") or []),
                job_titles=list(t.get("job_titles") or []),
                area=t.get("area"),
                raw_question=t.get("raw_question") or "",
            )
        )
    return PlanResult(tasks=tasks, raw=raw.get("raw") or {})


def _message_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(str(c.get("text", c)))
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content)


# =============================================================================
# 节点
# =============================================================================
async def reset_turn_node(state: SalaryGraphRAGState, config: RunnableConfig) -> dict:
    return _turn_reset_fields()


async def plan_node(state: SalaryGraphRAGState, config: RunnableConfig) -> dict:
    """多 Task 规划；LLM 带 skip_stream，结果不写 messages。"""
    user_text = _last_human_message_text(state)
    if not user_text:
        empty = PlanResult(tasks=[Task(type="chitchat", chain="none", raw_question="")])
        return {
            "current_user_text": "",
            "plan": empty.to_dict(),
            "needs_clarify": False,
            "clarify_message": None,
        }

    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    try:
        driver = get_salary_graphrag_neo4j_driver()
        tree = await asyncio.to_thread(load_industry_tree, driver)
        llm = get_model(model_name).with_config(tags=["skip_stream"])
        prompt = build_plan_prompt(user_text, tree)
        response = await llm.ainvoke([HumanMessage(content=prompt)], config)
        content = _message_content_to_str(
            response.content if hasattr(response, "content") else response
        )
        plan = plan_from_llm_content(user_text, content, tree)
    except Exception as e:
        logger.error("plan 失败，启发式兜底: %s", e)
        try:
            driver = get_salary_graphrag_neo4j_driver()
            plan = await asyncio.to_thread(plan_query, user_text, driver, None, None)
        except Exception as e2:
            logger.error("启发式 plan 也失败: %s", e2)
            plan = PlanResult(
                tasks=[Task(type="chitchat", chain="none", raw_question=user_text)],
                raw={"source": "fallback_chitchat"},
            )
            return {
                "current_user_text": user_text,
                "plan": plan.to_dict(),
                "needs_clarify": False,
                "clarify_message": None,
                "error": str(e),
            }

    ok, clarify_msg = check_plan_slots(plan)
    task_summary = [
        {
            "type": t.type,
            "chain": t.chain,
            "industries": t.industries,
            "sub_sectors": t.sub_sectors,
            "job_titles": t.job_titles,
            "area": t.area,
        }
        for t in plan.tasks
    ]
    logger.info("plan | needs_clarify=%s | %s", not ok, task_summary)

    return {
        "current_user_text": user_text,
        "plan": plan.to_dict(),
        "needs_clarify": not ok,
        "clarify_message": clarify_msg,
    }


async def clarify_node(state: SalaryGraphRAGState, config: RunnableConfig) -> dict:
    msg = state.get("clarify_message") or (
        "请补充您想查询的一级行业或二级行业，以便检索薪酬知识图谱。"
    )
    return _make_node_result(msg, needs_clarify=True)


async def parallel_execute_node(state: SalaryGraphRAGState, config: RunnableConfig) -> dict:
    """并行：business 召回 + chitchat LLM + chunk hybrid。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    plan = _plan_from_state(state)
    if plan is None:
        return {
            "retrieval_payload": {"mode": "empty", "data": []},
            "error": "missing plan",
        }

    try:
        driver = get_salary_graphrag_neo4j_driver()
        payload = await parallel_retrieve(driver, user_text, plan, config)
        biz_n = len(payload.get("business_tasks") or [])
        chat_n = len(payload.get("chitchat_tasks") or [])
        chunk_n = len(payload.get("chunk_context") or [])
        logger.info(
            "parallel_execute done | business=%s | chitchat=%s | chunk=%s",
            biz_n,
            chat_n,
            chunk_n,
        )
        return {"retrieval_payload": payload}
    except Exception as e:
        logger.error("parallel_execute 失败: %s", e)
        return {
            "retrieval_payload": {
                "mode": "plan_execute",
                "business_tasks": [],
                "chitchat_tasks": [],
                "chitchat_answers": [],
                "chunk_context": [],
                "extra": {"note": f"检索异常: {e}"},
            },
            "error": str(e),
        }


def _payload_has_usable_content(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("chitchat_answers"):
        return True
    for bt in payload.get("business_tasks") or []:
        if isinstance(bt, dict) and bt.get("data"):
            return True
    if payload.get("chunk_context"):
        return True
    # 纯闲聊但 answer 已在 chitchat_tasks
    for ct in payload.get("chitchat_tasks") or []:
        if isinstance(ct, dict) and (ct.get("answer") or "").strip():
            return True
    return False


async def generate_node(state: SalaryGraphRAGState, config: RunnableConfig) -> dict:
    """综合检索/闲聊结果生成最终答案；可流式到前端。"""
    user_text = state.get("current_user_text") or _last_human_message_text(state)
    payload = state.get("retrieval_payload")
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)

    # 仅闲聊且已有子答案：仍走综合 LLM，便于多任务统一口吻；无内容则空回复
    if not _payload_has_usable_content(payload):
        # 纯闲聊且子答案失败时，直接用用户原文再答一次（可流式）
        plan = _plan_from_state(state)
        if plan and plan.chitchat_tasks and not plan.business_tasks:
            llm = get_model(model_name)
            try:
                resp = await llm.ainvoke(
                    [HumanMessage(content=user_text)],
                    config,
                )
                content = resp.content if hasattr(resp, "content") else str(resp)
                answer = content if isinstance(content, str) else str(content)
                return _make_node_result((answer or "").strip() or EMPTY_CONTEXT_REPLY)
            except Exception as e:
                logger.error("纯闲聊生成失败: %s", e)
                return _make_node_result("抱歉，暂时无法回复。", error=str(e))
        return _make_node_result(EMPTY_CONTEXT_REPLY)

    context = format_context_for_llm("plan_execute", payload)
    if not (context or "").strip():
        return _make_node_result(EMPTY_CONTEXT_REPLY)

    try:
        # 最终答案不对前端 skip：允许流式
        llm = get_model(model_name)
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


def route_after_plan(state: SalaryGraphRAGState) -> str:
    if state.get("needs_clarify"):
        return "clarify"
    return "parallel_execute"


def build_salary_graphrag_agent():
    graph = StateGraph(SalaryGraphRAGState)

    graph.add_node("prepare_long_term", prepare_long_term_entry)
    graph.add_node("reset_turn", reset_turn_node)
    graph.add_node("plan", plan_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("parallel_execute", parallel_execute_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "prepare_long_term")
    graph.add_edge("prepare_long_term", "reset_turn")
    graph.add_edge("reset_turn", "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"clarify": "clarify", "parallel_execute": "parallel_execute"},
    )
    graph.add_edge("clarify", END)
    graph.add_edge("parallel_execute", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


salary_graphrag_agent = build_salary_graphrag_agent()
