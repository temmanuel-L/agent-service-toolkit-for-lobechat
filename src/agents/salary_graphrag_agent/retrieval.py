# -*- coding: utf-8 -*-
"""并行检索编排：business 召回 / chitchat LLM / chunk hybrid。"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.salary_graphrag_agent.entity_linker import EntityLinker, LinkedEntity
from agents.salary_graphrag_agent.indexes import ensure_indexes
from agents.salary_graphrag_agent.neo4j_client import get_salary_graphrag_neo4j_database
from agents.salary_graphrag_agent.planner import PlanResult, Task
from agents.salary_graphrag_agent.prompts import CHITCHAT_SUB_PROMPT
from agents.salary_graphrag_agent.query_templates import OntologyQueryTemplates
from agents.salary_graphrag_agent.schema.reasoning_chains import is_business_chain
from core import get_model, settings
from utils.log_utils import get_logger

logger = get_logger(__name__)


def _ids(linked: list[LinkedEntity]) -> list[str]:
    return [x.element_id for x in linked if x.element_id]


def _link_debug(linked: list[LinkedEntity]) -> list[dict]:
    return [
        {
            "query": x.query_text,
            "name": x.name,
            "element_id": x.element_id,
            "method": x.method,
            "score": x.score,
        }
        for x in linked
    ]


def _log_linked(linked: list[LinkedEntity] | None, label: str = "linked") -> str:
    if not linked:
        return f"{label}=[]"
    parts = []
    for x in linked:
        if x is None:
            continue
        name = x.name or "?"
        parts.append(f"{x.query_text}→{name}({x.method})")
    return f"{label}=[{'; '.join(parts)}]"


def _dominant_area(plan: PlanResult | None) -> str | None:
    if not plan:
        return None
    for t in plan.tasks:
        if t.type == "business" and t.area:
            return t.area
    return None


def execute_business_task(
    driver,
    task: Task,
    idx: int,
    database: str | None = None,
) -> dict[str, Any]:
    """同步执行一个业务 Task：对齐 → 通用 Cypher。"""
    db = database or get_salary_graphrag_neo4j_database()
    linker = EntityLinker(driver, db)
    templates = OntologyQueryTemplates(driver, db)

    industries = linker.link_industries_by_names(task.industries)
    industry_ids = _ids(industries)
    sub_sectors = linker.link_sub_sectors_by_names(task.sub_sectors)
    sub_sector_ids = _ids(sub_sectors)

    if not industry_ids and sub_sector_ids:
        for ss_eid in sub_sector_ids:
            parent = linker.resolve_parent_industry(ss_eid)
            if parent and parent.element_id and parent.element_id not in industry_ids:
                industry_ids.append(parent.element_id)
                if parent.name and parent.name not in task.industries:
                    task.industries.append(parent.name)
                logger.info("task %s 补父行业: %s", idx, parent.name)

    jobs = linker.link_many(task.job_titles, linker.link_job)
    job_ids = _ids(jobs)

    logger.info(
        "task %s business link | chain=%s | %s | %s | %s",
        idx,
        task.chain,
        _log_linked(industries, "industries"),
        _log_linked(sub_sectors, "sub_sectors"),
        _log_linked(jobs, "jobs"),
    )

    if task.chain == "salary_chain":
        data = templates.salary_chain_recall(
            industry_ids=industry_ids or None,
            sub_sector_ids=sub_sector_ids or None,
            job_ids=job_ids or None,
            area=task.area,
        )
    elif task.chain == "industry_supplement":
        data = templates.industry_supplement_recall(
            industry_ids=industry_ids or None,
            categories=None,
        )
    else:
        data = []

    return {
        "task_index": idx,
        "chain": task.chain,
        "industries": list(task.industries),
        "sub_sectors": list(task.sub_sectors),
        "job_titles": list(task.job_titles),
        "area": task.area,
        "linked_industries": _link_debug(industries),
        "linked_sub_sectors": _link_debug(sub_sectors),
        "linked_jobs": _link_debug(jobs),
        "data": data,
    }


def execute_chunk_hybrid(
    driver,
    query_text: str,
    database: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    db = database or get_salary_graphrag_neo4j_database()
    templates = OntologyQueryTemplates(driver, db)
    try:
        rows = templates.chunk_hybrid_recall(query_text, top_k=top_k)
        logger.info("chunk hybrid recall | rows=%s", len(rows))
        return rows
    except Exception as e:
        logger.warning("chunk hybrid recall failed: %s", e)
        return []


async def _run_chitchat_task(
    task: Task,
    idx: int,
    query_text: str,
    config: RunnableConfig,
) -> dict[str, Any]:
    """闲聊 Task：在并行阶段用 skip_stream 生成子答案。"""
    raw_q = (task.raw_question or query_text or "").strip()
    result: dict[str, Any] = {
        "task_index": idx,
        "type": "chitchat",
        "raw_question": raw_q,
        "answer": "",
    }
    if not raw_q:
        return result

    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    llm = get_model(model_name).with_config(tags=["skip_stream"])
    prompt = CHITCHAT_SUB_PROMPT.format(question=raw_q)
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)], config)
        content = resp.content if hasattr(resp, "content") else str(resp)
        answer = content if isinstance(content, str) else str(content)
        result["answer"] = (answer or "").strip()
    except Exception as e:
        logger.warning("chitchat gen failed (task %s): %s", idx, e)
        result["answer"] = ""
        result["error"] = str(e)
    return result


async def parallel_retrieve(
    driver,
    query_text: str,
    plan: PlanResult,
    config: RunnableConfig,
) -> dict[str, Any]:
    """并行执行：各 business 召回 + 各 chitchat LLM + chunk hybrid。"""
    if not plan or not plan.tasks:
        return {"mode": "empty", "data": []}

    db = get_salary_graphrag_neo4j_database()
    await asyncio.to_thread(ensure_indexes, driver, db)

    coros: list[Any] = []
    tags: list[tuple[str, int]] = []  # (kind, task_index_or_-1)

    for i, task in enumerate(plan.tasks):
        if task.type == "chitchat":
            coros.append(_run_chitchat_task(task, i, query_text, config))
            tags.append(("chitchat", i))
        elif is_business_chain(task.chain):
            coros.append(
                asyncio.to_thread(execute_business_task, driver, task, i, db)
            )
            tags.append(("business", i))
        else:
            logger.info("task %s unknown chain=%s, skip", i, task.chain)

    # 有业务 task 时才跑 chunk hybrid；纯闲聊可跳过
    run_chunk = bool(plan.business_tasks)
    if run_chunk:
        coros.append(asyncio.to_thread(execute_chunk_hybrid, driver, query_text, db))
        tags.append(("chunk", -1))

    results = await asyncio.gather(*coros, return_exceptions=True)

    business_payloads: list[dict] = []
    chitchat_payloads: list[dict] = []
    chunk_rows: list[dict] = []

    for tag, res in zip(tags, results):
        kind, _idx = tag
        if isinstance(res, Exception):
            logger.error("parallel branch failed kind=%s: %s", kind, res)
            if kind == "business":
                business_payloads.append(
                    {
                        "task_index": _idx,
                        "chain": "error",
                        "data": [],
                        "error": str(res),
                    }
                )
            elif kind == "chitchat":
                chitchat_payloads.append(
                    {
                        "task_index": _idx,
                        "type": "chitchat",
                        "raw_question": query_text,
                        "answer": "",
                        "error": str(res),
                    }
                )
            continue

        if kind == "business":
            business_payloads.append(res)  # type: ignore[arg-type]
        elif kind == "chitchat":
            chitchat_payloads.append(res)  # type: ignore[arg-type]
        elif kind == "chunk":
            chunk_rows = res  # type: ignore[assignment]

    # 兼容 format_context：把闲聊答案也挂到 chitchat_answers
    chitchat_answers = [
        {"question": c.get("raw_question") or "", "answer": c.get("answer") or ""}
        for c in chitchat_payloads
        if (c.get("answer") or "").strip()
    ]

    business_payloads.sort(key=lambda x: x.get("task_index", 0))
    chitchat_payloads.sort(key=lambda x: x.get("task_index", 0))

    return {
        "mode": "plan_execute",
        "business_tasks": business_payloads,
        "chitchat_tasks": chitchat_payloads,
        "chitchat_answers": chitchat_answers,
        "chunk_context": chunk_rows,
        "requested_area": _dominant_area(plan),
    }
