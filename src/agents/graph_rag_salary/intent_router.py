# -*- coding: utf-8 -*-
"""二级意图分类 + 实体抽取（薪酬 GraphRAG）。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from agents.graph_rag_salary.schema.aliases import (
    extract_region_from_text,
    normalize_salary_level,
)
from core import get_model, settings
from utils.log_utils import get_logger

logger = get_logger(__name__)

VALID_INTENTS = {
    "salary_lookup",
    "compare",
    "industry_overview",
    "skill_trend",
    "fallback",
}


@dataclass
class IntentResult:
    intent: str = "fallback"
    industries: list[str] = field(default_factory=list)
    job_titles: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    compare_groups: list[dict[str, Any]] = field(default_factory=list)
    region: str | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_slots(self) -> dict[str, Any]:
        """写入 State.intent_slots 的精简可序列化槽位。"""
        return {
            "industries": list(self.industries or []),
            "job_titles": list(self.job_titles or []),
            "skills": list(self.skills or []),
            "compare_groups": list(self.compare_groups or []),
            "region": self.region,
        }


INTENT_PROMPT = """你是薪酬知识图谱的意图分类与实体抽取器。
根据用户问题，输出严格 JSON（不要 markdown，不要解释），字段如下：
{{
  "intent": "salary_lookup|compare|industry_overview|skill_trend|fallback",
  "industries": ["行业名1"],
  "job_titles": ["岗位名1"],
  "skills": ["技能名1"],
  "region": "全国平均|华东|华北|华南|null",
  "compare_groups": [
    {{"industry": "行业A", "job_title": "岗位A"}},
    {{"industry": "行业B", "job_title": "岗位B"}}
  ]
}}

意图定义：
- salary_lookup: 查询某行业某岗位（或明确岗位）的薪资
- compare: 对比两个及以上行业/岗位的薪资或趋势
- industry_overview: 问某行业有哪些岗位、整体情况，未指定具体岗位薪资
- skill_trend: 主要问技能需求或市场趋势
- fallback: 无法归类或信息过少

规则：
1. 尽量从问题中抽出 industries / job_titles / skills 原文短语
2. compare 时必须填 compare_groups（至少两组）；其他意图 compare_groups 可为 []
3. 若只提行业未提岗位，intent 用 industry_overview
4. 若同时问薪资与趋势且是对比，intent 用 compare
5. region 仅填薪酬地理口径：全国平均/华东/华北/华南；用户未提地区时填 null
6. 用户说「全国」时 region 填「全国平均」；多地区对比时填首个提到的地区或 null

用户问题：
{query}
"""


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_region(value: Any, query_text: str = "") -> str | None:
    if value is None or value == "" or value == "null":
        return extract_region_from_text(query_text)
    level = normalize_salary_level(str(value))
    if level in ("全国平均", "华东", "华北", "华南"):
        return level
    return extract_region_from_text(query_text) or (level or None)


def _heuristic_intent(query_text: str) -> IntentResult:
    """LLM 不可用时的规则兜底。"""
    q = query_text or ""
    industries: list[str] = []
    jobs: list[str] = []
    skills: list[str] = []
    compare_groups: list[dict] = []
    region = extract_region_from_text(q)

    industry_hints = [
        "保险", "一级市场", "消费品营销", "消费品", "人工智能", "半导体",
        "银行", "科技", "财务与会计", "市场营销", "营销", "半导体材料",
    ]
    job_hints = [
        "营销负责人", "财务负责人", "财务责任人", "AI负责人", "研发负责人",
        "人力资源负责人", "销售负责人", "产品总监", "首席运营官",
    ]
    for h in industry_hints:
        if h in q and h not in industries:
            industries.append(h)
    for h in job_hints:
        if h in q and h not in jobs:
            jobs.append(h)

    if any(w in q for w in ("对比", "分别", "vs", "VS", "与")) and (
        len(industries) >= 2 or len(jobs) >= 2
    ):
        intent = "compare"
        if len(industries) >= 2:
            compare_groups = [
                {"industry": industries[0], "job_title": jobs[0] if jobs else ""},
                {
                    "industry": industries[1],
                    "job_title": jobs[1] if len(jobs) > 1 else (jobs[0] if jobs else ""),
                },
            ]
    elif any(w in q for w in ("技能", "趋势", "发展", "数字化")) and "薪资" not in q:
        intent = "skill_trend"
    elif any(w in q for w in ("哪些岗位", "有哪些", "岗位列表", "概览")) or (
        industries and not jobs
    ):
        intent = "industry_overview"
    elif "薪资" in q or "薪酬" in q or "工资" in q:
        intent = "salary_lookup"
    else:
        intent = "fallback"

    return IntentResult(
        intent=intent,
        industries=industries,
        job_titles=jobs,
        skills=skills,
        compare_groups=compare_groups,
        region=region,
        raw={"source": "heuristic"},
    )


def message_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def parse_intent_from_llm_content(content: str, query_text: str) -> IntentResult:
    """解析二级意图 LLM 输出，并做启发式纠偏。"""
    data = _extract_json(str(content))
    intent = data.get("intent", "fallback")
    if intent not in VALID_INTENTS:
        intent = "fallback"

    result = IntentResult(
        intent=intent,
        industries=[x for x in (data.get("industries") or []) if x],
        job_titles=[x for x in (data.get("job_titles") or []) if x],
        skills=[x for x in (data.get("skills") or []) if x],
        compare_groups=data.get("compare_groups") or [],
        region=_normalize_region(data.get("region"), query_text),
        raw=data,
    )
    return _postprocess_intent(result, query_text)


def _postprocess_intent(result: IntentResult, query_text: str) -> IntentResult:
    if result.intent != "compare":
        if any(w in query_text for w in ("对比", "分别", "vs", "VS")) and len(
            result.compare_groups
        ) >= 2:
            result.intent = "compare"

    if result.intent == "salary_lookup" and result.industries and not result.job_titles:
        if any(w in query_text for w in ("哪些岗位", "有哪些", "岗位列表", "概览")):
            result.intent = "industry_overview"

    if not result.region:
        result.region = extract_region_from_text(query_text)

    return result


def heuristic_intent_with_postprocess(query_text: str) -> IntentResult:
    """规则兜底 + 纠偏，供节点在 LLM 失败时使用。"""
    return _postprocess_intent(_heuristic_intent(query_text), query_text)


def classify_intent(query_text: str, model_name: Any | None = None) -> IntentResult:
    """同步意图分类：优先 LLM（带 skip_stream），失败则规则兜底。

    注意：在 LangGraph 流式服务中，优先由节点侧 ainvoke(tags=skip_stream) 调用，
    再走 parse_intent_from_llm_content；本函数供脚本/测试使用。
    """
    try:
        llm = get_model(model_name or settings.DEFAULT_MODEL).with_config(
            tags=["skip_stream"]
        )
        prompt = INTENT_PROMPT.format(query=query_text)
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = message_content_to_str(
            resp.content if hasattr(resp, "content") else resp
        )
        return parse_intent_from_llm_content(content, query_text)
    except Exception as e:
        logger.warning("LLM 分类失败，使用规则兜底: %s", e)
        return heuristic_intent_with_postprocess(query_text)


def intent_result_from_slots(
    salary_intent: str | None,
    intent_slots: dict[str, Any] | None,
) -> IntentResult:
    """从 State 字段还原 IntentResult，供检索使用。"""
    slots = intent_slots or {}
    intent = salary_intent if salary_intent in VALID_INTENTS else "fallback"
    return IntentResult(
        intent=intent,
        industries=list(slots.get("industries") or []),
        job_titles=list(slots.get("job_titles") or []),
        skills=list(slots.get("skills") or []),
        compare_groups=list(slots.get("compare_groups") or []),
        region=slots.get("region"),
        raw={"source": "state"},
    )
