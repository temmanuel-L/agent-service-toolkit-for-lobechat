# -*- coding: utf-8 -*-
"""Plan + 实体抽取合一：一次 LLM 产出 List[Task]。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.salary_graphrag_agent.schema.aliases import (
    AREA_CANONICAL_NAMES,
    extract_region_from_text,
    get_cached_industry_subsector_tree,
    normalize_salary_level,
)
from agents.salary_graphrag_agent.schema.reasoning_chains import is_business_chain
from utils.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """planner 产出的一个任务单元。"""

    type: str = "chitchat"  # business | chitchat
    chain: str = "none"  # salary_chain | industry_supplement | none
    industries: list[str] = field(default_factory=list)
    sub_sectors: list[str] = field(default_factory=list)
    job_titles: list[str] = field(default_factory=list)
    area: str | None = None
    raw_question: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanResult:
    """planner 产出。"""

    tasks: list[Task] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tasks": [t.to_dict() for t in self.tasks], "raw": self.raw}

    @property
    def business_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.type == "business"]

    @property
    def chitchat_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.type == "chitchat"]


def _format_tree(tree: dict[str, list[str]]) -> str:
    if not tree:
        return "（图里暂无一级行业）"
    lines = []
    for ind, subs in tree.items():
        if subs:
            lines.append(f"- {ind}（二级行业: {'、'.join(subs)}）")
        else:
            lines.append(f"- {ind}")
    return "\n".join(lines)


def build_plan_prompt(query: str, tree: dict[str, list[str]]) -> str:
    tree_text = _format_tree(tree)
    area_list = "、".join(AREA_CANONICAL_NAMES)
    chain_desc = "\n".join(
        f"- {c}: {desc}"
        for c, desc in [
            (
                "salary_chain",
                "薪资推理链：Industry → SubSector → JobPosition → Area。回答薪资查询/对比/概览。",
            ),
            (
                "industry_supplement",
                "行业补充信息链：Industry 属性召回（skills/hot_positions/high_paying_positions/trends/overview）"
                "+ JobPosition[category]。回答热门职位/高薪职位/关键技能/发展趋势。",
            ),
        ]
    )

    return f"""你是薪酬知识图谱的任务规划器与实体抽取器。
分析用户问题，输出严格 JSON（不要 markdown，不要解释），字段如下：
{{
  "tasks": [
    {{
      "type": "business|chitchat",
      "chain": "salary_chain|industry_supplement|none",
      "industries": ["一级行业规范名"],
      "sub_sectors": ["二级行业规范名"],
      "job_titles": ["用户问题中的岗位原话"],
      "area": "全国平均|华东|华北|华南|null",
      "raw_question": ""
    }}
  ]
}}

图里的一级行业与二级行业树（industry → sub_sectors）：
{tree_text}

Area 规范名（4 个枚举）：{area_list}

业务意图（chain）定义：
{chain_desc}

规则：
1. 把用户问题拆成一件或多件事（tasks）。例如"银行业财务负责人薪资 + 银行业热门职位"应拆成 2 个 business task。
2. 与薪酬报告业务无关的闲聊（如"你觉得这个报告准吗""你是谁"、写观后感等）type 用 chitchat，chain 用 none，raw_question 填该件事的用户原话片段。
3. business task 的 chain 二选一：
   - 涉及薪资数字/岗位薪资/行业对比 → salary_chain
   - 涉及热门职位/高薪职位/关键技能/新兴技能/发展趋势/行业概览 → industry_supplement
   - 同时涉及薪资和补充信息 → 拆成两个 business task
4. industries / sub_sectors 必须从图里的规范名树中选取，不要自造。
5. 如果用户说的是二级行业（如"保险""一级市场"），填到 sub_sectors，同时填对应的一级行业到 industries
   （看树：保险 挂在 银行与金融服务 下 → industries=["银行与金融服务"], sub_sectors=["保险"]）。
6. job_titles 填用户原话，不做映射，后续由实体链接器对齐到图节点。
   注意：只有薪资相关的岗位才填 job_titles；"热门职位""高薪职位"章节列出的岗位名不要填，
   它们会通过 industry_supplement 链从 Industry 属性召回。
7. area 仅填全国平均|华东|华北|华南|null。用户未提区域时填 null；说"全国"填"全国平均"。
8. 一个 task 可以同时有多个 industries（跨行业对比）。
9. 「对比」本身不单独拆成 task：为各方各建 salary_chain task，由最终回答综合对比。

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


def _clean_list(lst) -> list[str]:
    return [x for x in (lst or []) if x and x != "null"]


def _normalize_area(value: Any, query_text: str = "") -> str | None:
    if value is None or value == "" or value == "null":
        return extract_region_from_text(query_text)
    norm = normalize_salary_level(str(value))
    if norm in AREA_CANONICAL_NAMES:
        return norm
    return extract_region_from_text(query_text) or (norm or None)


def _parse_tasks(data: dict, query_text: str) -> list[Task]:
    raw_tasks = data.get("tasks") or []
    if not isinstance(raw_tasks, list):
        return []
    tasks: list[Task] = []
    for rt in raw_tasks:
        if not isinstance(rt, dict):
            continue
        ttype = (rt.get("type") or "chitchat").strip()
        chain = (rt.get("chain") or "none").strip()
        if ttype not in ("business", "chitchat"):
            ttype = "chitchat"
        if ttype == "chitchat":
            chain = "none"
        elif not is_business_chain(chain):
            chain = "none"
        area = _normalize_area(rt.get("area"), query_text)
        tasks.append(
            Task(
                type=ttype,
                chain=chain,
                industries=_clean_list(rt.get("industries")),
                sub_sectors=_clean_list(rt.get("sub_sectors")),
                job_titles=_clean_list(rt.get("job_titles")),
                area=area,
                raw_question=(rt.get("raw_question") or "").strip(),
            )
        )
    return tasks


def _heuristic_plan(query_text: str, tree: dict[str, list[str]]) -> PlanResult:
    """LLM 不可用时的规则兜底：从规范名树做子串匹配。"""
    q = query_text or ""
    sub_to_ind: dict[str, str] = {}
    for ind, subs in tree.items():
        for s in subs:
            sub_to_ind[s] = ind

    industries: list[str] = []
    sub_sectors: list[str] = []
    for ind in tree:
        if ind in q and ind not in industries:
            industries.append(ind)
    for sub, ind in sub_to_ind.items():
        if sub in q:
            if sub not in sub_sectors:
                sub_sectors.append(sub)
            if ind not in industries:
                industries.append(ind)

    area = extract_region_from_text(q)
    job_titles: list[str] = []

    has_salary = any(w in q for w in ("薪资", "薪酬", "工资", "多少钱"))
    has_supplement = any(
        w in q
        for w in (
            "热门职位",
            "高薪职位",
            "关键技能",
            "新兴技能",
            "趋势",
            "发展",
            "技能",
            "概览",
            "有哪些岗位",
        )
    )
    is_compare = any(w in q for w in ("对比", "分别", "vs", "VS"))

    tasks: list[Task] = []
    if has_salary or is_compare:
        tasks.append(
            Task(
                type="business",
                chain="salary_chain",
                industries=industries,
                sub_sectors=sub_sectors,
                job_titles=job_titles,
                area=area,
            )
        )
    if has_supplement:
        tasks.append(
            Task(
                type="business",
                chain="industry_supplement",
                industries=industries,
                sub_sectors=sub_sectors,
                job_titles=job_titles,
                area=area,
            )
        )
    if not tasks:
        if industries or sub_sectors:
            tasks.append(
                Task(
                    type="business",
                    chain="industry_supplement",
                    industries=industries,
                    sub_sectors=sub_sectors,
                    job_titles=job_titles,
                    area=area,
                )
            )
        else:
            tasks.append(Task(type="chitchat", chain="none", raw_question=q))

    return PlanResult(tasks=tasks, raw={"source": "heuristic"})


def load_industry_tree(driver=None, database: str | None = None) -> dict[str, list[str]]:
    if not driver:
        return {}
    try:
        return get_cached_industry_subsector_tree(driver, database)
    except Exception as e:
        logger.warning("加载 Industry→SubSector 树失败: %s", e)
        return {}


def plan_from_llm_content(
    query_text: str,
    content: str,
    tree: dict[str, list[str]] | None = None,
) -> PlanResult:
    """解析 LLM 返回的 JSON 为 PlanResult；失败则启发式兜底。"""
    tree = tree or {}
    try:
        data = _extract_json(content)
        tasks = _parse_tasks(data, query_text)
        if not tasks:
            return _heuristic_plan(query_text, tree)
        return PlanResult(tasks=tasks, raw=data)
    except Exception as e:
        logger.warning("解析 plan JSON 失败，使用规则兜底: %s", e)
        return _heuristic_plan(query_text, tree)


def plan_query(
    query_text: str,
    driver=None,
    database: str | None = None,
    llm=None,
) -> PlanResult:
    """plan + 实体抽取合一入口。

    llm: 已配置好的 chat model（建议带 skip_stream）。若为 None 则直接启发式。
    """
    tree = load_industry_tree(driver, database)

    if llm is None:
        return _heuristic_plan(query_text, tree)

    try:
        prompt = build_plan_prompt(query_text, tree)
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(content, list):
            content = "".join(
                str(c.get("text", c) if isinstance(c, dict) else c) for c in content
            )
        return plan_from_llm_content(query_text, str(content), tree)
    except Exception as e:
        logger.warning("LLM 规划失败，使用规则兜底: %s", e)
        return _heuristic_plan(query_text, tree)


def check_plan_slots(plan: PlanResult | None) -> tuple[bool, str | None]:
    """检查所有业务 task 的必填层（Industry）是否充足。"""
    from agents.salary_graphrag_agent.schema.aliases import INDUSTRY_CANONICAL_NAMES
    from agents.salary_graphrag_agent.schema.reasoning_chains import required_layers

    if not plan:
        return True, None

    industry_hint = "、".join(INDUSTRY_CANONICAL_NAMES[:6]) + " 等"
    missing_tasks = []
    for i, t in enumerate(plan.tasks):
        if t.type != "business":
            continue
        if not is_business_chain(t.chain):
            continue
        required = required_layers(t.chain)
        if "Industry" in required and not t.industries and not t.sub_sectors:
            missing_tasks.append(i + 1)

    if missing_tasks:
        return False, (
            f"第 {missing_tasks} 个业务问题缺少一级行业锚点，"
            f"请补充您想查询的一级行业（如{industry_hint}）或二级行业（如保险/一级市场）。"
        )
    return True, None
