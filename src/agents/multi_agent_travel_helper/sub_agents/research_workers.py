# -*- coding: utf-8 -*-
"""
旅行助手 — 联网检索子智能体（Research Workers）

每个 worker 的标准模式
--------------------
1. 根据 state 拼搜索 query；对子模型 ``bind_tools([tavily_search])`` 后
   ``ainvoke``，由模型发起 ``WebSearch`` 工具调用，再执行工具得到摘要（若未出 tool call 则回退直连 ``invoke``）。
2. 将检索摘要截断后塞进 LLM；要求模型输出「简短叙事 + JSON 数组 line_items」
   （文化 worker 仅叙事，line_items 固定为空）。
3. 返回 ``{"research": {<领域键>: block}}``，block 含 narrative / line_items / subtotal /
   currency / sources / disclaimer，供主图 ``merge_research`` 合并。

输入仅为 **文本**（``state`` 字符串字段 + 检索摘要），无图片/语音。默认模型见
``SUB_AGENT_MODEL``（``OpenAICompatibleName.OPENAI_NAME`` → MiniMax-M3；``get_model(..., fast=True)``）。

主图中的拓扑
------------
``research_fanout`` 为无操作分叉点；六条边并行触发本模块六个 ``worker_*``；
全部完成后进入 ``mobility_align_and_budget``。定价不满意时 ``rerun_selected_workers``
仅对 ``parse_pricing_rerun_targets`` 解析出的键重新 ``gather``，减少重复检索与费用。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Coroutine, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agents.multi_agent_travel_helper.multi_agent_travel_helper_agent import SUB_AGENT_MODEL
from agents.tools import tavily_search
from agents.utils import coerce_state_str, get_silent_config
from core import get_model
from utils.log_utils import get_logger

logger = get_logger(__name__)

# 展示在每条 research 子块中，提示用户勿当作实时成交价
DISCLAIMER = "以下为基于公开网页检索的估算，非实时成交价，下单前请以平台为准。"

# 参与「四类计价合计」与预算比较的 research 键；天气/文化不计入 grand_total
PRICED_KEYS = frozenset({"transport", "hotels", "food", "tickets"})

# 供 rerun_selected_workers 按名称调度协程
_WORKERS: dict[str, Callable[[dict[str, Any], RunnableConfig], Coroutine[Any, Any, dict[str, Any]]]] = {}


_WEBSEARCH_SYSTEM = (
    "你是检索调度助手。用户会给出一条搜索查询。"
    "你必须调用 WebSearch 工具执行检索，禁止编造网页或搜索结果。"
    "查询字符串请尽量直接使用用户消息中的要点，可适当补全地名或时间但不改变意图。"
)


def _normalize_tool_call(tc: Any) -> dict[str, Any]:
    """tool_calls 元素可能是 dict 或带 name/args 的对象。"""
    if isinstance(tc, dict):
        return tc
    name = getattr(tc, "name", None) or ""
    args = getattr(tc, "args", None)
    if args is not None and hasattr(args, "model_dump"):
        args = cast(Any, args).model_dump()
    return {"name": name, "args": args}


def _execute_tavily_search(query: str) -> str:
    """同步执行 Tavily 网页搜索。直接调用 ``_run``，绕过 ``invoke``/``run`` 在 content_and_artifact 下的严格元组校验。"""
    q = (query or "").strip() or "旅行"
    try:
        raw = tavily_search._run(q)
    except Exception as exc:
        logger.warning("tavily_search._run failed: %s", exc)
        return f"检索失败: {exc}"
    if isinstance(raw, tuple):
        raw = raw[0]
    return str(raw) if raw is not None else ""


def _tool_args_from_call(tc: dict[str, Any]) -> dict[str, Any]:
    """统一 LangChain tool_calls 中 args 形态（dict 或需 JSON 解析的字符串）。"""
    raw = tc.get("args")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return dict(obj) if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def _web(query: str, config: RunnableConfig) -> str:
    """先 ``get_model(...).bind_tools([tavily_search]).ainvoke``，再执行模型选择的 WebSearch；失败时回退直连工具。"""
    sub = get_silent_config(config)
    llm = get_model(config["configurable"].get("model", SUB_AGENT_MODEL), fast=True)
    bound = llm.bind_tools([tavily_search])
    try:
        ai = await bound.ainvoke(
            [
                SystemMessage(content=_WEBSEARCH_SYSTEM),
                HumanMessage(content=query.strip() or "旅行相关检索"),
            ],
            config=sub,
        )
    except Exception as exc:
        logger.warning("[research_workers] bound WebSearch ainvoke failed: %s", exc)
        ai = None

    if isinstance(ai, AIMessage) and getattr(ai, "tool_calls", None):
        for tc_raw in ai.tool_calls:
            tc = _normalize_tool_call(tc_raw)
            name = tc.get("name")
            if name != tavily_search.name:
                continue
            args = _tool_args_from_call(tc)
            q = (args.get("query") or query).strip()
            if not q:
                q = query
            try:
                return await asyncio.to_thread(_execute_tavily_search, q)
            except Exception as exc:
                logger.warning("WebSearch tool run failed: %s", exc)
                return f"检索失败: {exc}"
        logger.warning("[research_workers] tool_calls without WebSearch, fallback direct run")

    try:
        return await asyncio.to_thread(_execute_tavily_search, query.strip() or "旅行")
    except Exception as exc:
        logger.warning("WebSearch fallback run failed: %s", exc)
        return f"检索失败: {exc}"


def _safe_json_list(content: str) -> list[dict[str, Any]]:
    """从 LLM 输出中提取首个 JSON 数组：优先 Markdown ```json 围栏，否则贪婪匹配 [...]。"""
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", content, re.IGNORECASE | re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[[\s\S]*\]", content)
    if not m:
        return []
    try:
        data = json.loads(m.group())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _text_without_json_fence(content: str) -> str:
    """去掉代码围栏后剩余文本，用于拆分「叙事段落」与 JSON 数组。"""
    t = content.strip()
    t = re.sub(r"```(?:json)?\s*[\s\S]*?```", "", t, flags=re.IGNORECASE | re.DOTALL)
    return t.strip()


async def _llm_summarize(
    config: RunnableConfig,
    system: str,
    user: str,
) -> tuple[str, list[dict[str, Any]], float | None]:
    """调用主模型：返回 (叙事文本, line_items 列表, 子计)。

    subtotal 优先取单条含 line_total 的项，否则对全部 line_items 求和。
    """
    sub = get_silent_config(config)
    llm = get_model(config["configurable"].get("model", SUB_AGENT_MODEL), fast=True)
    prompt = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    resp = await llm.ainvoke(prompt, config=sub)
    text = (resp.content or "").strip()
    items = _safe_json_list(text)
    stripped = _text_without_json_fence(text)
    if "[" in stripped:
        narrative = stripped.split("[", 1)[0].strip()
    else:
        narrative = stripped or text
    subtotal = None
    for it in items:
        if isinstance(it, dict) and it.get("line_total") is not None:
            try:
                subtotal = float(it["line_total"])
            except (TypeError, ValueError):
                pass
    if subtotal is None and items:
        try:
            subtotal = sum(float(x.get("line_total", 0) or 0) for x in items if isinstance(x, dict))
        except (TypeError, ValueError):
            subtotal = None
    return narrative, items, subtotal


def _ctx(state: dict[str, Any]) -> str:
    """拼一段结构化上下文串，注入各 worker 的 user prompt；含定价反馈时附带用户调整诉求。"""
    dest = coerce_state_str(state.get("destination")) or "未知目的地"
    origin = coerce_state_str(state.get("origin_city")) or "未知出发地"
    start = state.get("trip_start_date") or "待定"
    days = state.get("trip_duration_days") or 5
    party = state.get("party_size") or 2
    sites = state.get("sites") or []
    food = state.get("food_preference") or "无特殊"
    tp = state.get("travel_price_preference") or "均衡"
    sp = state.get("site_price_preference") or "均衡"
    hp = state.get("hotel_price_preference") or "均衡"
    budget = state.get("budget_amount")
    bmode = state.get("budget_mode") or "unspecified"
    btxt = f"{budget} {state.get('budget_currency') or 'CNY'}" if budget is not None else f"未声明({bmode})"
    feedback = (state.get("pricing_feedback_text") or "").strip()
    fb = f"；用户本轮调整诉求: {feedback}" if feedback else ""
    meals = state.get("meals_per_day") or 2
    return (
        f"目的地={dest}, 出发地={origin}, 开始日={start}, 天数={days}, 人数={party}, "
        f"每日餐次数={meals}, 预算={btxt}, 景点={sites}, 美食偏好={food}, "
        f"交通价位偏好={tp}, 门票价位偏好={sp}, 酒店价位偏好={hp}{fb}"
    )


# 约束模型把 JSON 放在围栏内，便于 _safe_json_list 稳定解析
_JSON_ARRAY_HINT = "务必用 Markdown 代码块输出，例如：\n```json\n[...]\n```\n数组元素为对象。"


async def worker_weather(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """目的地 + 行程窗口天气；line_items 可为空或仅含无金额的气象要点。"""
    q = f"{coerce_state_str(state.get('destination'))} 天气预报 {coerce_state_str(state.get('trip_start_date'))} 未来{state.get('trip_duration_days') or 5}天"
    snippets = await _web(q, config)
    sys = (
        "你是旅行天气助手。根据检索摘要先写简短中文预报，再单独给出 line_items。"
        f"{_JSON_ARRAY_HINT} "
        "每项含 date, city, evidence_summary（无金额则省略 line_total）。无数据则 line_items 为 []。"
    )
    user = f"上下文: {_ctx(state)}\n\n检索摘要:\n{snippets[:6000]}"
    narrative, items, _st = await _llm_summarize(config, sys, user)
    block = {
        "narrative": narrative,
        "line_items": items,
        "subtotal": None,
        "currency": "CNY",
        "sources": snippets[:1500],
        "disclaimer": DISCLAIMER,
    }
    return {"research": {"weather": block}}


async def worker_transport(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """出发地→目的地 大交通（机票/高铁等）检索与结构化估价。"""
    d, o = coerce_state_str(state.get("destination")), coerce_state_str(state.get("origin_city"))
    q = f"{o}到{d} 机票 OR 高铁 价格 时刻表 {state.get('trip_start_date') or ''}"
    snippets = await _web(q, config)
    sys = (
        "你是交通询价助手。根据检索摘要整理方案；若摘要为空或检索失败，仍须给出**保守可比的货币估算**"
        "（例如同城短途人均数百、跨城高铁往返按人均约 1800–2800 元量级 × 人数），不得整表 line_total 全为 0。"
        f"{_JSON_ARRAY_HINT} "
        "字段: title, date, city, unit_price, quantity, line_total, pricing_basis(per_person|per_order), "
        "evidence_summary。quantity 与 line_total 须自洽。正文可短。"
    )
    user = f"上下文: {_ctx(state)}\n\n检索摘要:\n{snippets[:6000]}"
    narrative, items, subtotal = await _llm_summarize(config, sys, user)
    block = {
        "narrative": narrative,
        "line_items": items,
        "subtotal": subtotal,
        "currency": "CNY",
        "sources": snippets[:1500],
        "disclaimer": DISCLAIMER,
    }
    return {"research": {"transport": block}}


async def worker_hotel(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """目的地酒店：间夜、起止日期等写入 line_items，供 mobility 按窗口过滤。"""
    dest = coerce_state_str(state.get("destination"))
    q = f"{dest} 酒店 价格 每晚 近景区 {state.get('hotel_price_preference') or ''}"
    snippets = await _web(q, config)
    sys = (
        "你是酒店询价助手。"
        f"{_JSON_ARRAY_HINT} "
        "line_items 每项: title, start_date, end_date, city, unit_price(每间每晚), "
        "quantity(间夜数), line_total, pricing_basis(per_room_night), evidence_summary。"
        "说明：quantity=**间夜数**=房间数×住宿晚数；主方案优先**一行**汇总（勿用两家酒店各写 10 间夜却表示同一行程这种易误解拆法）；"
        "start_date/end_date 应覆盖入住至离店；若按日拆分则每条 date 填对应自然日。"
        "若检索摘要为空、失败或与目的地无关，仍须根据上下文（目的地、天数、人数、价位偏好）**自行给出至少 1 条**"
        "含合理 unit_price、quantity、line_total 的估算行，evidence_summary 写明「无可靠检索，按市场行情粗估」，禁止 line_items 为空或整表小计为 0。"
    )
    user = f"上下文: {_ctx(state)}\n\n检索摘要:\n{snippets[:6000]}"
    narrative, items, subtotal = await _llm_summarize(config, sys, user)
    block = {
        "narrative": narrative,
        "line_items": items,
        "subtotal": subtotal,
        "currency": "CNY",
        "sources": snippets[:1500],
        "disclaimer": DISCLAIMER,
    }
    return {"research": {"hotels": block}}


async def worker_food(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """餐饮人均与人数乘积形成 line_total，纳入预算四类合计。"""
    dest = coerce_state_str(state.get("destination"))
    q = f"{dest} 餐厅 推荐 人均消费 {state.get('food_preference') or ''}"
    snippets = await _web(q, config)
    sys = (
        "你是美食助手。餐饮预算须覆盖**整段行程**（上下文中的天数×每日餐次数×人数），"
        "不要只写一顿饭金额当全程总价；可输出**一行**汇总：line_total ≈ 人均每餐参考价 × 餐次 × 天数 × 人数。"
        f"{_JSON_ARRAY_HINT} "
        "line_items: title, date, city, unit_price(人均每餐参考), quantity(总餐顿数=餐次×天数×人数或分项说明), "
        "line_total, pricing_basis(per_person), evidence_summary。按日分列时各 date 填当日。"
    )
    user = f"上下文: {_ctx(state)}\n\n检索摘要:\n{snippets[:6000]}"
    narrative, items, subtotal = await _llm_summarize(config, sys, user)
    block = {
        "narrative": narrative,
        "line_items": items,
        "subtotal": subtotal,
        "currency": "CNY",
        "sources": snippets[:1500],
        "disclaimer": DISCLAIMER,
    }
    return {"research": {"food": block}}


async def worker_culture(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """文化背景叙事，不参与计价；compose 阶段截取 narrative 注入行程文案。"""
    dest = coerce_state_str(state.get("destination"))
    sites = state.get("sites") or []
    q = f"{dest} 文化 历史 博物馆 节庆 {sites}"
    snippets = await _web(q, config)
    sys = "你是文化导游。根据摘要写一段中文文化介绍（景点、节庆、博物馆等）；不必输出 JSON 或 line_items。"
    user = f"上下文: {_ctx(state)}\n\n检索摘要:\n{snippets[:6000]}"
    sub = get_silent_config(config)
    llm = get_model(config["configurable"].get("model", SUB_AGENT_MODEL), fast=True)
    resp = await llm.ainvoke(
        [SystemMessage(content=sys), HumanMessage(content=user)],
        config=sub,
    )
    narrative = (resp.content or "").strip()
    block = {
        "narrative": narrative,
        "line_items": [],
        "subtotal": None,
        "currency": "CNY",
        "sources": snippets[:1500],
        "disclaimer": DISCLAIMER,
    }
    return {"research": {"culture": block}}


async def worker_ticket(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """景点门票：结合 sites 列表检索成人票等公开报价并结构化。"""
    dest = coerce_state_str(state.get("destination"))
    sites = state.get("sites") or []
    q = f"{dest} {' '.join(sites)} 门票 价格 成人票"
    snippets = await _web(q, config)
    sys = (
        "你是门票助手。"
        f"{_JSON_ARRAY_HINT} "
        "line_items: title(景点), date, city, unit_price(单人), quantity(人数*张数), "
        "line_total, pricing_basis(per_person), evidence_summary。"
        "若检索摘要为空或失败，仍须结合上下文 sites 与人数给出**至少 1 条**合理门票估算（可合并为多景点一行），"
        "evidence_summary 注明粗估依据；禁止 line_items 为空或合计为 0。"
    )
    user = f"上下文: {_ctx(state)}\n\n检索摘要:\n{snippets[:6000]}"
    narrative, items, subtotal = await _llm_summarize(config, sys, user)
    block = {
        "narrative": narrative,
        "line_items": items,
        "subtotal": subtotal,
        "currency": "CNY",
        "sources": snippets[:1500],
        "disclaimer": DISCLAIMER,
    }
    return {"research": {"tickets": block}}


_WORKERS.update(
    {
        "weather": worker_weather,
        "transport": worker_transport,
        "hotels": worker_hotel,
        "food": worker_food,
        "culture": worker_culture,
        "tickets": worker_ticket,
    }
)


def parse_pricing_rerun_targets(user_text: str) -> list[str]:
    """从用户自然语言中解析要重跑的 research 键；未命中时重跑四类计价。"""
    t = user_text.lower()
    out: list[str] = []
    if any(k in t for k in ("机票", "火车", "高铁", "航班", "交通", "直飞", "动车")):
        out.append("transport")
    if "酒店" in t or "住宿" in t or "宾馆" in t:
        out.append("hotels")
    if "餐" in t or "饭店" in t or "美食" in t or "吃" in t:
        out.append("food")
    if "门票" in t or "景点" in t or "票" in t:
        out.append("tickets")
    if "天气" in t:
        out.append("weather")
    if "文化" in t:
        out.append("culture")
    seen: set[str] = set()
    ordered: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered if ordered else sorted(PRICED_KEYS)


def research_fanout(state: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
    """LangGraph 并行分叉锚点：不修改 state，仅使多条出边共享同一前置节点。

    形参须命名为 ``state`` / ``config``：框架只向名为 ``config`` 的参数注入 ``RunnableConfig``，
    使用 ``_config`` 时 Studio 等路径下可能只传入 state，触发缺参 TypeError。
    """
    return {}


async def rerun_selected_workers(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """仅重跑指定键，合并回现有 research。

    若 targets 非法或为空则回退为四类计价全跑；结束后清空 pricing_feedback_text /
    pricing_rerun_targets，避免死循环携带旧反馈。
    """
    targets = list(state.get("pricing_rerun_targets") or [])
    if not targets:
        targets = sorted(PRICED_KEYS)
    targets = [k for k in targets if k in _WORKERS]
    if not targets:
        targets = sorted(PRICED_KEYS)
    merged = dict(state.get("research") or {})
    st = {**state, "research": merged}
    tasks = [_WORKERS[k](st, config) for k in targets]
    results = await asyncio.gather(*tasks)
    for part in results:
        for rk, rv in (part.get("research") or {}).items():
            merged[rk] = rv
    logger.info("[rerun_workers] keys=%s", targets)
    return {
        "research": merged,
        "pricing_feedback_text": None,
        "pricing_rerun_targets": None,
    }


async def run_all_workers_parallel(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """单节点内 asyncio.gather 六路；供测试脚本或简化图使用。主生产图用 fan-out + 屏障模式。"""
    results = await asyncio.gather(
        worker_weather(state, config),
        worker_transport(state, config),
        worker_hotel(state, config),
        worker_food(state, config),
        worker_culture(state, config),
        worker_ticket(state, config),
    )
    merged: dict[str, Any] = {}
    for part in results:
        merged.update(part.get("research") or {})
    return {"research": merged}
