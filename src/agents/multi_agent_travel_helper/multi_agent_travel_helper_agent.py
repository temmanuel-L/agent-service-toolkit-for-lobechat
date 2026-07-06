# -*- coding: utf-8 -*-
"""
multi_agent_travel_helper — 旅行多智能体编排入口（LangGraph StateGraph）

功能概览
--------
1. **偏好水合**：从 SQLite 读取用户历史「交通/门票/酒店」价位偏好，填入 state（仅补缺）。
2. **多模态 Intake**：规范化最新 HumanMessage → 可选 Vision 描述 → 工具调用抽取结构化行程字段。
3. **双阶段 HITL**：
   - 补全轮：缺关键字段时 interrupt，用户补充后回到 normalize 重跑；
   - 确认轮：Markdown 摘要 +「确认」后才进入检索。
4. **六路并行检索**：`research_fanout` 仅作分叉点，六条边并行执行 sub_agents 内各 worker，结果经
   `merge_research` 浅合并入 `state["research"]`。各 worker 仅消费**文本**上下文与检索摘要（无图/音）；
   默认 LLM 为模块常量 ``SUB_AGENT_MODEL``（可与主对话 ``settings.DEFAULT_MODEL`` 分离，便于高并发纯文本 API）。
5. **对齐与预算**：六路 worker fan-in 后，归桶去重、日期窗口过滤、四类合计与预算对照（表格由 ``build_pricing_md`` 确定性渲染，不另调 LLM）。
6. **定价 HITL**：`pricing_user_confirm` 将 ``build_pricing_md`` 全文经 ``interrupt`` 交用户确认；耗时的主要来源是各路 worker 的检索与抽取，而非 fan-in 拼表。
7. **合成**：通过后生成 Markdown 行程单，并在同一节点内按 ``TravelHelperState`` 清空业务槽位写回 checkpoint
   （与 ``simple_travel_planner_agent.create_itinerary`` 在返回里重置 destination/interests 同理）。

图拓扑与边关系见文件末尾 `workflow.add_*`；更完整设计说明见同目录 ``plan.md``。
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Annotated, Any, List, Literal, Optional, Union, get_args, get_origin, get_type_hints

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.types import interrupt
from pydantic import BaseModel, Field, field_validator

from core import get_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry
from schema.models import AllModelEnum, OpenAICompatibleName

# 六路 research worker 默认 SUB_AGENT_MODEL（OpenAICompatible → MiniMax-M3，llm.py 默认关闭 thinking）。
# SUB_AGENT_MODEL: AllModelEnum = settings.DEFAULT_MODEL
SUB_AGENT_MODEL: AllModelEnum = OpenAICompatibleName.OPENAI_NAME

from agents.multimodal_input_processor import MultimodalInputProcessor
from agents.multi_agent_travel_helper.pref_db import get_price_preferences, upsert_price_preferences
from agents.multi_agent_travel_helper.sub_agents import (
    parse_pricing_rerun_targets,
    research_fanout,
    rerun_selected_workers,
    worker_culture,
    worker_food,
    worker_hotel,
    worker_ticket,
    worker_transport,
    worker_weather,
)
from agents.utils import (
    build_interrupt_text_message_update,
    coerce_optional_str,
    coerce_state_str,
    get_silent_config,
    normalize_date_optional_str,
)
from utils.log_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 流程控制常量
# ---------------------------------------------------------------------------
# 补全阶段：最多 interrupt 追问轮数（超过则不再 ask，靠默认值推进）
INTAKE_COLLECT_MAX_ROUNDS = 2
# 预留：当前 intake_confirm 未用该上限做硬截断，便于后续扩展
INTAKE_CONFIRM_MAX_ROUNDS = 3
# 定价不确认时，最多允许几轮反馈后再强制视为通过（避免无限循环）
PRICING_CONFIRM_MAX_ROUNDS = 2

# 用户未说明天数/人数时的默认
DEFAULT_TRIP_DAYS = 5
DEFAULT_PARTY = 2
# 三类价位偏好均未抽取到时写入的中性描述，供 worker 检索提示使用
NEUTRAL_PRICE_PREF = "性价比均衡，可接受高铁二等座/经济舱、三星级酒店、常规门票"
# interrupt 回调里若只有图片无文本，合并进 messages 的占位说明
INTERRUPT_IMAGE_ONLY_FALLBACK_TEXT = "[本轮仅收到图片，暂未识别出可用文本]"


def _cfg_uid(config: RunnableConfig | None) -> str:
    """从 RunnableConfig 取 user_id，用于 SQLite 偏好读写；缺失时统一为 default_user。"""
    if not config:
        return "default_user"
    return (config.get("configurable") or {}).get("user_id") or "default_user"


def merge_research(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """LangGraph reducer：并行 worker 各返回 ``{"research": {k: block}}`` 时逐次浅合并。

    - 同一顶层键（如 transport）后者覆盖前者整块。
    - ``right is None``：用于 ``compose_itinerary`` 结束清空 research，整字段置 None。
    """
    if right is None:
        return None
    out = dict(left or {})
    out.update(right)
    return out


class TravelHelperState(MessagesState, total=False):
    """LangGraph 状态：在 MessagesState（messages 由 reducer 追加）上扩展旅行专用字段。

    注意：`research` 使用自定义 reducer ``merge_research``，多 worker 并行返回的 dict 会按键浅合并；
    合成行程节点可将 ``research`` 置为 None 以清空大块检索结果。
    """

    remaining_steps: RemainingSteps
    # --- 行程核心 ---
    destination: Optional[str]
    origin_city: Optional[str]
    sites: Optional[List[str]]
    food_preference: Optional[str]
    travel_price_preference: Optional[str]
    site_price_preference: Optional[str]
    hotel_price_preference: Optional[str]
    trip_start_date: Optional[str]
    trip_duration_days: Optional[int]
    budget_amount: Optional[float]
    budget_currency: Optional[str]
    budget_mode: Optional[str]
    party_size: Optional[int]
    occupancy_per_room: Optional[int]
    meals_per_day: Optional[int]
    hotel_preferences: Optional[str]
    dietary_restrictions: Optional[str]
    # --- 多模态规范化（normalize_input / vision_enrich）---
    current_user_text: Optional[str]
    has_visual_input: Optional[bool]
    extraction_input_text: Optional[str]
    # --- Intake 轮次与确认 ---
    intake_collect_round: int
    intake_confirm_round: int
    intake_confirmed: bool
    intake_defaults_applied: dict[str, Any]
    intake_summary_md: Optional[str]
    # weather/transport/hotels/food/culture/tickets 等子块，结构见 sub_agents.research_workers
    research: Annotated[Optional[dict[str, Any]], merge_research]
    # --- 定价 HITL 与重跑 ---
    pricing_feedback_text: Optional[str]
    pricing_rerun_targets: Optional[List[str]]
    mobility_timeline: Optional[str]
    alignment_report: Optional[str]
    budget_breakdown: Optional[dict[str, Any]]
    budget_vs_user: Optional[str]
    pricing_confirmation_md: Optional[str]
    pricing_confirm_round: int
    pricing_user_ok: Optional[bool]
    itinerary: Optional[str]


# compose 写回 checkpoint 时跳过：消息由节点返回追加；remaining_steps 由 LangGraph 管理
_TRAVEL_HELPER_RESET_SKIP = frozenset({"messages", "remaining_steps"})


def _is_dict_type(tp: Any) -> bool:
    return tp is dict or get_origin(tp) is dict


def _is_list_type(tp: Any) -> bool:
    return tp is list or get_origin(tp) is list


def _neutral_value_for_travel_state(key: str, ann: Any) -> Any:
    """按字段注解给出「清空」用的占位值；``research`` 必须整表置 ``None`` 以触发 ``merge_research`` 清空。"""
    if key == "research":
        return None
    origin = get_origin(ann)
    args = get_args(ann)
    if origin is Annotated:
        return _neutral_value_for_travel_state(key, args[0])
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if type(None) in args and len(non_none) == 1:
            inner = non_none[0]
            # Optional 标量：清空为 None（勿用 False/0，否则与「未声明」语义混淆）
            if inner is bool or inner is int or inner is float or inner is str:
                return None
            if _is_dict_type(inner):
                return {}
            if _is_list_type(inner):
                return None
            return _neutral_value_for_travel_state(key, inner)
        return None
    if _is_dict_type(ann):
        return {}
    if _is_list_type(ann):
        return None
    if ann is int:
        return 0
    if ann is float:
        return 0.0
    if ann is bool:
        return False
    return None


def travel_helper_checkpoint_tail_reset() -> dict[str, Any]:
    """遍历 ``TravelHelperState`` + ``MessagesState`` 的注解，生成节点返回用的「字段→清空值」映射。

    供 ``compose_itinerary`` 在生成行程后与 ``simple_travel_planner_agent.create_itinerary`` 一样，
    在同一轮更新里写回 checkpoint，使下一笔用户消息从 START 进入时业务槽位已空。
    """
    th = get_type_hints(TravelHelperState, include_extras=True)
    ms = get_type_hints(MessagesState, include_extras=True)
    merged = {**ms, **th}
    out: dict[str, Any] = {}
    for key, hint in merged.items():
        if key in _TRAVEL_HELPER_RESET_SKIP:
            continue
        out[key] = _neutral_value_for_travel_state(key, hint)
    return out


class TravelHelperExtraction(BaseModel):
    """供 LLM bind_tools 使用的结构化抽取 schema；仅填充用户明确提到的字段，禁止臆造。"""

    destination: Optional[str] = Field(None, description="目的地城市或地区")
    origin_city: Optional[str] = Field(None, description="出发城市")
    sites: Optional[List[str]] = Field(None, description="想去的景点或地标")
    food_preference: Optional[str] = Field(None, description="美食偏好简述")
    travel_price_preference: Optional[str] = Field(None, description="交通价位/舒适度偏好")
    site_price_preference: Optional[str] = Field(None, description="门票价位偏好")
    hotel_price_preference: Optional[str] = Field(None, description="酒店档次/价位偏好")
    trip_start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    trip_duration_days: Optional[int] = Field(None, description="行程天数")
    budget_amount: Optional[float] = Field(None, description="旅行总预算金额")
    budget_currency: Optional[str] = Field(None, description="币种，默认CNY")
    party_size: Optional[int] = Field(None, description="出行人数")

    @field_validator(
        "destination",
        "origin_city",
        "food_preference",
        "travel_price_preference",
        "site_price_preference",
        "hotel_price_preference",
        "budget_currency",
        mode="before",
    )
    @classmethod
    def _coerce_str_fields(cls, v: Any) -> Any:
        return coerce_optional_str(v)

    @field_validator("trip_start_date", mode="before")
    @classmethod
    def _coerce_trip_start_date(cls, v: Any) -> Any:
        return normalize_date_optional_str(v)

    @field_validator("sites", mode="before")
    @classmethod
    def _coerce_sites(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return v


# 工具调用抽取：强调不臆造、景点粒度、必须通过 TravelHelperExtraction 产出
EXTRACTION_SYSTEM_PROMPT = """你是旅行信息抽取助手。从用户消息中提取字段；只提取明确信息，不猜测。
规则：具体景点放入 sites，不要把整座城市误标为单个景点名称。
若正文含「[图片内容识别]」段落（来自多模态识图），必须结合该段与用户配文抽取：段落中的城市/景区/地标须反映到
destination 与 sites（配文已写明城市时以配文为准）；不得整段忽略。仅当识图段完全无法对应任何地点时，可不填 destination。
必须调用 TravelHelperExtraction 工具返回结果。"""

# Vision 只负责自然语言描述，结构化留给 extract_info，避免模型在图里输出混杂 JSON
VISION_SYSTEM_PROMPT = """你是旅行场景图像理解助手。根据图片及附带文字，用简洁中文输出，便于后续抽取：
写出可判断的国家/省份/城市；逐条列出画面中可辨认的著名景点或地标（多条用换行或顿号分隔）。
若无法确定城市，写明「无法辨认具体城市」并仍描述可见的建筑或景区特征。不要 JSON。"""

# interrupt 追问时展示的字段中文名（与 soft_missing_fields 键一致）
FIELD_LABELS = {
    "destination": "目的地",
    "origin_city": "出发地",
    "trip_start_date": "行程开始日期（YYYY-MM-DD）",
    "sites": "想去的景点或活动（至少一项）",
}

def _last_human_message(messages: List[Any]) -> HumanMessage | None:
    """从消息列表末尾向前查找最近一条 HumanMessage（当前轮用户输入）。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def _normalize_extraction_args(args: dict[str, Any]) -> dict[str, Any]:
    """统一工具调用参数：空串/null 转 None；sites 若为 JSON 字符串则解析为 list。"""
    out: dict[str, Any] = {}
    for key, value in args.items():
        if value is None or value == "null" or value == "":
            out[key] = None
            continue
        if key == "sites" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                out[key] = list(parsed) if isinstance(parsed, list) else None
            except (json.JSONDecodeError, TypeError):
                out[key] = None
            continue
        out[key] = value
    return out


def hydrate_price_preferences(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """图入口节点：按 user_id 从 SQLite 载入历史价位偏好，仅当 state 中对应键为空时写入。"""
    uid = _cfg_uid(config)
    row = get_price_preferences(uid)
    updates: dict[str, Any] = {}
    if not row:
        return updates
    for k in ("travel_price_preference", "site_price_preference", "hotel_price_preference"):
        if state.get(k) is None and row.get(k):
            updates[k] = row[k]
    if updates:
        logger.info("[hydrate] user=%s loaded prefs keys=%s", uid, list(updates.keys()))
    return updates


def normalize_input(state: TravelHelperState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """解析最新用户消息：拆出纯文本、是否含图片等，供 vision 分支与抽取使用。"""
    messages = state.get("messages", [])
    last_human = _last_human_message(messages)
    if not last_human:
        return {
            "current_user_text": "",
            "has_visual_input": False,
            "extraction_input_text": "",
        }
    normalized = MultimodalInputProcessor.normalize_human_content(last_human.content)
    logger.info(
        "[normalize] has_visual=%s text_len=%s content=%s",
        normalized["has_visual_input"],
        len(normalized["current_user_text"] or ""),
        MultimodalInputProcessor.summarize_content_for_log(last_human.content),
    )
    return normalized


def route_input_modality(state: TravelHelperState) -> Literal["vision", "text"]:
    """条件边：含视觉内容则先走 vision_enrich，否则直接进入 extract_info。"""
    return "vision" if state.get("has_visual_input") else "text"


async def vision_enrich(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """多模态：用 Vision 模型把图片内容转为中文描述，并与用户配文合并为 extraction_input_text。"""
    user_text = (state.get("current_user_text") or "").strip()
    messages = state.get("messages", [])
    last_human = _last_human_message(messages)
    raw_content = last_human.content if last_human else ""
    if not state.get("has_visual_input"):
        return {"extraction_input_text": user_text}
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    vision_text = await MultimodalInputProcessor.vision_to_text(
        llm,
        config,
        VISION_SYSTEM_PROMPT,
        raw_content,
        logger=logger,
        log_prefix="[MA Travel 视觉]",
    )
    if user_text and vision_text:
        merged = f"{user_text}\n\n[图片内容识别]\n{vision_text}"
    elif vision_text:
        merged = vision_text
    else:
        merged = user_text or ""
    return {"extraction_input_text": merged}


def _build_tool_call_example(user_input: str, **kwargs: Any) -> List[Any]:
    """构造一条 Human → AI(tool_calls) → Tool 的 few-shot，与 simple_travel_planner 同源模式。

    ``**kwargs`` 仅传本条用户话中应抽取的 ``TravelHelperExtraction`` 字段；未出现的键不要传入，
    以便示例展示「部分字段缺失」时 tool args 中也不应臆造这些键。
    """
    tool_call_id = str(uuid.uuid4())
    allowed = frozenset(TravelHelperExtraction.model_fields)
    args = {k: v for k, v in kwargs.items() if k in allowed}
    return [
        HumanMessage(content=user_input),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": "TravelHelperExtraction",
                    "args": args,
                }
            ],
        ),
        ToolMessage(content="已提取", tool_call_id=tool_call_id),
    ]


def _get_extraction_examples() -> List[Any]:
    """多条 few-shot，覆盖字段全量/全缺/部分缺失（与 simple_travel_planner._get_extraction_examples 同理）。"""
    examples: List[Any] = []
    # 1. 字段尽量齐全（日期用 YYYY-MM-DD）
    examples.extend(
        _build_tool_call_example(
            "五一假期从北京去西安看兵马俑，5月1日出发玩4天，两个人，预算8000人民币，高铁二等座、门票和酒店都选性价比高的。",
            destination="西安",
            origin_city="北京",
            sites=["兵马俑"],
            trip_start_date="2026-05-01",
            trip_duration_days=4,
            party_size=2,
            budget_amount=8000.0,
            budget_currency="CNY",
            travel_price_preference="高铁二等座",
            site_price_preference="性价比高",
            hotel_price_preference="性价比高",
            food_preference="陕西面食",
        )
    )
    # 2. 仅有行程骨架：出发地/目的地/景点/日期，无预算与人数
    examples.extend(
        _build_tool_call_example(
            "下周四想从上海去成都，主要想去大熊猫基地和宽窄巷子。",
            destination="成都",
            origin_city="上海",
            sites=["大熊猫基地", "宽窄巷子"],
            trip_start_date="2026-04-09",
        )
    )
    # 3. 仅预算与人数、币种（无地点）
    examples.extend(
        _build_tool_call_example(
            "我们一共4个人，总预算大概2万，都按人民币算。",
            party_size=4,
            budget_amount=20000.0,
            budget_currency="CNY",
        )
    )
    # 4. 仅价位偏好（无具体城市日期）
    examples.extend(
        _build_tool_call_example(
            "交通想舒服一点可以商务座，门票别省，酒店三星左右就行。",
            travel_price_preference="商务座，舒适优先",
            site_price_preference="不省，可买优速通等",
            hotel_price_preference="三星左右",
        )
    )
    # 5. 仅目的地一句（其余字段缺失）
    examples.extend(
        _build_tool_call_example(
            "今年想去一趟新疆。",
            destination="新疆",
        )
    )
    # 6. 无行程信息可抽（寒暄）
    examples.extend(_build_tool_call_example("你好"))
    # 7. 有目的地与景点，缺出发地与出发日期（常见缺口）
    examples.extend(
        _build_tool_call_example(
            "想去北京环球影城玩两天，还没定从哪走、哪天走。",
            destination="北京",
            sites=["环球影城"],
            trip_duration_days=2,
        )
    )
    # 8. 美食偏好 + 部分行程字段，缺 sites 与预算
    examples.extend(
        _build_tool_call_example(
            "4月4日从上海出发到北京，想吃正宗烤鸭，玩三天，两个人。",
            origin_city="上海",
            destination="北京",
            trip_start_date="2026-04-04",
            trip_duration_days=3,
            party_size=2,
            food_preference="北京烤鸭",
        )
    )
    # 9. 配文 + [图片内容识别]（与 vision_enrich 合并格式一致）
    examples.extend(
        _build_tool_call_example(
            "人数3，玩4天，目的地西安，景点参见上传图片。\n\n[图片内容识别]\n"
            "图1：秦始皇兵马俑博物馆；图2：西安城墙；图3：大雁塔；图4：华山。",
            destination="西安",
            party_size=3,
            trip_duration_days=4,
            sites=["秦始皇兵马俑博物馆", "西安城墙", "大雁塔", "华山"],
        )
    )
    # 10. 追问轮仅补充缺口（示例中不出现 destination，避免重复抽取或误标）
    examples.extend(
        _build_tool_call_example(
            "从北京出发，2026-05-01 开始。",
            origin_city="北京",
            trip_start_date="2026-05-01",
        )
    )
    # 11. 追问轮「字段名：值」写法（与 interrupt 用户回复常见格式一致）
    examples.extend(
        _build_tool_call_example(
            "出发地：上海；开始日期：2026-7-10",
            origin_city="上海",
            trip_start_date="2026-07-10",
        )
    )
    return examples


EXTRACTION_EXAMPLES = _get_extraction_examples()


def _updates_from_extraction(extracted: TravelHelperExtraction) -> dict[str, Any]:
    updates = extracted.model_dump(exclude_none=True)
    if updates.get("budget_amount") is not None:
        updates["budget_mode"] = "declared"
    return updates


async def extract_info(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """对 extraction_input_text 做工具绑定调用，将非空字段合并进 state（覆盖式更新由图合并策略决定）。"""
    text = (state.get("extraction_input_text") or "").strip()
    if not text:
        return {}
    hint = _intake_context_hint(state)
    user_message = f"{text}\n\n{hint}" if hint else text
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_tools = llm.bind_tools(
        [TravelHelperExtraction],
        tool_choice="TravelHelperExtraction",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="examples"),
            ("human", "{user_message}"),
        ]
    )
    sub = get_silent_config(config)
    msgs = prompt.format_messages(examples=EXTRACTION_EXAMPLES, user_message=user_message)

    updates: dict[str, Any] = {}
    resp = None
    try:
        try:
            resp = await llm_tools.ainvoke(msgs, config=sub)
        except Exception as tool_exc:
            logger.warning("[extract] 强制 tool_call 失败，回退普通 bind: %s", tool_exc)
            resp = await llm.bind_tools([TravelHelperExtraction]).ainvoke(msgs, config=sub)
    except Exception as exc:
        logger.error("extract_info failed: %s", exc)
        return {}

    if resp and getattr(resp, "tool_calls", None):
        for tc in resp.tool_calls:
            if tc.get("name") != "TravelHelperExtraction":
                continue
            args = _normalize_extraction_args(dict(tc.get("args") or {}))
            try:
                extracted = TravelHelperExtraction(**args)
            except Exception as exc:
                logger.warning("TravelHelperExtraction 校验失败: %s", exc)
                continue
            updates.update(_updates_from_extraction(extracted))
        if updates:
            logger.info("[extract] keys=%s", list(updates.keys()))
            return updates

    logger.warning("[extract] 无 tool_calls，回退 with_structured_output(TravelHelperExtraction)")
    try:
        structured = llm.with_structured_output(TravelHelperExtraction)
        extracted = await structured.ainvoke(msgs, config=sub)
        updates = _updates_from_extraction(extracted)
        if updates:
            logger.info("[extract] keys=%s (structured)", list(updates.keys()))
    except Exception as exc:
        logger.warning("[extract] structured_output 失败: %s", exc)
    return updates


def soft_missing_fields(state: TravelHelperState) -> List[str]:
    """规划前「软」必填：缺则可通过 interrupt 追问（受 INTAKE_COLLECT_MAX_ROUNDS 限制）。"""
    missing: List[str] = []
    if not coerce_state_str(state.get("destination")):
        missing.append("destination")
    if not coerce_state_str(state.get("origin_city")):
        missing.append("origin_city")
    if not coerce_state_str(state.get("trip_start_date")):
        missing.append("trip_start_date")
    sites = state.get("sites")
    if not sites or (isinstance(sites, list) and len(sites) == 0):
        missing.append("sites")
    return missing


def _intake_context_hint(state: TravelHelperState) -> str:
    """抽取时告知 LLM 会话已收集的槽位，追问轮勿重复输出。"""
    known: list[str] = []
    dest = coerce_state_str(state.get("destination"))
    if dest:
        known.append(f"目的地={dest}")
    origin = coerce_state_str(state.get("origin_city"))
    if origin:
        known.append(f"出发地={origin}")
    start = coerce_state_str(state.get("trip_start_date"))
    if start:
        known.append(f"开始日期={start}")
    sites = state.get("sites")
    if sites:
        known.append(f"景点={sites}")
    if not known:
        return ""
    missing = soft_missing_fields(state)
    missing_labels = "、".join(FIELD_LABELS.get(f, f) for f in missing) if missing else "无"
    return (
        f"【会话已收集】{'；'.join(known)}。"
        f"仍缺：{missing_labels}。"
        "请根据用户本轮消息，通过 TravelHelperExtraction 补充或更新相应字段。"
    )


def route_intake_collect(state: TravelHelperState) -> Literal["ask", "proceed"]:
    """抽取后路由：仍缺字段且未超追问上限 → ask_missing；否则进入默认值填充。"""
    rnd = state.get("intake_collect_round") or 0
    missing = soft_missing_fields(state)
    if rnd < INTAKE_COLLECT_MAX_ROUNDS and missing:
        return "ask"
    if missing:
        logger.info("[intake] 追问轮次已达上限，仍缺字段: %s", missing)
    return "proceed"


async def _build_interrupt_multimodal_update(
    user_response: Any,
    config: RunnableConfig,
) -> dict[str, Any]:
    """interrupt 回合：不把 image_url 写入 checkpoint，在节点内完成 vision→文本。"""
    logger.info(
        "[interrupt] 用户回复摘要: %s",
        MultimodalInputProcessor.summarize_content_for_log(user_response),
    )
    update = build_interrupt_text_message_update(user_response)
    normalized = MultimodalInputProcessor.normalize_human_content(user_response)
    text_only = (normalized.get("current_user_text") or "").strip()

    if normalized.get("has_visual_input"):
        llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        vision_text = await MultimodalInputProcessor.vision_to_text(
            llm,
            config,
            VISION_SYSTEM_PROMPT,
            user_response,
            logger=logger,
            log_prefix="[MA Travel interrupt 视觉]",
        )
        if text_only and vision_text:
            merged = f"{text_only}\n\n[图片内容识别]\n{vision_text}"
        elif vision_text:
            merged = vision_text
        else:
            merged = text_only or ""

        update["extraction_input_text"] = merged
        if merged:
            update["messages"] = [HumanMessage(content=merged)]
        elif "messages" in update:
            del update["messages"]

    update, fallback_applied = MultimodalInputProcessor.apply_interrupt_image_only_fallback(
        update,
        has_visual_input=bool(normalized.get("has_visual_input")),
        fallback_text=INTERRUPT_IMAGE_ONLY_FALLBACK_TEXT,
    )
    if fallback_applied:
        logger.warning("[interrupt] 仅图输入且识别失败，写入哨兵消息")
    return update


async def ask_missing(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """interrupt 列出缺失字段；恢复后将用户回复写入 messages 并递增 intake_collect_round，回到 normalize。"""
    missing = soft_missing_fields(state)
    lines = [f"{i+1}. 请补充：{FIELD_LABELS.get(f, f)}" for i, f in enumerate(missing)]
    prompt = "为完成旅行规划，还需要以下信息：\n" + "\n".join(lines)
    user_response = interrupt(prompt)
    update = await _build_interrupt_multimodal_update(user_response, config)
    update["intake_collect_round"] = (state.get("intake_collect_round") or 0) + 1
    return update


def apply_intake_defaults(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """补全未声明的天数、人数、餐宿参数、价位中性描述；无景点时用「目的地+市区经典游览」兜底。

    ``intake_defaults_applied`` 记录实际填入的默认值，供确认页展示透明性。
    """
    applied: dict[str, Any] = {}
    updates: dict[str, Any] = {}
    if state.get("trip_duration_days") is None:
        updates["trip_duration_days"] = DEFAULT_TRIP_DAYS
        applied["trip_duration_days"] = DEFAULT_TRIP_DAYS
    if state.get("party_size") is None:
        updates["party_size"] = DEFAULT_PARTY
        applied["party_size"] = DEFAULT_PARTY
    if state.get("occupancy_per_room") is None:
        updates["occupancy_per_room"] = 2
    if state.get("meals_per_day") is None:
        updates["meals_per_day"] = 2
    if state.get("budget_amount") is not None:
        updates["budget_mode"] = "declared"
    else:
        updates["budget_mode"] = "unspecified"
    if state.get("budget_currency") is None and state.get("budget_amount") is not None:
        updates["budget_currency"] = "CNY"
    for key in ("travel_price_preference", "site_price_preference", "hotel_price_preference"):
        if not coerce_state_str(state.get(key)):
            updates[key] = NEUTRAL_PRICE_PREF
            applied[key] = NEUTRAL_PRICE_PREF
    if not coerce_state_str(state.get("food_preference")):
        updates["food_preference"] = "无特殊"
    sites = state.get("sites")
    dest = coerce_state_str(state.get("destination"))
    if (not sites or len(sites) == 0) and dest:
        updates["sites"] = [f"{dest}市区经典游览"]
        applied["sites"] = updates["sites"]
    merged_defaults = dict(state.get("intake_defaults_applied") or {})
    merged_defaults.update(applied)
    out: dict[str, Any] = dict(updates)
    if merged_defaults:
        out["intake_defaults_applied"] = merged_defaults
    return out


def _fmt_money(x: Any) -> str:
    """Markdown 表格中金额展示，非法输入则 str 原样返回。"""
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return str(x)


def build_intake_summary_md(state: TravelHelperState) -> str:
    """生成第二段 HITL（intake_confirm）展示的 Markdown（不向用户暴露内部默认字段 JSON）。"""
    lines = [
        "## 请确认以下出行信息（第二遍可直接回复「确认」若无误）",
        "",
        f"- **目的地**: {state.get('destination') or '（空）'}",
        f"- **出发地**: {state.get('origin_city') or '（空）'}",
        f"- **开始日期**: {state.get('trip_start_date') or '（空）'}",
        f"- **天数**: {state.get('trip_duration_days')}",
        f"- **人数**: {state.get('party_size')}",
        f"- **景点/活动**: {state.get('sites')}",
        f"- **美食偏好**: {state.get('food_preference')}",
        f"- **交通价位偏好**: {state.get('travel_price_preference')}",
        f"- **门票价位偏好**: {state.get('site_price_preference')}",
        f"- **酒店价位偏好**: {state.get('hotel_price_preference')}",
        f"- **总预算**: {state.get('budget_amount') if state.get('budget_amount') is not None else '未声明'} {state.get('budget_currency') or ''}",
        "",
        "若天数、人数或价位等未主动说明，上表可能含系统预设；可直接回复 **确认**，或写出要修改的内容。",
        "",
        "回复 **确认** 开始检索；如需修改请直接写出正确信息。",
    ]
    return "\n".join(lines)


def critical_intake_ok(state: TravelHelperState) -> bool:
    """用户口头「确认」时仍需满足的最小集合；不满足则拒绝进入 persist/检索。"""
    if not coerce_state_str(state.get("destination")):
        return False
    if not coerce_state_str(state.get("origin_city")):
        return False
    if not coerce_state_str(state.get("trip_start_date")):
        return False
    sites = state.get("sites") or []
    if not sites:
        return False
    return True


def _approval_text(raw: Any) -> str:
    """从 interrupt 返回值中提取用于判断是否「确认」的小写文本（兼容 str / dict 载荷）。"""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip().lower()
    if isinstance(raw, dict):
        for k in ("text", "content", "message"):
            v = raw.get(k)
            if isinstance(v, str):
                return v.strip().lower()
    return str(raw).strip().lower()


def _interrupt_user_text(raw: Any) -> str:
    """定价阶段：保留用户原始反馈全文（大小写不强制），供 parse_pricing_rerun_targets 解析。"""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for k in ("text", "content", "message"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(raw).strip()


def user_approves(text: str) -> bool:
    """中英混合关键词判断用户是否同意当前摘要/报价。"""
    t = text.lower()
    return any(k in t for k in ("确认", "同意", "没问题", "可以", "ok", "yes"))


async def intake_user_confirm(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """Intake 第二段 HITL：展示摘要 → interrupt → 确认则 intake_confirmed，否则带用户修改回写 messages。"""
    md = build_intake_summary_md(state)
    raw = interrupt(md)
    text = _approval_text(raw)
    agreed = user_approves(text)
    rnd = (state.get("intake_confirm_round") or 0) + 1
    if agreed and critical_intake_ok(state):
        return {"intake_confirmed": True, "intake_confirm_round": rnd, "intake_summary_md": md}
    if agreed and not critical_intake_ok(state):
        return {
            "intake_confirmed": False,
            "intake_confirm_round": rnd,
            "messages": [
                AIMessage(
                    content="关键信息仍不完整（目的地、出发地、开始日期、景点列表），请补充后再确认。"
                )
            ],
        }
    upd = await _build_interrupt_multimodal_update(raw, config)
    upd["intake_confirmed"] = False
    upd["intake_confirm_round"] = rnd
    return upd


def persist_price_preferences(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """用户确认 intake 后，将三类价位偏好写入 SQLite，供下次会话 hydrate。"""
    uid = _cfg_uid(config)
    try:
        upsert_price_preferences(
            uid,
            travel_price_preference=state.get("travel_price_preference"),
            site_price_preference=state.get("site_price_preference"),
            hotel_price_preference=state.get("hotel_price_preference"),
        )
        logger.info("[pref.db] upsert user=%s", uid)
    except Exception as exc:
        logger.warning("[pref.db] upsert failed: %s", exc)
    return {}


def supervisor_dispatch(state: TravelHelperState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """占位节点：便于日志与将来扩展（如动态选 worker）；当前无状态变更。

    参数须命名为 ``state`` / ``config``：LangGraph / Runnable 只会向名为 ``config`` 的形参注入
    ``RunnableConfig``；若写成 ``_config``，Studio 等路径下可能只传入 state，导致缺参报错。
    """
    logger.info("[supervisor] fan-out six research workers")
    return {}


async def rerun_workers_node(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """定价未通过时调用：按 pricing_rerun_targets 并行重跑部分 worker，合并回 research。"""
    return await rerun_selected_workers({**state}, config)


def _parse_item_date(val: Any) -> date | None:
    """将 line_item 上的日期字段解析为 date；支持 YYYY-MM-DD 子串与 date 实例。"""
    if val is None or val == "":
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    s = str(val).strip()[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _trip_date_window(state: TravelHelperState) -> tuple[date, date] | None:
    """根据行程开始日与天数得到闭区间 [w0, w1]（含首尾日）；解析失败返回 None。"""
    start_s = coerce_state_str(state.get("trip_start_date"))[:10]
    try:
        ds = datetime.strptime(start_s, "%Y-%m-%d").date()
    except ValueError:
        return None
    days = int(state.get("trip_duration_days") or DEFAULT_TRIP_DAYS)
    de = ds + timedelta(days=max(days - 1, 0))
    return ds, de


def _hotel_row_in_window(it: dict[str, Any], w0: date, w1: date) -> tuple[bool, bool]:
    """酒店行：用 start_date/end_date 与行程窗口求交；无日期则保留（keep=True, parsed=False）。"""
    sd = _parse_item_date(it.get("start_date") or it.get("date"))
    ed = _parse_item_date(it.get("end_date"))
    if sd is None and ed is None:
        return True, False
    if sd is not None and ed is not None:
        if ed < w0 or sd > w1:
            return False, True
        return True, True
    if sd is not None:
        return w0 <= sd <= w1, True
    assert ed is not None
    return w0 <= ed <= w1, True


def _single_date_row_in_window(it: dict[str, Any], w0: date, w1: date) -> tuple[bool, bool]:
    """单日类计价行（交通/餐/票）：单日期落在窗口内则保留。"""
    d = _parse_item_date(it.get("date") or it.get("start_date"))
    if d is None:
        return True, False
    return w0 <= d <= w1, True


def _city_mismatch_hint(it: dict[str, Any], dest: str, origin: str) -> bool:
    """城市名与目的地/出发地子串均不匹配时记一条「提示」统计（不据此删行，避免误杀）。"""
    c = (it.get("city") or it.get("city_name") or "")
    if not isinstance(c, str) or len(c.strip()) < 2:
        return False
    c = c.strip()
    if not dest and not origin:
        return False
    if dest and (c in dest or dest in c):
        return False
    if origin and (c in origin or origin in c):
        return False
    return bool(dest or origin)


def _filter_block_items(
    block: dict[str, Any],
    w0: date,
    w1: date,
    kind: Literal["hotel", "dated"],
    dest: str,
    origin: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """过滤单个 research 子块中的 line_items，重算 subtotal，并返回对齐统计供 alignment_report 使用。"""
    stats = {"dropped_out": 0, "kept_unparsed_date": 0, "city_mismatch": 0}
    items_in = block.get("line_items") or []
    kept: list[Any] = []
    for it in items_in:
        if not isinstance(it, dict):
            kept.append(it)
            continue
        if _city_mismatch_hint(it, dest, origin):
            stats["city_mismatch"] += 1
        if kind == "hotel":
            keep, parsed = _hotel_row_in_window(it, w0, w1)
        else:
            keep, parsed = _single_date_row_in_window(it, w0, w1)
        if not parsed:
            stats["kept_unparsed_date"] += 1
        if not keep:
            stats["dropped_out"] += 1
            continue
        kept.append(it)
    out = dict(block)
    out["line_items"] = kept
    total = 0.0
    for it in kept:
        if isinstance(it, dict):
            try:
                total += float(it.get("line_total") or 0)
            except (TypeError, ValueError):
                pass
    out["subtotal"] = total
    return out, stats


def _block_subtotal(block: dict[str, Any] | None) -> float:
    """优先读 block['subtotal']，否则对 line_items 逐项累加 line_total。"""
    if not block:
        return 0.0
    st = block.get("subtotal")
    if st is not None:
        try:
            return float(st)
        except (TypeError, ValueError):
            pass
    total = 0.0
    for it in block.get("line_items") or []:
        if not isinstance(it, dict):
            continue
        try:
            total += float(it.get("line_total") or 0)
        except (TypeError, ValueError):
            continue
    return total


def _sum_line_items_totals(items: list[Any]) -> float:
    t = 0.0
    for it in items:
        if isinstance(it, dict):
            try:
                t += float(it.get("line_total") or 0)
            except (TypeError, ValueError):
                pass
    return t


# 四类计价 worker 并行返回后仅按键合并，模型常在「交通」里塞酒店/餐/票或在「酒店」里塞大交通。
# 以下在 harness 层按标题+概述做规则归桶与同桶去重，不依赖提示词自律。
_PRICED_RESEARCH_KEYS: tuple[str, ...] = ("transport", "hotels", "food", "tickets")
_PRICING_TICKETS_RE = re.compile(
    r"门票|入园|套票|联票|观光票|预约票|讲解员|"
    r"景交|观光车|环保车|摆渡车|"
    r"(?:兵马俑|大雁塔|故宫|博物院|博物馆)(?:景区|景点)?(?:门票|联票)?|"
    r"成人票|学生票|老年票",
    re.UNICODE,
)
# 「\d+天\d+晚」不得单独作为酒店判据：易与「4天3晚美食总览」等餐饮行冲突；若晚后文出现餐饮语义则排除。
_PRICING_HOTEL_DATENIGHT_DISALLOW = (
    r"(?:美食(?!街|巷|城|广场)|美食总览|餐饮总览|用餐总览|伙食总览|"
    r"餐费|每餐|餐\s*/\s*天|日均餐|吃饭|自助餐|"
    r"(?:早餐|午餐|晚餐)|人均.{0,10}餐|每人.{0,10}餐|"
    r"\d+\s*天\s*×\s*\d+\s*餐|×\s*\d+\s*餐|\d+\s*餐\b|meals?\b)"
)
_PRICING_HOTELS_RE = re.compile(
    r"酒店|宾馆|住宿|民宿|客栈|房费|房型|含早|不含早|连住|"
    r"(?:三星|四星|五星|3星|4星|5星)(?:级)?(?:以上)?(?:酒店|宾馆|住宿)?|"
    r"(?:\d+天\d+晚|\d+晚\d+天)(?!.*"
    + _PRICING_HOTEL_DATENIGHT_DISALLOW
    + r")|"
    r"每.{0,2}晚(?!.*(?:餐|美食|用餐|伙食))",
    re.UNICODE | re.IGNORECASE,
)
_PRICING_FOOD_RE = re.compile(
    r"餐饮|餐费|用餐|伙食|美食(?!(?:街|巷|城|广场))|美食总览|餐饮总览|伙食总览|吃饭|自助餐|加餐|"
    r"(?:早餐|午餐|晚餐)|日均餐|餐\s*/\s*天|"
    r"(?:每人|人均).{0,10}餐|每餐|"
    r"备用金|应急支出|其他杂费|杂费(?!.*票)",
    re.UNICODE,
)
_PRICING_TRANSPORT_RE = re.compile(
    r"高铁|动车|城际铁路|飞机(?:票)?|航班|机票|民航|"
    r"火车(?:票|票价)?|普快|特快|长途汽车|大巴|客运|包车|租车|自驾|"
    r"地铁|轻轨|公交|打车|出租|网约车|滴滴|市内交通|接驳|"
    r"二等座|一等座|商务座|经济舱|头等舱|"
    r"往返.{0,12}(?:高铁|火车|飞机|动车|交通)|大交通",
    re.UNICODE,
)


def _line_item_text_for_bucket(it: dict[str, Any]) -> str:
    t = str(it.get("title") or it.get("name") or "")
    e = str(it.get("evidence_summary") or "")
    return f"{t} {e}".strip()


def _infer_priced_block_key(text: str) -> str | None:
    if not text.strip():
        return None
    if _PRICING_TICKETS_RE.search(text):
        return "tickets"
    # 酒店先于餐饮：避免「五星酒店自助餐」等整行被食品关键词抢走；「4天3晚美食*」靠 HOTELS 中
    # datenight 负向断言与 FOOD 中 美食总览 等规则区分。
    if _PRICING_HOTELS_RE.search(text):
        return "hotels"
    if _PRICING_FOOD_RE.search(text):
        return "food"
    if _PRICING_TRANSPORT_RE.search(text):
        return "transport"
    return None


def _rebucket_priced_line_items_across_workers(research: dict[str, Any]) -> str:
    """将四类 research 子块中的 line_items 按语义归到唯一桶，并做同桶指纹去重。

    在 ``mobility_align_and_budget`` 中先于日期窗口过滤调用，避免并行 worker
    「全家桶」式 JSON 导致四类合计重复累加。
    """
    classified: list[tuple[str, Any]] = []
    moved = 0
    for src in _PRICED_RESEARCH_KEYS:
        blk = research.get(src)
        if not isinstance(blk, dict):
            continue
        for it in blk.get("line_items") or []:
            if not isinstance(it, dict):
                classified.append((src, it))
                continue
            target = _infer_priced_block_key(_line_item_text_for_bucket(it)) or src
            if target != src:
                moved += 1
            classified.append((target, copy.deepcopy(it)))

    buckets: dict[str, list[Any]] = {k: [] for k in _PRICED_RESEARCH_KEYS}
    for target, it in classified:
        if target in buckets:
            buckets[target].append(it)

    deduped = 0
    for key in _PRICED_RESEARCH_KEYS:
        raw = buckets[key]
        seen: set[tuple[str, float]] = set()
        out: list[Any] = []
        for it in raw:
            if not isinstance(it, dict):
                out.append(it)
                continue
            title = re.sub(r"\s+", "", str(it.get("title") or it.get("name") or ""))[:160]
            try:
                lt = round(float(it.get("line_total") or 0), 2)
            except (TypeError, ValueError):
                lt = 0.0
            fp = (title, lt)
            if title and fp in seen:
                deduped += 1
                continue
            if title:
                seen.add(fp)
            out.append(it)

        blk = research.get(key)
        if isinstance(blk, dict):
            blk["line_items"] = out
            st = 0.0
            for row in out:
                if isinstance(row, dict):
                    try:
                        st += float(row.get("line_total") or 0)
                    except (TypeError, ValueError):
                        pass
            blk["subtotal"] = st

    if moved or deduped:
        logger.info(
            "[pricing.rebucket] moved_across_buckets=%s same_bucket_deduped=%s",
            moved,
            deduped,
        )
        return f"计价行已按类别自动归并（跨桶调整 {moved} 条，同桶去重 {deduped} 条）"
    return ""


def _apply_research_pricing_fallbacks(research_work: dict[str, Any], state: TravelHelperState) -> None:
    """检索失败或模型输出明显不合理时，为交通/酒店/餐饮/门票写入保守占位行，避免报价表空表或小计为 0。"""
    party = max(1, int(state.get("party_size") or DEFAULT_PARTY))
    days = max(1, int(state.get("trip_duration_days") or DEFAULT_TRIP_DAYS))
    meals = max(1, int(state.get("meals_per_day") or 2))
    start_s = coerce_state_str(state.get("trip_start_date"))[:10]
    origin = coerce_state_str(state.get("origin_city"))
    dest = coerce_state_str(state.get("destination"))
    end_label = ""
    if start_s:
        try:
            dt0 = datetime.strptime(start_s, "%Y-%m-%d")
            end_label = (dt0 + timedelta(days=max(days - 1, 0))).strftime("%Y-%m-%d")
        except ValueError:
            end_label = ""

    tblk = research_work.get("transport")
    if isinstance(tblk, dict):
        items = list(tblk.get("line_items") or [])
        if _sum_line_items_totals(items) < 1.0:
            if origin and dest and origin != dest:
                per_person_rt = 2400.0
            else:
                per_person_rt = 400.0
            lt = per_person_rt * party
            tblk["line_items"] = [
                {
                    "title": f"{origin or '出发地'}↔{dest or '目的地'} 大交通保守估算（检索无可靠单价）",
                    "date": f"{start_s}～{end_label}" if end_label else start_s,
                    "city": dest or origin,
                    "unit_price": per_person_rt,
                    "quantity": float(party),
                    "line_total": lt,
                    "pricing_basis": "per_person",
                    "evidence_summary": "系统默认跨城往返交通量级，非实时票价",
                }
            ]
            tblk["subtotal"] = lt

    fblk = research_work.get("food")
    if isinstance(fblk, dict):
        items = list(fblk.get("line_items") or [])
        cur = _sum_line_items_totals(items)
        min_total = float(meals * days * party * 40)
        if cur < min_total * 0.55:
            per_meal = 70.0
            n_meals = meals * days * party
            lt = per_meal * n_meals
            span = f"{start_s}～{end_label}" if start_s and end_label else (start_s or "行程窗内")
            fblk["line_items"] = [
                {
                    "title": f"全程餐饮粗估（{meals} 餐/天 × {days} 天 × {party} 人）",
                    "date": span,
                    "city": dest,
                    "unit_price": per_meal,
                    "quantity": float(n_meals),
                    "line_total": lt,
                    "pricing_basis": "per_person",
                    "evidence_summary": "检索摘要不足以覆盖全程餐饮时的保守合计",
                }
            ]
            fblk["subtotal"] = lt

    hblk = research_work.get("hotels")
    if isinstance(hblk, dict):
        h_items = list(hblk.get("line_items") or [])
        if _sum_line_items_totals(h_items) < 1.0:
            occ = max(1, int(state.get("occupancy_per_room") or 2))
            n_rooms = max(1, (party + occ - 1) // occ)
            nights = max(1, days - 1) if days > 1 else 1
            per_room_night = 400.0
            qty = float(n_rooms * nights)
            lt = per_room_night * qty
            span_sd, span_ed = start_s, end_label or start_s
            hblk["line_items"] = [
                {
                    "title": f"{dest or '目的地'} 住宿保守估算（检索无有效酒店报价）",
                    "start_date": span_sd,
                    "end_date": span_ed,
                    "city": dest,
                    "unit_price": per_room_night,
                    "quantity": qty,
                    "line_total": lt,
                    "pricing_basis": "per_room_night",
                    "evidence_summary": (
                        f"按约{per_room_night:.0f}元/间/晚×{n_rooms}间×{nights}晚粗估（{n_rooms}间按每间住{occ}人推算），"
                        "非实时房价，检索失败时的占位价"
                    ),
                }
            ]
            hblk["subtotal"] = lt

    tkblk = research_work.get("tickets")
    if isinstance(tkblk, dict):
        tk_items = list(tkblk.get("line_items") or [])
        if _sum_line_items_totals(tk_items) < 1.0:
            sites = state.get("sites") or []
            if not isinstance(sites, list):
                sites = []
            n_attr = max(1, len(sites)) if sites else 2
            label = "、".join(str(s) for s in sites[:6]) if sites else "主要景点"
            per_ticket = 150.0
            qty_t = float(party * n_attr)
            lt = per_ticket * qty_t
            span = f"{start_s}～{end_label}" if start_s and end_label else (start_s or "行程窗内")
            tkblk["line_items"] = [
                {
                    "title": f"{dest or '目的地'} 门票保守估算（检索无有效报价）",
                    "date": start_s,
                    "city": dest,
                    "unit_price": per_ticket,
                    "quantity": qty_t,
                    "line_total": lt,
                    "pricing_basis": "per_person",
                    "evidence_summary": (
                        f"按成人票约{per_ticket:.0f}元/点×{n_attr}个景点×{party}人粗估；"
                        f"景点：{label}。非官方实时价，检索失败时的占位合计"
                    ),
                }
            ]
            tkblk["subtotal"] = lt


async def mobility_align_and_budget(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """六路 worker fan-in：归桶/日期过滤/预算汇总，与用户预算比较。

    写入 mobility_timeline、alignment_report、budget_breakdown、budget_vs_user；
    通过 ``research`` delta 写回合并后的计价子块。报价 Markdown 由后续 ``build_pricing_md`` 确定性生成，此处不调 LLM。
    """
    research_work: dict[str, Any] = dict(state.get("research") or {})
    rebucket_msg = _rebucket_priced_line_items_across_workers(research_work)
    dest = coerce_state_str(state.get("destination"))
    origin = coerce_state_str(state.get("origin_city"))
    start = coerce_state_str(state.get("trip_start_date"))
    days = int(state.get("trip_duration_days") or DEFAULT_TRIP_DAYS)
    try:
        dt = datetime.strptime(start[:10], "%Y-%m-%d")
        end = dt + timedelta(days=max(days - 1, 0))
        end_s = end.strftime("%Y-%m-%d")
    except ValueError:
        end_s = "未知"
    mobility = (
        f"行程窗口 {start or '（未解析）'} 起共 {days} 天（含首尾至 {end_s}）："
        f"{origin} → {dest} → 返回 {origin} 的单枢纽假设；计价行按窗口过滤后汇总。"
    )
    win = _trip_date_window(state)
    research_delta: dict[str, Any] = {}
    align_parts: list[str] = []
    if win:
        w0, w1 = win
        for key, kind in (
            ("hotels", "hotel"),
            ("food", "dated"),
            ("tickets", "dated"),
            ("transport", "dated"),
        ):
            blk = research_work.get(key)
            if not isinstance(blk, dict):
                continue
            filtered, st = _filter_block_items(blk, w0, w1, kind, dest, origin)
            if st["dropped_out"] or st["kept_unparsed_date"] or st["city_mismatch"]:
                align_parts.append(
                    f"{key}: 剔除窗口外 {st['dropped_out']} 条；"
                    f"未解析日期保留 {st['kept_unparsed_date']} 条；"
                    f"城市与目的地/出发地提示 {st['city_mismatch']} 条。"
                )
            research_delta[key] = filtered
        align = (
            "；".join(align_parts)
            if align_parts
            else f"日期窗口 {w0}～{w1}：已解析日期的计价行均在窗内，或未标注日期者已保留。"
        )
    else:
        align = "无法解析 trip_start_date，未对 line_items 做日期过滤；请人工核对报价日期。"
    if rebucket_msg:
        align = f"{rebucket_msg}；{align}"
    for rk, rv in research_delta.items():
        research_work[rk] = rv
    _apply_research_pricing_fallbacks(research_work, state)
    final_research_delta = dict(research_delta)
    for k in ("transport", "food", "hotels", "tickets"):
        blk = research_work.get(k)
        if isinstance(blk, dict):
            final_research_delta[k] = blk
    transport = research_work.get("transport") or {}
    t = _block_subtotal(transport if isinstance(transport, dict) else {})
    h = _block_subtotal((research_work.get("hotels") if isinstance(research_work.get("hotels"), dict) else {}) or {})
    f = _block_subtotal((research_work.get("food") if isinstance(research_work.get("food"), dict) else {}) or {})
    tk = _block_subtotal((research_work.get("tickets") if isinstance(research_work.get("tickets"), dict) else {}) or {})
    grand = t + h + f + tk
    bd = {
        "transport": t,
        "hotel": h,
        "food": f,
        "ticket": tk,
        "grand_total": grand,
        "currency": state.get("budget_currency") or "CNY",
    }
    bmode = state.get("budget_mode")
    amt = state.get("budget_amount")
    if amt is None or bmode == "unspecified":
        vs = "unspecified"
    else:
        try:
            cap = float(amt)
            if grand <= cap:
                vs = "under_budget"
            elif grand <= cap * 1.1:
                vs = "within_tolerance"
            else:
                vs = "over_cap"
        except (TypeError, ValueError):
            vs = "unspecified"
    out: dict[str, Any] = {
        "mobility_timeline": mobility,
        "alignment_report": align,
        "budget_breakdown": bd,
        "budget_vs_user": vs,
    }
    if final_research_delta:
        out["research"] = final_research_delta
    return out


def _md_table_cell(val: Any) -> str:
    """表格单元格：不截断字数；去掉换行并替换 | 以免破坏 Markdown 表格。"""
    s = "" if val is None else str(val)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return s.replace("|", "｜").strip()


def _md_table(title: str, block: dict[str, Any] | None) -> str:
    """将单个 research 子块渲染为 Markdown 表格，用于定价 interrupt 文案。"""
    lines = [f"### {title}", "", "| 项目 | 日期/说明 | 单价 | 数量 | 小计 | 概述 |", "|---|---|---|---|---|---|"]
    if not block:
        lines.append("| — | — | — | — | — | 无数据 |")
        return "\n".join(lines)
    for it in block.get("line_items") or []:
        if not isinstance(it, dict):
            continue
        d0 = it.get("date")
        if d0:
            date_cell = _md_table_cell(d0)
        else:
            sd, ed = it.get("start_date"), it.get("end_date")
            date_cell = _md_table_cell(f"{sd}～{ed}" if sd and ed else (sd or ed or ""))
        lines.append(
            "| {title} | {date} | {up} | {qty} | {lt} | {ev} |".format(
                title=_md_table_cell(it.get("title", it.get("name", ""))),
                date=date_cell,
                up=_md_table_cell(it.get("unit_price", "")),
                qty=_md_table_cell(it.get("quantity", "")),
                lt=_md_table_cell(it.get("line_total", "")),
                ev=_md_table_cell(it.get("evidence_summary", "")),
            )
        )
    st = block.get("subtotal")
    lines.append("")
    lines.append(f"**小计（估算）**: {_fmt_money(st) if st is not None else '见上行累计'} {block.get('currency') or 'CNY'}")
    return "\n".join(lines)


def _budget_vs_user_label(vs: str | None) -> str:
    return {
        "unspecified": "未提供可比对预算",
        "under_budget": "低于您声明的总预算",
        "within_tolerance": "在声明预算约 10% 容差内",
        "over_cap": "超过声明的总预算",
    }.get(vs or "", vs or "未提供可比对预算")


def build_pricing_md(state: TravelHelperState) -> str:
    """聚合四张表 + 合计 + 预算对照 + mobility/alignment 说明，供 pricing_confirm 展示。"""
    research = state.get("research") or {}
    bd = state.get("budget_breakdown") or {}
    parts = [
        "## 报价估算（请确认或提出修改）",
        "",
        _md_table("交通", research.get("transport") if isinstance(research.get("transport"), dict) else {}),
        "",
        _md_table("酒店", research.get("hotels") if isinstance(research.get("hotels"), dict) else {}),
        "",
        _md_table("餐饮", research.get("food") if isinstance(research.get("food"), dict) else {}),
        "",
        _md_table("门票", research.get("tickets") if isinstance(research.get("tickets"), dict) else {}),
        "",
        f"**四类合计（估算）**: {_fmt_money(bd.get('grand_total'))} {bd.get('currency') or 'CNY'}",
        f"**相对您的总预算**: {_budget_vs_user_label(state.get('budget_vs_user'))}",
        "",
        f"**行程与对齐**: {state.get('mobility_timeline')}",
        f"**对齐说明**: {state.get('alignment_report')}",
        "",
        "回复 **确认** 生成最终行程；或说明要调整的项目（如换高铁、酒店降档）。",
    ]
    return "\n".join(parts)


def pricing_user_confirm(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """第三段 HITL：将确定性渲染的报价全文经 interrupt 交用户确认或反馈。"""
    md = build_pricing_md(state)
    raw = interrupt(md)
    text = _approval_text(raw)
    agreed = user_approves(text)
    rnd = (state.get("pricing_confirm_round") or 0) + 1
    if agreed:
        return {"pricing_confirmation_md": md, "pricing_confirm_round": rnd, "pricing_user_ok": True}
    if rnd >= PRICING_CONFIRM_MAX_ROUNDS:
        return {"pricing_confirmation_md": md, "pricing_confirm_round": rnd, "pricing_user_ok": True}
    feedback = _interrupt_user_text(raw)
    upd = build_interrupt_text_message_update(raw)
    upd["pricing_confirm_round"] = rnd
    upd["pricing_user_ok"] = False
    upd["pricing_feedback_text"] = feedback
    upd["pricing_rerun_targets"] = parse_pricing_rerun_targets(feedback)
    return upd


def route_pricing_ok(state: TravelHelperState) -> Literal["compose", "rerun"]:
    """pricing_confirm 之后：已确认 → 合成行程；否则 → 局部重跑 workers。"""
    if state.get("pricing_user_ok"):
        return "compose"
    return "rerun"


async def compose_itinerary(state: TravelHelperState, config: RunnableConfig) -> dict[str, Any]:
    """最终 LLM：综合天气/文化叙事、预算 breakdown、用户行程字段生成 Markdown 行程单。

    返回中附带 ``travel_helper_checkpoint_tail_reset()``，与 simple_travel_planner 的 create_itinerary 相同：
    生成 itinerary 的同时把状态机上业务字段清空，便于同一线程上下一轮从 intake 重新收集。
    """
    research = state.get("research") or {}
    w = (research.get("weather") or {}).get("narrative", "") if isinstance(research.get("weather"), dict) else ""
    c = (research.get("culture") or {}).get("narrative", "") if isinstance(research.get("culture"), dict) else ""
    bd = state.get("budget_breakdown") or {}
    # 与 simple_travel_planner.create_itinerary 一致：最终行程单走主 config，便于 handlers 流式下发 token
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    ctx = (
        f"目的地={state.get('destination')}, 出发={state.get('origin_city')}, "
        f"日期={state.get('trip_start_date')}, 天数={state.get('trip_duration_days')}, "
        f"人数={state.get('party_size')}, 景点={state.get('sites')}\n"
        f"预算对照={state.get('budget_vs_user')}, 合计={bd}\n"
        f"天气摘要={w[:1200]}\n文化摘要={c[:1200]}\n"
        f"交通/酒店/美食/门票详见前述 research。"
    )
    prompt = [
        HumanMessage(
            content=(
                "你是旅行规划师。请用中文输出一份**充实**的 Markdown 行程单，避免寥寥数段敷衍。\n"
                "硬性要求：\n"
                "1) 按「每一天」分节（Day1、Day2… 或 第N天），每天写满**上午 / 下午 / 晚间**三块，"
                "每块至少 2～4 句具体安排（景点顺序、大致时段、休息与转场）。\n"
                "2) 单独章节：大交通与市内交通建议、餐饮推荐与忌口提示、门票/预约提醒、"
                "预算汇总（与上下文合计呼应，标明估算）、行李与天气/安全注意事项。\n"
                "3) 若天数>1，严禁只写「多日游览」一句话带过；应拆到每日。\n"
                "4) 声明价格为估算，可引用上下文中的检索摘要，勿编造精确实时票价。\n\n"
                + ctx
            )
        )
    ]
    prompt = build_llm_messages(prompt, config)
    resp = await llm.ainvoke(prompt, config)
    body = (resp.content or "").strip()
    return {
        **travel_helper_checkpoint_tail_reset(),
        "messages": [AIMessage(content=body)],
        "itinerary": body,
    }


def route_after_intake(state: TravelHelperState) -> Literal["persist", "retry"]:
    """intake_confirm 后：确认则写库并检索；否则回到 normalize_input 用新消息重跑 intake。"""
    return "persist" if state.get("intake_confirmed") else "retry"


# ---------------------------------------------------------------------------
# LangGraph：节点注册与边（START → … → END）
# ---------------------------------------------------------------------------
# 并行语义：自 research_fanout 连出六条边到各 worker_*，LangGraph 会并行调度；
# 随后六条边均汇入 mobility_budget，等价 fan-in 屏障。
workflow = StateGraph(TravelHelperState)
workflow.add_node("prepare_long_term", prepare_long_term_entry)
workflow.add_node("hydrate_prefs", hydrate_price_preferences)
workflow.add_node("normalize_input", normalize_input)
workflow.add_node("vision_enrich", vision_enrich)
workflow.add_node("extract_info", extract_info)
workflow.add_node("ask_missing", ask_missing)
workflow.add_node("apply_defaults", apply_intake_defaults)
workflow.add_node("intake_confirm", intake_user_confirm)
workflow.add_node("persist_prefs", persist_price_preferences)
workflow.add_node("supervisor", supervisor_dispatch)
workflow.add_node("research_fanout", research_fanout)
workflow.add_node("worker_weather", worker_weather)
workflow.add_node("worker_transport", worker_transport)
workflow.add_node("worker_hotel", worker_hotel)
workflow.add_node("worker_food", worker_food)
workflow.add_node("worker_culture", worker_culture)
workflow.add_node("worker_ticket", worker_ticket)
workflow.add_node("rerun_workers", rerun_workers_node)
workflow.add_node("mobility_budget", mobility_align_and_budget)
workflow.add_node("pricing_confirm", pricing_user_confirm)
workflow.add_node("compose", compose_itinerary)


def _route_collect(state: TravelHelperState) -> Literal["ask", "proceed"]:
    """薄包装：满足 LangGraph 对 conditional_edges 可调用签名的习惯。"""
    return route_intake_collect(state)


workflow.add_edge(START, "prepare_long_term")
workflow.add_edge("prepare_long_term", "hydrate_prefs")
workflow.add_edge("hydrate_prefs", "normalize_input")
workflow.add_conditional_edges(
    "normalize_input",
    route_input_modality,
    {"vision": "vision_enrich", "text": "extract_info"},
)
workflow.add_edge("vision_enrich", "extract_info")
workflow.add_conditional_edges(
    "extract_info",
    _route_collect,
    {"ask": "ask_missing", "proceed": "apply_defaults"},
)
workflow.add_edge("ask_missing", "normalize_input")
workflow.add_edge("apply_defaults", "intake_confirm")
workflow.add_conditional_edges(
    "intake_confirm",
    route_after_intake,
    {"persist": "persist_prefs", "retry": "normalize_input"},
)
workflow.add_edge("persist_prefs", "supervisor")
workflow.add_edge("supervisor", "research_fanout")
workflow.add_edge("research_fanout", "worker_weather")
workflow.add_edge("research_fanout", "worker_transport")
workflow.add_edge("research_fanout", "worker_hotel")
workflow.add_edge("research_fanout", "worker_food")
workflow.add_edge("research_fanout", "worker_culture")
workflow.add_edge("research_fanout", "worker_ticket")
workflow.add_edge("worker_weather", "mobility_budget")
workflow.add_edge("worker_transport", "mobility_budget")
workflow.add_edge("worker_hotel", "mobility_budget")
workflow.add_edge("worker_food", "mobility_budget")
workflow.add_edge("worker_culture", "mobility_budget")
workflow.add_edge("worker_ticket", "mobility_budget")
workflow.add_edge("mobility_budget", "pricing_confirm")


def _route_price(state: TravelHelperState) -> Literal["compose", "rerun"]:
    """薄包装：定价阶段条件边。"""
    return route_pricing_ok(state)


workflow.add_conditional_edges(
    "pricing_confirm",
    _route_price,
    {"compose": "compose", "rerun": "rerun_workers"},
)
workflow.add_edge("rerun_workers", "mobility_budget")
workflow.add_edge("compose", END)

# recursion_limit：interrupt 恢复、多轮追问与定价重跑会消耗步数，60 为经验安全余量
multi_agent_travel_helper_agent = workflow.compile().with_config({"recursion_limit": 60})

# try:
#     graph_obj = multi_agent_travel_helper_agent.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('multi_agent_travel_helper_agent.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")