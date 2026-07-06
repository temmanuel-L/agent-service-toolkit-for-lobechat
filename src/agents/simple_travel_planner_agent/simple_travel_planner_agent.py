# -*- coding: utf-8 -*-
"""
@Time ： 2026/1/19 19:05
@Auth ： luanxing
@File ：simple_travel_planner_agent.py
@IDE ：PyCharm

已重构为基于 LLM 的信息抽取，配合工具调用与智能路由。
"""

import json
import uuid
from datetime import datetime
from typing import Any, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.types import interrupt
from pydantic import BaseModel, Field, field_validator

from agents.multimodal_input_processor import MultimodalInputProcessor
from agents.utils import build_interrupt_text_message_update, coerce_optional_str
from core import get_model, settings
from memory.long_term_concat_for_agents import build_llm_messages, prepare_long_term_entry
from utils.log_utils import get_logger

logger = get_logger(__name__)
INTERRUPT_IMAGE_ONLY_FALLBACK_TEXT = "[本轮仅收到图片，暂未识别出可用文本]"


# =============================================================================
# 状态定义
# =============================================================================
class PlannerState(MessagesState, total=False):
    """规划器状态，包含可选的已收集信息字段。"""
    destination: Optional[str]
    interests: Optional[List[str]]
    itinerary: Optional[str]
    remaining_steps: RemainingSteps
    # 当前轮输入快照：每一轮从最新 HumanMessage 规范化得到
    current_user_text: Optional[str]
    has_visual_input: Optional[bool]
    # extract_info 的唯一输入源
    extraction_input_text: Optional[str]


# =============================================================================
# 抽取模式 - LLM 驱动信息收集的关键
# =============================================================================
class TravelInfoExtraction(BaseModel):
    """从用户消息中抽取旅行规划信息。
    
    该模式作为工具供 LLM 抽取结构化数据。
    """
    destination: Optional[str] = Field(
        default=None,
        description="用户想要前往的城市、国家或地点。仅在明确提及时抽取。"
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="用户对本次旅行的兴趣列表（如：美食、历史、自然、游乐园、购物中心等）。仅在提及时以列表形式抽取。"
    )

    @field_validator("destination", mode="before")
    @classmethod
    def _unwrap_destination(cls, v: Any) -> Any:
        return coerce_optional_str(v)

    @field_validator("interests", mode="before")
    @classmethod
    def _parse_interests_json(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return v


# =============================================================================
# 必填字段配置 - 便于扩展
# =============================================================================
REQUIRED_FIELDS = ["destination", "interests"]

FIELD_PROMPTS = {
    "destination": "请问你想去哪个城市或地方旅游？",
    "interests": "请告诉我你对这次旅行有哪些兴趣？（如：美食、历史古迹、自然风光、游乐园或者购物中心等）",
}


def _last_human_message(messages: List[Any]) -> HumanMessage | None:
    """从后向前取最后一条 HumanMessage。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def get_missing_fields(state: PlannerState) -> List[str]:
    """检查哪些必填字段仍缺失或为空。"""
    missing = []
    for required_field in REQUIRED_FIELDS:
        value = state.get(required_field)
        # 空列表也视为缺失
        if value is None or (isinstance(value, list) and len(value) == 0):
            missing.append(required_field)
    logger.info(f'缺失字段为：{missing}')
    return missing


# =============================================================================
# 工具调用的 Few-shot 示例
# =============================================================================
def _build_tool_call_example(
        user_input: str,
        destination: Optional[str],
        interests: Optional[List[str]]) -> List:
    """构建一个符合 tool_calls 格式的 few-shot 示例。"""
    tool_call_id = str(uuid.uuid4())
    return [
        HumanMessage(content=user_input),
        AIMessage(content="", tool_calls=[{
            "id": tool_call_id,
            "name": "TravelInfoExtraction",
            "args": {"destination": destination, "interests": interests}
        }]),
        ToolMessage(content="已成功提取信息", tool_call_id=tool_call_id),
    ]


def _get_extraction_examples() -> List:
    """构建所有 few-shot 示例。"""
    examples = []
    # 示例 1：仅目的地
    examples.extend(_build_tool_call_example("我想去北京", "北京", None))
    # 示例 2：仅兴趣
    examples.extend(_build_tool_call_example("我喜欢历史古迹和美食", None, ["历史古迹", "美食"]))
    # 示例 3：兴趣（具体地点）
    examples.extend(_build_tool_call_example("去看故宫和长城", None, ["故宫", "长城"]))
    # 示例 4：无信息可抽取
    examples.extend(_build_tool_call_example("你好", None, None))
    return examples


# 预构建示例（只构建一次）
EXTRACTION_EXAMPLES = _get_extraction_examples()

# 系统提示
EXTRACTION_SYSTEM_PROMPT = """你是一个信息提取助手。你的任务是从用户消息中提取旅行规划信息。

规则：
1. destination：提取城市或国家名称。具体景点如"长城"、"故宫"属于 interests，不是 destination。
2. interests：提取景点名称或活动类型，如：长城、故宫、美食、购物。
3. 只提取明确提到的信息，不猜测。
4. 必须调用 TravelInfoExtraction 工具返回结果。"""

VISION_SYSTEM_PROMPT = """你是旅行场景下的图像理解助手。请根据用户提供的图片（及附带文字），用简洁中文输出识别结果，便于后续抽取旅行目的地与兴趣。

请尽量覆盖（有则写出，无则省略，不要编造）：
- 国家、地区/省份、城市
- 景点、地标、场馆等具体名称

直接输出一段自然语言描述即可，不要输出 JSON 或列表编号。"""


def _normalize_extraction_args(args: dict[str, Any]) -> dict[str, Any]:
    """将 LLM 返回的 tool call args 归一化：智谱等可能返回 'null' 或 '["x"]' 等字符串。"""
    out = {}
    for key, value in args.items():
        if value is None or value == "null" or value == "":
            out[key] = None
            continue
        out[key] = value
    return out


# =============================================================================
# 节点：规范化当前轮输入
# =============================================================================
def normalize_input(state: PlannerState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """从最新 HumanMessage 生成当前轮输入快照，供后续路由/视觉/抽取统一消费。"""
    messages = state.get("messages", [])
    last_human = _last_human_message(messages)
    if not last_human:
        logger.info("[规范化输入] 无 HumanMessage，清空当前轮快照")
        return {
            "current_user_text": "",
            "has_visual_input": False,
            "extraction_input_text": "",
        }

    normalized = MultimodalInputProcessor.normalize_human_content(last_human.content)
    logger.info(
        "[规范化输入] has_visual=%s text_len=%s",
        normalized["has_visual_input"],
        len(normalized["current_user_text"] or ""),
    )
    return normalized


# =============================================================================
# 路由：纯文本走抽取；含图先视觉理解
# =============================================================================
def route_input_modality(state: PlannerState) -> Literal["vision", "text"]:
    """根据规范化后的状态判断先视觉理解或直接进入抽取。"""
    if state.get("has_visual_input"):
        logger.info("[路由] 检测到图像 -> vision_enrich")
        return "vision"
    logger.info("[路由] 纯文本 -> extract_info")
    return "text"


# =============================================================================
# 节点：多模态视觉理解（写入 extraction_input_text）
# =============================================================================
async def vision_enrich(state: PlannerState, config: RunnableConfig) -> dict:
    """对含图输入调用多模态模型，将识别结果写入 extraction_input_text。"""
    logger.info(f"--- [视觉理解] 剩余步数: {state.get('remaining_steps')} ---")
    user_text = (state.get("current_user_text") or "").strip()
    messages = state.get("messages", [])
    last_human = _last_human_message(messages)
    raw_content = last_human.content if last_human else ""
    logger.info(f"[视觉理解] 用户可见文本: {user_text[:300]!r}")
    if not state.get("has_visual_input"):
        logger.info("[视觉理解] 当前轮无图像输入，直接沿用 extraction_input_text")
        return {"extraction_input_text": user_text}

    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    vision_text = await MultimodalInputProcessor.vision_to_text(
        llm,
        config,
        VISION_SYSTEM_PROMPT,
        raw_content,
        logger=logger,
        log_prefix="[视觉理解]",
    )
    logger.info(f"[视觉理解] 模型输出内容: {vision_text[:200]!r}")

    if user_text and vision_text:
        merged = f"{user_text}\n\n[图片内容识别]\n{vision_text}"
    elif vision_text:
        merged = vision_text
    else:
        merged = user_text or ""

    if not merged:
        logger.warning("[视觉理解] 合并结果为空，仍进入抽取（可能无有效信息）")
    else:
        logger.info(f"[视觉理解] 写入 extraction_input_text: {merged[:300]!r}")

    return {"extraction_input_text": merged}


# =============================================================================
# 节点：使用 LLM + 工具调用抽取信息
# =============================================================================
async def extract_info(state: PlannerState, config: RunnableConfig) -> dict:
    """使用 LLM 与工具调用从最新用户消息中抽取旅行信息。"""
    logger.info(f"--- [抽取信息] 剩余步数: {state.get('remaining_steps')} ---")
    extraction_input_text = (state.get("extraction_input_text") or "").strip()
    if not extraction_input_text:
        logger.info("当前轮无有效 extraction_input_text，跳过抽取")
        return {}

    # 获取模型并绑定抽取工具
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([TravelInfoExtraction])

    # 带示例占位符的提示
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="examples"),
        ("human", "{user_message}"),
    ])
    logger.info(f"[抽取信息] extraction_input_text 长度: {len(extraction_input_text)}")
    logger.info(f"[抽取信息] extraction_input_text 内容: {extraction_input_text[:300]!r}")
    if "[图片内容识别]" in extraction_input_text:
        vision_section = extraction_input_text.split("[图片内容识别]", 1)[1].strip()
        logger.info(f"[抽取信息] 图片识别片段: {vision_section[:200]!r}")

    try:
        # Invoke LLM with tool calling - include few-shot examples
        formatted_messages = extraction_prompt.format_messages(
            user_message=extraction_input_text,
            examples=EXTRACTION_EXAMPLES
        )
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(formatted_messages, config)

        # 解析工具调用以得到抽取结果
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = dict(tool_call.get("args") or {})
            # 智谱等模型有时将 tool call args 中的 null/数组以 JSON 字符串形式返回，需归一化为 Python 类型
            args = _normalize_extraction_args(args)

            extracted = TravelInfoExtraction(**args)
            logger.info(f"Extracted: destination={extracted.destination}, interests={extracted.interests}")

            # 构建状态更新字典
            updates: dict[str, Any] = {}

            # 1. 状态感知的目的地逻辑：若已有目的地，新抽取的目的地降级为兴趣
            if extracted.destination:
                if not state.get("destination"):
                    updates["destination"] = extracted.destination
                else:
                    logger.info(
                        f"Destination already set ({state['destination']}), demoting '{extracted.destination}' to interests"
                    )
                    if extracted.interests is None:
                        extracted.interests = []
                    if extracted.destination not in extracted.interests:
                        extracted.interests.append(extracted.destination)

            # 2. 兴趣累加逻辑：将新抽取的兴趣与已有兴趣合并（去重）
            if extracted.interests:
                existing_interests = state.get("interests") or []
                # 用 dict.fromkeys 保持顺序并去重
                merged_interests = list(dict.fromkeys(existing_interests + extracted.interests))
                updates["interests"] = merged_interests

            return updates
        else:
            logger.warning("LLM 未返回工具调用，尝试解析内容")
            return {}

    except Exception as e:
        logger.error(f"抽取失败: {e}")
        return {}


# =============================================================================
# 节点：询问缺失信息
# =============================================================================
async def ask_missing_info(state: PlannerState, config: RunnableConfig) -> dict:
    """通过 interrupt 动态询问缺失的必填字段。"""
    logger.info(f"--- [ASK MISSING] Remaining steps: {state.get('remaining_steps')} ---")
    missing_fields = get_missing_fields(state)
    
    if not missing_fields:
        return {}
    
    # Build prompt for missing fields - use numbered list for better display
    prompts = [f"{i+1}. {FIELD_PROMPTS.get(f, f'请提供{f}')}" for i, f in enumerate(missing_fields)]
    combined_prompt = "为了帮您规划旅行，我需要了解以下信息：\n" + "\n".join(prompts)
    
    logger.info(f"正在询问缺失字段: {missing_fields}, 提示: {combined_prompt}")
    
    # 通过 interrupt 获取用户输入
    user_response = interrupt(combined_prompt)
    logger.info(f"用户回复摘要: {MultimodalInputProcessor.summarize_content_for_log(user_response)}")
    update = build_interrupt_text_message_update(user_response)
    normalized = MultimodalInputProcessor.normalize_human_content(user_response)
    text_only = normalized.get("current_user_text", "")

    # interrupt 回合若含图，直接在当前节点完成图转文；只把文本写入 state
    if normalized.get("has_visual_input"):
        llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        vision_text = await MultimodalInputProcessor.vision_to_text(
            llm,
            config,
            VISION_SYSTEM_PROMPT,
            user_response,
            logger=logger,
            log_prefix="[ASK MISSING 视觉理解]",
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
    # 仅图片且图转文失败时，写入哨兵 HumanMessage，避免 normalize_input 回读旧消息
    update, fallback_applied = MultimodalInputProcessor.apply_interrupt_image_only_fallback(
        update,
        has_visual_input=bool(normalized.get("has_visual_input")),
        fallback_text=INTERRUPT_IMAGE_ONLY_FALLBACK_TEXT,
    )
    if fallback_applied:
        logger.warning("[ASK MISSING] 仅图输入且识别失败，写入哨兵消息并继续追问")

    return update


# =============================================================================
# 节点：创建行程
# =============================================================================
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一位专业的旅行规划师。今天的日期是 {date}。\n"
     "请为用户创建一份**内容充实**的 {destination} 旅游行程（Markdown）。用户的兴趣是：{interests}。\n"
     "要求：按天分节，每天至少写上午/下午/晚间三块，每块多句具体安排；"
     "另含交通、餐饮、预算估算、注意事项等独立小节；勿用寥寥数段概括敷衍。\n"
     "若兴趣点较多，合理拆成多日并写明动线。"),
    ("human", "请为我规划行程。"),
])


async def create_itinerary(state: PlannerState, config: RunnableConfig) -> dict:
    """Generate the travel itinerary using the LLM."""
    logger.info(f"--- [创建行程] 剩余步数: {state.get('remaining_steps')} ---")
    destination = state.get("destination", "未知目的地")
    interests = state.get("interests", [])
    interests_str = ", ".join(interests) if interests else "未指定"
    
    logger.info(f"正在创建行程: destination={destination}, interests={interests_str}")
    
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    formatted_messages = itinerary_prompt.format_messages(
        date=current_date,
        destination=destination,
        interests=interests_str
    )
    agent_system = next(
        (m.content for m in formatted_messages if isinstance(m, SystemMessage)),
        None,
    )
    conversation = [m for m in formatted_messages if not isinstance(m, SystemMessage)]
    formatted_messages = build_llm_messages(
        conversation,
        config,
        agent_system=agent_system if isinstance(agent_system, str) else None,
    )
    
    try:
        # 已开启流式输出，handlers.py 中有去重逻辑避免重复消息
        response = await llm.ainvoke(formatted_messages, config)
        itinerary_content = response.content
        
        if not itinerary_content or len(itinerary_content.strip()) < 20:
            raise ValueError("Response too short")
        
        logger.info(f"Itinerary created, length: {len(itinerary_content)}")
        
    except Exception as e:
        logger.error(f"Itinerary creation failed: {e}")
        itinerary_content = (
            f"# {destination} 旅游行程\n\n"
            f"**兴趣**: {interests_str}\n\n"
            "抱歉，生成详细行程时遇到问题。请稍后重试。"
        )
    
    return {
        "messages": [AIMessage(content=itinerary_content)],
        "itinerary": itinerary_content,
        # 为下一次行程重置
        "destination": None,
        "interests": None,
        "current_user_text": "",
        "has_visual_input": False,
        "extraction_input_text": "",
    }


# =============================================================================
# 路由逻辑 - 基于完整性的智能判断
# =============================================================================
def route_by_completeness(state: PlannerState) -> Literal["complete", "incomplete"]:
    """根据必填字段是否已全部填写进行路由。"""
    missing = get_missing_fields(state)
    if missing:
        logger.info(f"缺失字段: {missing} -> incomplete")
        return "incomplete"
    logger.info("所有字段已完整 -> 可创建行程")
    return "complete"


# =============================================================================
# 图构建
# =============================================================================
workflow = StateGraph(PlannerState)

# 添加节点
workflow.add_node("prepare_long_term", prepare_long_term_entry)
workflow.add_node("normalize_input", normalize_input)
workflow.add_node("vision_enrich", vision_enrich)
workflow.add_node("extract_info", extract_info)
workflow.add_node("ask_missing", ask_missing_info)
workflow.add_node("create_itinerary", create_itinerary)

# 入口：先规范化当前轮输入，再按多模态路由
workflow.add_edge(START, "prepare_long_term")
workflow.add_edge("prepare_long_term", "normalize_input")
workflow.add_conditional_edges(
    "normalize_input",
    route_input_modality,
    {"vision": "vision_enrich", "text": "extract_info"},
)
workflow.add_edge("vision_enrich", "extract_info")

# 添加带智能路由的边
workflow.add_conditional_edges(
    "extract_info",
    route_by_completeness,
    {
        "complete": "create_itinerary",
        "incomplete": "ask_missing"
    }
)

# 询问后重新规范化当前轮输入，再按多模态路由
workflow.add_edge("ask_missing", "normalize_input")

# 创建行程后结束
workflow.add_edge("create_itinerary", END)

# 编译图
simple_travel_planner_agent = workflow.compile().with_config({'recursion_limit': 10})

# try:
#     graph_obj = simple_travel_planner_agent.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('state_graph_simple_travel_planner.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")
