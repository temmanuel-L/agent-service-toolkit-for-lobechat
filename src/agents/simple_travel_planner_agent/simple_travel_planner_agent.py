# -*- coding: utf-8 -*-
"""
@Time ： 2026/1/19 19:05
@Auth ： luanxing
@File ：simple_travel_planner_agent.py
@IDE ：PyCharm
"""

from datetime import datetime
from typing import List, Literal

from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt

from core import get_model, settings
from utils.log_utils import get_logger

logger = get_logger(__name__)


class PlannerState(MessagesState, total=False):
    """Planner State."""
    city: str
    interests: List[str]
    itinerary: str


# Configure Tools (optional, for weather lookup)
tools = []
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))


# Itinerary generation prompt template - key to getting good output
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一位专业的旅行规划师。今天的日期是 {date}。\n"
     "请为用户创建一份详细的 {city} 一日游行程。用户的兴趣是：{interests}。\n"
     "请以 Markdown 格式输出，包含时间、地点、活动和建议。"
    ),
    ("human", "请为我规划一日游行程。"),
])


def reset_state(state: PlannerState, config: RunnableConfig) -> dict:
    """Reset the planner state for a new trip. This ensures that after completing
    a trip, the user can start a new one without inheriting old data."""
    # Check if we have a completed itinerary from a previous run
    if state.get("itinerary"):
        logger.info("检测到已完成的行程，重置状态以开始新的旅程规划")
        return {
            "city": None,
            "interests": None,
            "itinerary": None,
        }
    return {}


def input_city(state: PlannerState, config: RunnableConfig) -> dict:
    """Ask for the destination city if missing."""
    city = state.get("city")
    if city:
        logger.info(f"城市已存在: {city}")
        return {}
    
    # Interrupt to ask user for city
    prompt = "请问你想去哪里旅游?"
    logger.info(f"询问城市: {prompt}")
    user_message = interrupt(prompt)
    logger.info(f"用户回复: {user_message}")
    
    return {
        "city": user_message,
        "messages": [HumanMessage(content=user_message)],
    }


def input_interests(state: PlannerState, config: RunnableConfig) -> dict:
    """Ask for interests if missing."""
    interests = state.get("interests")
    if interests:
        logger.info(f"兴趣已存在: {interests}")
        return {}
    
    city = state.get("city", "目的地")
    prompt = f"请告诉我你对{city}的旅游有哪些兴趣？(用逗号分隔，如：美食, 历史古迹, 自然风光)"
    logger.info(f"询问兴趣: {prompt}")
    user_message = interrupt(prompt)
    logger.info(f"用户回复: {user_message}")
    
    interests_list = [interest.strip() for interest in user_message.split(',') if interest.strip()]
    
    return {
        "interests": interests_list,
        "messages": [HumanMessage(content=user_message)],
    }


async def create_itinerary(state: PlannerState, config: RunnableConfig) -> dict:
    """Generate the travel itinerary using the LLM."""
    city = state.get("city", "未知城市")
    interests = state.get("interests", [])
    interests_str = ", ".join(interests) if interests else "未指定"
    
    logger.info(f"生成行程: 城市={city}, 兴趣={interests_str}")
    
    # Get the model
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    # Format the prompt using the template
    current_date = datetime.now().strftime("%Y年%m月%d日")
    formatted_messages = itinerary_prompt.format_messages(
        date=current_date,
        city=city,
        interests=interests_str
    )
    
    # Invoke the LLM directly with the formatted messages
    try:
        response = await llm.ainvoke(formatted_messages, config)
        itinerary_content = response.content
        
        if not itinerary_content or len(itinerary_content.strip()) < 20:
            raise ValueError("Response too short")
            
        logger.info(f"行程生成成功，长度: {len(itinerary_content)}")
        
    except Exception as e:
        logger.error(f"行程生成失败: {e}")
        # Fallback with more detail
        itinerary_content = (
            f"# {city} 一日游行程\n\n"
            f"**兴趣**: {interests_str}\n\n"
            "## 上午\n"
            f"- **9:00** 从市中心出发，前往{city}最具代表性的景点\n"
            "- **10:30** 探索当地特色街区\n\n"
            "## 中午\n"
            f"- **12:00** 品尝{city}特色美食\n\n"
            "## 下午\n"
            f"- **14:00** 根据您对{interests_str}的兴趣，推荐前往相关景点\n"
            "- **16:00** 自由活动，购买纪念品\n\n"
            "## 晚上\n"
            f"- **18:00** 在{city}热门餐厅享用晚餐\n"
            "- **20:00** 欣赏夜景，结束美好的一天\n\n"
            "*提示：建议提前预订热门景点门票和餐厅。*"
        )
    
    return {
        "messages": [AIMessage(content=itinerary_content)],
        "itinerary": itinerary_content,
    }


# Routing logic
def route_after_city(state: PlannerState) -> Literal["input_interests", "input_city"]:
    if state.get("city"):
        return "input_interests"
    return "input_city"


def route_after_interests(state: PlannerState) -> Literal["create_itinerary", "input_interests"]:
    if state.get("interests"):
        return "create_itinerary"
    return "input_interests"


# Graph Construction
workflow = StateGraph(PlannerState)

workflow.add_node("reset_state", reset_state)
workflow.add_node("input_city", input_city)
workflow.add_node("input_interests", input_interests)
workflow.add_node("create_itinerary", create_itinerary)

workflow.set_entry_point("reset_state")

workflow.add_edge("reset_state", "input_city")

workflow.add_conditional_edges(
    "input_city", 
    route_after_city,
    {"input_interests": "input_interests", "input_city": "input_city"}
)

workflow.add_conditional_edges(
    "input_interests",
    route_after_interests,
    {"create_itinerary": "create_itinerary", "input_interests": "input_interests"}
)

workflow.add_edge("create_itinerary", END)

simple_travel_planner_agent = workflow.compile()

