# -*- coding: utf-8 -*-
"""
@Time ： 2026/1/19 19:05
@Auth ： luanxing
@File ：simple_travel_planner_agent.py
@IDE ：PyCharm

Refactored to use LLM-based extraction with tool calling and smart routing.
"""

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings
from utils.log_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# State Definition
# =============================================================================
class PlannerState(MessagesState, total=False):
    """Planner State with optional fields for collected information."""
    destination: Optional[str]
    interests: Optional[List[str]]
    itinerary: Optional[str]


# =============================================================================
# Extraction Schema - The key to LLM-powered information gathering
# =============================================================================
class TravelInfoExtraction(BaseModel):
    """Extract travel planning information from user message.
    
    This schema is used as a tool for the LLM to extract structured data.
    """
    destination: Optional[str] = Field(
        default=None,
        description="The city, country or place the user wants to travel to. "
                    "Extract only if explicitly mentioned."
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="List of user interests for the trip (e.g., food, history, nature, play field, shopping center, etc.). "
                    "Extract as a list only if mentioned."
    )


# =============================================================================
# Required Fields Configuration - Easy to extend
# =============================================================================
REQUIRED_FIELDS = ["destination", "interests"]

FIELD_PROMPTS = {
    "destination": "请问你想去哪个城市或地方旅游？",
    "interests": "请告诉我你对这次旅行有哪些兴趣？（如：美食、历史古迹、自然风光、游乐园或者购物中心等）",
}


def get_missing_fields(state: PlannerState) -> List[str]:
    """Check which required fields are still missing or empty."""
    missing = []
    for required_field in REQUIRED_FIELDS:
        value = state.get(required_field)
        # Consider empty lists as missing too
        if value is None or (isinstance(value, list) and len(value) == 0):
            missing.append(required_field)
    logger.info(f'缺失字段为：{missing}')
    return missing


# =============================================================================
# Few-shot Examples for Tool Calling - Correct Pattern
# =============================================================================
def _build_tool_call_example(user_input: str, destination: Optional[str], interests: Optional[List[str]]) -> List:
    """Build a single few-shot example with proper tool_calls format."""
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
    """Build all few-shot examples."""
    examples = []
    # Example 1: destination only
    examples.extend(_build_tool_call_example("我想去北京", "北京", None))
    # Example 2: interests only
    examples.extend(_build_tool_call_example("我喜欢历史古迹和美食", None, ["历史古迹", "美食"]))
    # Example 3: interests (specific places)
    examples.extend(_build_tool_call_example("去看故宫和长城", None, ["故宫", "长城"]))
    # Example 4: nothing to extract
    examples.extend(_build_tool_call_example("你好", None, None))
    return examples


# Pre-build examples once
EXTRACTION_EXAMPLES = _get_extraction_examples()

# System prompt
EXTRACTION_SYSTEM_PROMPT = """你是一个信息提取助手。你的任务是从用户消息中提取旅行规划信息。

规则：
1. destination：提取城市或国家名称。具体景点如"长城"、"故宫"属于 interests，不是 destination。
2. interests：提取景点名称或活动类型，如：长城、故宫、美食、购物。
3. 只提取明确提到的信息，不猜测。
4. 必须调用 TravelInfoExtraction 工具返回结果。"""


# =============================================================================
# Node: Extract Information using LLM + Tool Calling
# =============================================================================
async def extract_info(state: PlannerState, config: RunnableConfig) -> dict:
    """Extract travel information from the latest user message using LLM with tool calling."""
    messages = state.get("messages", [])
    if not messages:
        logger.info("No messages to extract from, skipping extraction")
        return {}
    
    # Get the model and bind the extraction tool
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    llm_with_tools = llm.bind_tools([TravelInfoExtraction])
    
    # Prompt with examples placeholder
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="examples"),
        ("human", "{user_message}"),
    ])
    
    # Get the last human message for extraction
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    
    if not last_human_msg:
        logger.info("No human message found, skipping extraction")
        return {}
    
    logger.info(f"Extracting info from: {last_human_msg}")
    
    try:
        # Invoke LLM with tool calling - include few-shot examples
        formatted_messages = extraction_prompt.format_messages(
            user_message=last_human_msg,
            examples=EXTRACTION_EXAMPLES
        )
        response = await llm_with_tools.with_config(tags=["skip_stream"]).ainvoke(formatted_messages, config)
        
        # Parse tool calls to get extracted data
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = tool_call["args"]
            
            # Auto-correction: If destination is a list, move it to interests
            if isinstance(args.get("destination"), list):
                logger.warning(f"LLM put list in destination field, auto-correcting: {args['destination']}")
                if not args.get("interests"):
                    args["interests"] = args["destination"]
                args["destination"] = None
            
            extracted = TravelInfoExtraction(**args)
            logger.info(f"Extracted: destination={extracted.destination}, interests={extracted.interests}")
            
            # Build update dict with only non-None values
            updates = {}
            if extracted.destination and not state.get("destination"):
                updates["destination"] = extracted.destination
            if extracted.interests and not state.get("interests"):
                updates["interests"] = extracted.interests
            
            return updates
        else:
            logger.warning("LLM did not return tool calls, trying to parse content")
            return {}
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {}


# =============================================================================
# Node: Ask for Missing Information
# =============================================================================
def ask_missing_info(state: PlannerState, config: RunnableConfig) -> dict:
    """Dynamically ask for missing required fields using interrupt."""
    missing_fields = get_missing_fields(state)
    
    if not missing_fields:
        return {}
    
    # Build prompt for missing fields - use numbered list for better display
    prompts = [f"{i+1}. {FIELD_PROMPTS.get(f, f'请提供{f}')}" for i, f in enumerate(missing_fields)]
    combined_prompt = "为了帮您规划旅行，我需要了解以下信息：\n" + "\n".join(prompts)
    
    logger.info(f"Asking for missing fields: {missing_fields}, prompt: {combined_prompt}")
    
    # Interrupt to get user input
    user_response = interrupt(combined_prompt)
    logger.info(f"User response: {user_response}")
    
    return {
        "messages": [HumanMessage(content=user_response)]
    }


# =============================================================================
# Node: Create Itinerary
# =============================================================================
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "你是一位专业的旅行规划师。今天的日期是 {date}。\n"
     "请为用户创建一份详细的 {destination} 旅游行程。用户的兴趣是：{interests}。\n"
     "请以 Markdown 格式输出，包含时间、地点、活动和建议。\n"
     "如果景点较多，可以安排多日行程。"),
    ("human", "请为我规划行程。"),
])


async def create_itinerary(state: PlannerState, config: RunnableConfig) -> dict:
    """Generate the travel itinerary using the LLM."""
    destination = state.get("destination", "未知目的地")
    interests = state.get("interests", [])
    interests_str = ", ".join(interests) if interests else "未指定"
    
    logger.info(f"Creating itinerary: destination={destination}, interests={interests_str}")
    
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    formatted_messages = itinerary_prompt.format_messages(
        date=current_date,
        destination=destination,
        interests=interests_str
    )
    
    try:
        # Streaming enabled - handlers.py has dedup logic to prevent duplicate messages
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
        # Reset for next trip
        "destination": None,
        "interests": None,
    }


# =============================================================================
# Routing Logic - Smart completeness check
# =============================================================================
def route_by_completeness(state: PlannerState) -> Literal["complete", "incomplete"]:
    """Route based on whether all required fields are filled."""
    missing = get_missing_fields(state)
    if missing:
        logger.info(f"Missing fields: {missing} -> incomplete")
        return "incomplete"
    logger.info("All fields complete -> ready to create itinerary")
    return "complete"


# =============================================================================
# Graph Construction
# =============================================================================
workflow = StateGraph(PlannerState)

# Add nodes
workflow.add_node("extract_info", extract_info)
workflow.add_node("ask_missing", ask_missing_info)
workflow.add_node("create_itinerary", create_itinerary)

# Set entry point
workflow.set_entry_point("extract_info")

# Add edges with smart routing
workflow.add_conditional_edges(
    "extract_info",
    route_by_completeness,
    {
        "complete": "create_itinerary",
        "incomplete": "ask_missing"
    }
)

# After asking, go back to extraction to process the new input
workflow.add_edge("ask_missing", "extract_info")

# End after creating itinerary
workflow.add_edge("create_itinerary", END)

# Compile the graph
simple_travel_planner_agent = workflow.compile()
