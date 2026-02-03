from datetime import datetime
from typing import Literal, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import (
    RunnableConfig,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from rag.tools import SearchKnowledgeTool
from rag.service import rag_service
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs."""
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    kb_context: str  # To store retrieved context


tools = [SearchKnowledgeTool()]


def get_instructions(kb_ids: List[str] = None) -> str:
    current_date = datetime.now().strftime("%B %d, %Y")
    kb_info = f"Available Knowledge Base IDs: {', '.join(kb_ids)}" if kb_ids else "No specific knowledge bases associated."
    base = f"""
    You are a professional assistant with access to a knowledge base. 
    Today's date is {current_date}.
    {kb_info}

    Your goal is to answer user questions accurately by retrieving information from the knowledge base using the `search_knowledge` tool.
    
    GUIDELINES:
    1. ALWAYS use the `search_knowledge` tool if the user asks a question that requires factual information from the knowledge base.
    2. Once you have the search results, synthesize an answer that directly addresses the user's query. 
    3. DO NOT simply output the raw search results. Provide a coherent, natural language response.
    4. If the search results do not contain the answer, explain what you found and what is missing.
    5. Cite your sources if the context provides document names or IDs.
    6. Be concise and professional.
    """
    return base


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    # 1. Get KB IDs from configuration
    kb_ids = config["configurable"].get("kb_ids") or []
    
    # 2. Setup model and tools
    # We no longer perform automatic retrieval here to avoid redundant calls and overwhelming context.
    # The agent will use the search_knowledge tool as needed.
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    bound_model = m.bind_tools(tools)
    
    # 3. Prepare messages with instructions
    system_msg = SystemMessage(content=get_instructions(kb_ids))
    messages = [system_msg] + state["messages"]
    
    response = await bound_model.with_config(tags=["skip_stream"]).ainvoke(messages, config)

    # 5. Safety Check
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [AIMessage(content=f"This conversation was flagged for unsafe content: {', '.join(safety_output.unsafe_categories)}")],
            "safety": safety_output,
        }

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [AIMessage(id=response.id, content="Sorry, need more steps to process this request.")]
        }
        
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    content = f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    return {"messages": [AIMessage(content=content)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")

def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state.get("safety")
    if safety and safety.safety_assessment == SafetyAssessment.UNSAFE:
        return "unsafe"
    return "safe"

agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)
agent.add_edge("block_unsafe_content", END)
agent.add_edge("tools", "model")

def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "done"

agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

rag_assistant = agent.compile()
