"""
RAG 知识库助手智能体

这是一个简洁的 RAG 智能体示例，演示如何使用 /src/rag 模块快速构建
具有知识库检索能力的智能体。

所有 RAG 相关的通用逻辑（语言检测、提示词生成、回退机制等）
都已封装在 /src/rag 模块中，智能体只需专注于图的编排。

图结构约束（防止检索死循环）：
- 使用 tool_rounds 状态对「每轮对话内」的检索次数做硬性上限（MAX_RAG_TOOL_ROUNDS）。
- 不依赖 RemainingSteps（其与全局 recursion_limit 绑定，无法表达「每轮最多 N 次工具」）。
- 达到上限后由条件边路由到 force_done 节点并结束，不再回到 model。
"""

from typing import Annotated, Literal, TypedDict, List

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage

# 导入通用 RAG 模块（含防死循环的 router / 原子计数节点 / force_done 节点）
from rag import (
    SearchKnowledgeTool,
    create_rag_model_node,
    create_model_to_tools_router,
    create_force_done_node,
    create_reset_rounds_node,
    create_tools_node_with_rounds_increment,
    create_rag_evaluator_node,
    create_memory_vs_kb_router_node,
    tool_rounds_add_reducer,
)
from utils.log_utils import get_logger

logger = get_logger(__name__)

# 每轮对话内最多执行的检索轮数，不同智能体可调整此参数
MAX_RAG_TOOL_ROUNDS = 2

# ============================================================================
# 状态定义
# ============================================================================
class AgentState(TypedDict):
    """RAG 智能体状态。显式定义以确保 LangGraph 能够正确追踪所有 Key。"""
    messages: Annotated[List[BaseMessage], add_messages]
    tool_rounds: Annotated[int, tool_rounds_add_reducer]
    retrieval_eval: Literal["sufficient", "insufficient", "not_found"]
    remaining_steps: RemainingSteps


# ============================================================================
# 工具配置
# ============================================================================

tools = [SearchKnowledgeTool()]

# ============================================================================
# 图定义 - 复用 rag 的通用能力，仅编排与参数
# ============================================================================

agent = StateGraph(AgentState)


agent.add_node("reset_rounds", create_reset_rounds_node())
agent.add_node("memory_router", create_memory_vs_kb_router_node())
agent.add_node("model", create_rag_model_node(tools=tools, safety_check=True))
agent.add_node("tools", create_tools_node_with_rounds_increment(ToolNode(tools)))
agent.add_node("evaluator", create_rag_evaluator_node())
agent.add_node("force_done", create_force_done_node())

agent.set_entry_point("reset_rounds")
agent.add_edge("reset_rounds", "memory_router")
agent.add_edge("memory_router", "model")

# 核心路由逻辑：100% 依赖 state.tool_rounds
agent.add_conditional_edges(
    "model",
    create_model_to_tools_router(max_tool_rounds=MAX_RAG_TOOL_ROUNDS),
    {"tools": "tools", "done": END, "force_done": "force_done"},
)

# 链路：tools (rounds+1) -> evaluator (eval) -> model
agent.add_edge("tools", "evaluator")
agent.add_edge("evaluator", "model")
agent.add_edge("force_done", END)

# 编译；设置 recursion_limit 作为安全网
rag_assistant = agent.compile().with_config({"recursion_limit": 20})

# try:
#     graph_obj = rag_assistant.get_graph()
#     pic = graph_obj.draw_mermaid_png()
#     with open('state_graph_rag_assistant.png', 'wb') as f:
#         f.write(pic)
# except Exception as e:
#     logger.warning(f"生成图例失败: {e}")
