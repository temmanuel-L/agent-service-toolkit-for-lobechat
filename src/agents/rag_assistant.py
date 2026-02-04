"""
RAG 知识库助手智能体

这是一个简洁的 RAG 智能体示例，演示如何使用 /src/rag 模块快速构建
具有知识库检索能力的智能体。

所有 RAG 相关的通用逻辑（语言检测、提示词生成、回退机制等）
都已封装在 /src/rag 模块中，智能体只需专注于图的编排。
"""
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

# 导入通用 RAG 模块
from rag import SearchKnowledgeTool, create_rag_model_node, pending_tool_calls


# ============================================================================
# 状态定义
# ============================================================================

class AgentState(MessagesState, total=False):
    """RAG 智能体状态"""
    remaining_steps: RemainingSteps


# ============================================================================
# 工具配置
# ============================================================================

tools = [SearchKnowledgeTool()]


# ============================================================================
# 图定义 - 这就是智能体的核心，专注于编排
# ============================================================================

agent = StateGraph(AgentState)

# 添加节点 - 使用通用 RAG 模型节点
agent.add_node("model", create_rag_model_node(tools=tools, safety_check=True))
agent.add_node("tools", ToolNode(tools))

# 设置入口
agent.set_entry_point("model")

# 定义边
agent.add_conditional_edges(
    "model",
    pending_tool_calls,
    {"tools": "tools", "done": END}
)
agent.add_edge("tools", "model")

# 编译
rag_assistant = agent.compile()
