"""
RAG 模块 - 提供通用的知识库检索能力

这个模块为所有智能体提供可复用的 RAG 功能：

组件:
-----
- service.py  : 核心检索服务（文档摄入、向量检索）
- tools.py    : LangChain 工具（SearchKnowledgeTool）
- nodes.py    : LangGraph 节点工厂（create_rag_model_node）
- utils.py    : 工具函数（截断、回退格式化）

快速使用（带工具轮次上限，推荐）:
-------------------------------
```python
from rag import SearchKnowledgeTool, create_rag_model_node, create_model_to_tools_router, create_reset_rounds_node, create_increment_rounds_node, create_force_done_node, tool_rounds_add_reducer
from langgraph.prebuilt import ToolNode

tools = [SearchKnowledgeTool()]
agent = StateGraph(AgentState)  # 状态需含 tool_rounds: Annotated[int, tool_rounds_add_reducer]
agent.add_node("reset_rounds", create_reset_rounds_node())
agent.add_node("model", create_rag_model_node(tools=tools))
agent.add_node("tools", ToolNode(tools))
agent.add_node("increment_rounds", create_increment_rounds_node())
agent.add_node("force_done", create_force_done_node())
agent.set_entry_point("reset_rounds")
agent.add_edge("reset_rounds", "model")
agent.add_conditional_edges("model", create_model_to_tools_router(max_tool_rounds=2), {"tools": "tools", "done": END, "force_done": "force_done"})
agent.add_edge("tools", "increment_rounds")
agent.add_edge("increment_rounds", "model")
agent.add_edge("force_done", END)
```

详细文档见各子模块。
"""

# 便捷导出
from .tools import SearchKnowledgeTool
from .nodes import (
    create_rag_model_node,
    create_model_to_tools_router,
    create_reset_rounds_node,
    create_increment_rounds_node,
    create_tools_node_with_rounds_increment,
    create_force_done_node,
    create_rag_evaluator_node,
    create_memory_vs_kb_router_node,
    tool_rounds_add_reducer,
    detect_user_language,
    DEFAULT_FORCE_DONE_MESSAGE,
)
from .utils import truncate_rag_result, format_rag_fallback_response
from .service import rag_service

__all__ = [
    # 工具
    "SearchKnowledgeTool",
    # 节点与条件边
    "create_rag_model_node",
    "create_model_to_tools_router",
    "create_reset_rounds_node",
    "create_increment_rounds_node",
    "create_tools_node_with_rounds_increment",
    "create_force_done_node",
    "create_rag_evaluator_node",
    "create_memory_vs_kb_router_node",
    "tool_rounds_add_reducer",
    "detect_user_language",
    "DEFAULT_FORCE_DONE_MESSAGE",
    # 工具函数
    "truncate_rag_result",
    "format_rag_fallback_response",
    # 服务
    "rag_service",
]
