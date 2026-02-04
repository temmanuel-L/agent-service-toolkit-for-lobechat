"""
RAG 模块 - 提供通用的知识库检索能力

这个模块为所有智能体提供可复用的 RAG 功能：

组件:
-----
- service.py  : 核心检索服务（文档摄入、向量检索）
- tools.py    : LangChain 工具（SearchKnowledgeTool）
- nodes.py    : LangGraph 节点工厂（create_rag_model_node）
- utils.py    : 工具函数（截断、回退格式化）

快速使用:
--------
```python
from rag.tools import SearchKnowledgeTool
from rag.nodes import create_rag_model_node, pending_tool_calls

# 创建 RAG 智能体
tools = [SearchKnowledgeTool()]
agent = StateGraph(AgentState)
agent.add_node("model", create_rag_model_node(tools=tools))
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")
```

详细文档见各子模块。
"""

# 便捷导出
from .tools import SearchKnowledgeTool
from .nodes import create_rag_model_node, pending_tool_calls, detect_user_language
from .utils import truncate_rag_result, format_rag_fallback_response
from .service import rag_service

__all__ = [
    # 工具
    "SearchKnowledgeTool",
    # 节点
    "create_rag_model_node",
    "pending_tool_calls",
    "detect_user_language",
    # 工具函数
    "truncate_rag_result",
    "format_rag_fallback_response",
    # 服务
    "rag_service",
]
