"""
RAG 通用节点模块

提供可复用的 RAG 节点工厂函数，让智能体编写专注于图的编排。

使用示例:
---------
from rag.nodes import create_rag_model_node
from rag.tools import SearchKnowledgeTool

# 创建 RAG 智能体只需几行
agent = StateGraph(AgentState)
agent.add_node("model", create_rag_model_node(tools=[SearchKnowledgeTool()]))
agent.add_node("tools", ToolNode([SearchKnowledgeTool()]))
agent.set_entry_point("model")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")
"""
from typing import List, Optional, Callable, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState

from core import get_model, settings
from utils.log_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# 语言检测
# ============================================================================

def detect_user_language(messages: list) -> str:
    """
    检测用户使用的语言
    
    规则：只要消息中有任何中文字符，就用中文回答
    否则用英文回答
    
    Args:
        messages: 消息列表
        
    Returns:
        "中文" 或 "English"
    """
    user_texts = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            user_texts.append(content)
            if len(user_texts) >= 3:
                break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_texts.append(msg.get("content", ""))
            if len(user_texts) >= 3:
                break
    
    combined_text = " ".join(user_texts)
    if not combined_text:
        return "中文"
    
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in combined_text)
    return "中文" if has_chinese else "English"


# ============================================================================
# 系统提示词生成
# ============================================================================

def create_rag_system_prompt(
    kb_ids: List[str] = None,
    user_language: str = "中文",
    custom_instructions: str = None
) -> str:
    """
    创建 RAG 系统提示词
    
    Args:
        kb_ids: 知识库 ID 列表
        user_language: 用户语言
        custom_instructions: 自定义附加指令
        
    Returns:
        完整的系统提示词
    """
    if not kb_ids:
        kb_info = ""
        tool_instruction = "当前未绑定知识库，无法使用 search_knowledge。若用户询问文档内容，请告知其需要先绑定知识库。"
    else:
        kb_info = f"本对话已绑定知识库: {', '.join(kb_ids)}。"
        # 引导 LLM 先检索再回答，但明确禁止重复调用
        tool_instruction = """
**检索流程规则**:
1. 收到用户问题后，先调用 search_knowledge 工具检索相关信息
2. 收到检索结果后，直接根据结果回答用户问题，**不要再次调用工具**
3. 如果检索结果不相关或为空，看看用户提问与长期记忆是否相关。
4. **禁止**连续多次调用同一工具或使用相似查询重复检索"""

    base_prompt = f"""【强制】用{user_language}回答。无论检索内容是什么语言，输出必须是{user_language}。

你是知识库助手。{kb_info}
{tool_instruction}

**回答原则**:
1. 简洁回答 - 直接回应用户问题，不需要重复检索结果原文
2. 不编造 - 找不到就说找不到，不要凭空捏造答案
3. 用{user_language}输出"""

    if custom_instructions:
        base_prompt += f"\n\n{custom_instructions}"
    
    return base_prompt


# ============================================================================
# 消息预处理
# ============================================================================

def filter_frontend_messages(messages: list) -> list:
    """
    过滤前端（如 LobeChat）发送的系统消息
    
    前端可能会注入与我们指令冲突的 SystemMessage，
    这个函数将其过滤掉，只保留用户和助手的消息。以及长期记忆消息
    
    Args:
        messages: 原始消息列表
        
    Returns:
        过滤后的消息列表
    """
    filtered = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            source = getattr(msg, "additional_kwargs", {}) or {}
            if source.get("source") == "long_term_memory":
                filtered.append(msg)
                continue
            continue
        filtered.append(msg)
    return filtered


# ============================================================================
# 通用 RAG 模型节点工厂
# ============================================================================

def create_rag_model_node(
    tools: List[BaseTool],
    system_prompt_fn: Callable[[List[str], str], str] = None,
    safety_check: bool = False,
    max_tool_iterations: int = 5
):
    """
    创建通用的 RAG 模型调用节点
    
    这个工厂函数返回一个可以直接用于 LangGraph 的节点函数，
    封装了所有 RAG 相关的通用逻辑：
    - 语言检测
    - 系统提示词生成
    - 前端消息过滤
    - 模型调用
    - 可选的安全检查
    - 工具调用迭代限制
    
    Args:
        tools: 工具列表（通常包含 SearchKnowledgeTool）
        system_prompt_fn: 自定义系统提示词生成函数，签名为 (kb_ids, language) -> str
        safety_check: 是否启用 LlamaGuard 安全检查
        max_tool_iterations: 最大工具调用迭代次数
        
    Returns:
        可用于 StateGraph.add_node() 的异步函数
        
    使用示例:
    ---------
    from rag.nodes import create_rag_model_node
    from rag.tools import SearchKnowledgeTool
    
    tools = [SearchKnowledgeTool()]
    model_node = create_rag_model_node(tools=tools)
    
    agent = StateGraph(AgentState)
    agent.add_node("model", model_node)
    """
    
    # 使用默认的系统提示词函数
    if system_prompt_fn is None:
        system_prompt_fn = create_rag_system_prompt
    
    async def rag_model_node(state: MessagesState, config: RunnableConfig) -> dict:
        """RAG 模型调用节点"""
        
        # 1. 从配置中获取知识库 ID（由前端 LobeChat 在请求体中传入）
        kb_ids = config["configurable"].get("kb_ids") or []
        if not kb_ids:
            logger.info(
                "RAG 未收到 kb_ids，知识库检索将不可用。请确认对话是否已绑定知识库且前端请求传入了 kb_ids。"
            )
        
        # 2. 设置模型和工具
        model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        bound_model = model.bind_tools(tools)
        
        # 3. 过滤前端系统消息
        filtered_messages = filter_frontend_messages(state["messages"])
        
        # 4. 检测用户语言
        user_language = detect_user_language(state["messages"])
        
        # 5. 生成系统提示词
        system_prompt = system_prompt_fn(kb_ids, user_language)
        system_msg = SystemMessage(content=system_prompt)
        
        # 6. 组装消息
        messages = [system_msg] + filtered_messages
        
        # 7. 调用模型
        response = await bound_model.ainvoke(messages, config)
        
        # 8. 可选的安全检查
        if safety_check:
            try:
                from agents.llama_guard import LlamaGuard, SafetyAssessment
                llama_guard = LlamaGuard()
                safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
                if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
                    return {
                        "messages": [AIMessage(content=f"此对话被标记为不安全内容: {', '.join(safety_output.unsafe_categories)}")]
                    }
            except Exception as e:
                logger.warning(f"安全检查跳过: {e}")
        
        # 9. 检查工具调用迭代限制
        remaining_steps = state.get("remaining_steps", max_tool_iterations)
        if remaining_steps < 2 and response.tool_calls:
            return {
                "messages": [AIMessage(id=response.id, content="抱歉，需要更多步骤来处理此请求。")]
            }
        
        return {"messages": [response]}
    
    return rag_model_node


# ============================================================================
# 便捷的条件边函数
# ============================================================================

def pending_tool_calls(state: MessagesState) -> str:
    """
    检查是否有待处理的工具调用
    
    用于 StateGraph 的条件边，判断是否需要执行工具。
    
    Args:
        state: 当前状态
        
    Returns:
        "tools" 如果有工具调用，否则 "done"
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "done"
