"""
RAG 通用节点模块

提供可复用的 RAG 节点工厂函数，让智能体编写专注于图的编排。

关于 model–tools 死循环：
- 工具只向模型返回文本，不返回 similarity/BM25 分数，也没有「匹配度不好再查」的逻辑。
- 是否再次调用完全由模型根据上下文决定，故有时会循环调用有时不会。
- 通过 create_model_to_tools_router + create_reset_rounds_node + create_tools_node_with_rounds_increment + create_force_done_node
  做轮次上限；+1 与 tools 同一次返回，避免流式/checkpoint 下下一节点读不到更新。

使用示例（推荐，带轮次上限）:
------------------------------
from rag.nodes import (
    create_rag_model_node,
    create_model_to_tools_router,
    create_reset_rounds_node,
    create_tools_node_with_rounds_increment,
    create_force_done_node,
)
from rag.tools import SearchKnowledgeTool
from langgraph.prebuilt import ToolNode

tools = [SearchKnowledgeTool()]
agent = StateGraph(AgentState)  # 状态需含 tool_rounds: Annotated[int, tool_rounds_add_reducer]
agent.add_node("reset_rounds", create_reset_rounds_node())
agent.add_node("model", create_rag_model_node(tools=tools))
agent.add_node("tools", create_tools_node_with_rounds_increment(ToolNode(tools)))
agent.add_node("force_done", create_force_done_node())
agent.set_entry_point("reset_rounds")
agent.add_edge("reset_rounds", "model")
agent.add_conditional_edges("model", create_model_to_tools_router(max_tool_rounds=2),
    {"tools": "tools", "done": END, "force_done": "force_done"})
agent.add_edge("tools", "model")
agent.add_edge("force_done", END)

简单用法（无轮次上限，仅适合不需要 cap 的图）:
-----------------------------------------
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})
agent.add_edge("tools", "model")
"""
from typing import List, Optional, Callable, Literal, Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState

from core import get_model, settings
from utils.log_utils import get_logger

logger = get_logger(__name__)

# 工具轮次达上限时，force_done 节点默认追加的系统提示（可被 create_force_done_node 的 message 覆盖）
DEFAULT_FORCE_DONE_MESSAGE = "[系统] 本轮检索次数已达上限，请基于已获得的检索结果回答或重新提问。"

# 评估提示词：判断检索到的信息是否足以回答用户问题
EVALUATOR_SYSTEM_PROMPT = """你是一个检索质量评估专家。
你的任务是判断最新的检索结果是否足以回答用户的最后一个问题。

**判断标准**:
1. [SUFFICIENT]: 检索结果包含足够的事实性信息（即使是间接的、隐含的或者概括性的），能让模型基于此给出有据可查的回答。
2. [INSUFFICIENT]: 检索结果与问题相关，但确实缺少核心关键事实（如特定的公司名称、合同金额、起止日期等无法推断的信息）。
3. [NOT_FOUND]: 检索结果与问题完全无关或为空。

**注意事项**:
- **宽容原则**：如果检索到的内容定义了角色的含义或提供了背景，即使没有具体名称，只要对回答有帮助，也可在第一轮优先评为 SUFFICIENT，诱导模型先尝试回答。
- **防止幻觉**：如果问题要求的是具体的唯一标识点（如“这家公司的注册资金是多少”），而检索结果没提，则必须评为 INSUFFICIENT。

**输出格式**:
只输出标签（SUFFICIENT/INSUFFICIENT/NOT_FOUND），严禁任何额外解释。"""


def tool_rounds_add_reducer(current: int | None, update: int | None) -> int:
    """
    状态合并逻辑：累加式（用于统计工具调用轮次）。
    """
    return (current or 0) + (update or 0)


def create_reset_rounds_node(tool_rounds_key: str = "tool_rounds") -> Callable:
    """
    返回「入口重置节点」：将 tool_rounds 归零，供每轮新提问时使用。

    与 tool_rounds_add_reducer 配合：返回 -current，当前值加 -current 结果为 0。
    """
    def _node(state: dict) -> dict:
        current = state.get(tool_rounds_key, 0)
        # 1. 用负数抵消当前值，实现归零（配合累加式 reducer）
        # 2. 同时清除上一次的评估状态，防止污染下一轮提问
        logger.info(f"[reset_rounds] {tool_rounds_key}: {current} -> 0, clearing retrieval_eval")
        return {
            tool_rounds_key: -current,
            "retrieval_eval": None
        }
    return _node


def create_increment_rounds_node(tool_rounds_key: str = "tool_rounds") -> Callable:
    """
    返回「原子计数节点」：向 state 增加一次增量。
    """
    def _node(state: dict) -> dict:
        logger.info(f"[increment_rounds] submitting {tool_rounds_key}=1")
        return {tool_rounds_key: 1}
    return _node


def create_tools_node_with_rounds_increment(
    tool_node: Any,
    tool_rounds_key: str = "tool_rounds",
) -> Callable:
    """
    包装 ToolNode：在返回的 state 中带上 tool_rounds: 1。
    配合 additive reducer，这会使总计次数增加 1。
    """
    async def _node(state: dict, config: RunnableConfig) -> dict:
        result = await tool_node.ainvoke(state, config)
        logger.info("[tools_node_with_rounds_increment] %s: +1", tool_rounds_key)
        return {**result, tool_rounds_key: 1}
    return _node


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
        # 引导 LLM 先检索再回答，但明确禁止重复调用，同时兼顾「长期记忆类」问题
        tool_instruction = """
**检索流程规则**:
1. 当用户询问知识库文档中的具体内容（如合同甲方/乙方、公司名、金额、日期、条款、附件内容、论文/报告的作者/标题/期刊名等）时，**必须先调用 search_knowledge 检索**，再根据检索结果回答；不得仅依据长期记忆回答此类问题。
2. 对于「显然是关于用户本人或日常生活偏好」的问题（例如“我是谁”“我喜欢吃什么”“我最近都在干什么”等），优先使用对话上下文和长期记忆回答，**不要调用 search_knowledge**，除非用户明确提到要查某个文档。
3. 收到用户问题后，若本轮对话中还没有调用过 search_knowledge，且问题明显依赖知识库文档中的事实信息，应先调用 search_knowledge 工具检索相关信息。
4. 收到检索结果后，直接根据结果回答用户问题，**不要再次调用工具**。
5. 如果检索结果不相关或为空，可结合长期记忆或说明未在知识库找到。
6. **禁止**连续多次调用同一工具或使用相似查询重复检索。"""

    base_prompt = f"""【强制】用{user_language}回答。无论检索内容是什么语言，输出必须是{user_language}。

你是知识库助手。{kb_info}
{tool_instruction}

**回答原则**:
1. 优先利用现有结果 - 若当前的 `search_knowledge` 结果或上下文已足以回答问题，**严禁**再次调用工具。
2. **事实优先 & 拒绝模糊** - 如果检索结果（Context）中包含具体的实体名称（如公司名、人名）、时间、地点或金额，必须直接提取并回答。严禁在信息已提供时回答“需结合签署页确认”或“未明确提及”。
3. 简洁准确 - 直接回应用户问题，不要重复检索结果原文。
4. 不编造 - 不要凭空捏造答案；仅当完全没有相关检索内容时才说明「未在知识库找到相关内容」。
5. 用{user_language}输出"""

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
    
    async def rag_model_node(state: dict, config: RunnableConfig) -> dict:
        """RAG 模型调用节点"""
        
        # 1. 状态追踪日志
        tool_rounds = state.get("tool_rounds", 0)
        retrieval_eval = state.get("retrieval_eval", "unknown")
        remaining = state.get("remaining_steps", "unknown")
        logger.info(f"[State Dump][rag_model_node] tool_rounds={tool_rounds}, remaining={remaining}, retrieval_eval={retrieval_eval}, keys={list(state.keys())}")

        # 2. 从配置中获取知识库 ID（由前端 LobeChat 在请求体中传入）
        kb_ids = config["configurable"].get("kb_ids") or []
        if not kb_ids:
            logger.info(
                "RAG 未收到 kb_ids，知识库检索将不可用。请确认对话是否已绑定知识库且前端请求传入了 kb_ids。"
            )
        
        # 2. 设置模型和工具（可根据 kb_enabled 决定是否绑定检索工具）
        kb_enabled = state.get("kb_enabled", True)
        model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        if kb_enabled and tools:
            bound_model = model.bind_tools(tools)
        else:
            bound_model = model
        
        # 3. 过滤前端系统消息
        filtered_messages = filter_frontend_messages(state["messages"])
        
        # 4. 检测用户语言
        user_language = detect_user_language(state["messages"])
        
        # 5. 生成系统提示词
        system_prompt = system_prompt_fn(kb_ids, user_language)
        system_msg = SystemMessage(content=system_prompt)

        # 6. 组装消息
        messages = [system_msg] + filtered_messages

        # 6.1 记录本次调用时的检索上下文情况，便于排查「RAG 失效」
        # 查找最近一次 search_knowledge 工具返回的 ToolMessage（若有）
        rag_tool_messages = [
            msg for msg in state.get("messages", [])
            if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "search_knowledge"
        ]
        last_rag_msg: ToolMessage | None = rag_tool_messages[-1] if rag_tool_messages else None
        if last_rag_msg and isinstance(last_rag_msg.content, str):
            preview = last_rag_msg.content[:200].replace("\n", " ")
            logger.info(
                "[rag_model_node] Context Check: tool_rounds=%s, kb_ids=%s, search_knowledge_len=%d",
                tool_rounds,
                kb_ids,
                len(last_rag_msg.content),
            )
        else:
            logger.info(
                "[rag_model_node] Context Check: tool_rounds=%s, kb_ids=%s, no_search_knowledge_context",
                tool_rounds,
                kb_ids,
            )
        
        # 7. 调用模型
        response = await bound_model.ainvoke(messages, config)

        # 7.1 若当前轮已被 memory_vs_kb_router 判定为应启用 KB（kb_enabled=True），
        # 且这是本轮第一次检索尝试（tool_rounds==0 且还没有 search_knowledge 结果），
        # 但模型本次没有提出任何 tool_calls，则强制触发一次 search_knowledge 调用。
        #
        # 这样可以减少「明明是文档类问题却完全不走检索」的随机性，同时不改变图结构。
        kb_enabled = state.get("kb_enabled", True)
        if kb_enabled and tool_rounds == 0:
            # 查找当前 state 中是否已经有 search_knowledge 的 ToolMessage（避免重复强制）
            existing_rag_msgs = [
                msg for msg in state.get("messages", [])
                if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "search_knowledge"
            ]
            no_existing_search = not existing_rag_msgs

            if no_existing_search and (not getattr(response, "tool_calls", None)):
                # 从最近一条用户消息中提取查询文本
                human_messages = [m for m in filtered_messages if isinstance(m, HumanMessage)]
                last_question = human_messages[-1].content if human_messages else ""
                if not isinstance(last_question, str):
                    last_question = str(last_question)

                from uuid import uuid4

                forced_tool_call = {
                    "id": f"auto_search_{uuid4()}",
                    "name": "search_knowledge",
                    "args": {"query": last_question},
                }

                logger.info(
                    "[rag_model_node] 未收到模型的 tool_calls，且 kb_enabled=True，"
                    "自动触发一次 search_knowledge 工具调用。"
                )

                # 用一个只包含 tool_calls 的 AIMessage 替代本次模型响应，
                # 让后续 Router 按正常流程路由到 tools 节点。
                response = AIMessage(content="", tool_calls=[forced_tool_call])
        
        # 8. 可选的安全检查
        if safety_check:
            try:
                from agents.llama_guard import LlamaGuard, SafetyAssessment
                llama_guard = LlamaGuard()
                safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
                if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
                    return {
                        "messages": [AIMessage(content=f"此对话被标记为不安全内容: {', '.join(safety_output.unsafe_categories)}")],
                    }
            except Exception as e:
                logger.warning(f"安全检查跳过: {e}")
        
        # 9. 检查工具调用迭代限制
        remaining_steps = state.get("remaining_steps", max_tool_iterations)
        if remaining_steps < 2 and response.tool_calls:
            return {
                "messages": [AIMessage(id=response.id, content="抱歉，需要更多步骤来处理此请求。")],
                "tool_rounds": 0,
            }
        return {"messages": [response]}
    
    return rag_model_node


def create_rag_evaluator_node(
    model_name: Optional[str] = None,
    tool_names: List[str] = ["search_knowledge"],
) -> Callable:
    """
    创建 RAG 评估节点：评估检索结果是否足以回答问题。
    """
    async def evaluator_node(state: MessagesState, config: RunnableConfig) -> dict:
        remaining = state.get("remaining_steps", "unknown")
        logger.info(f"[State Dump][evaluator_node] tool_rounds={state.get('tool_rounds')}, remaining={remaining}, retrieval_eval={state.get('retrieval_eval')}, keys={list(state.keys())}")
        messages = state.get("messages", [])
        if not messages:
            return {"retrieval_eval": "not_found"}

        # 获取最后一次检索结果（从指定的工具列表中匹配）
        tool_messages = [
            m for m in messages 
            if isinstance(m, ToolMessage) and getattr(m, "name", "") in tool_names
        ]
        if not tool_messages:
            return {"retrieval_eval": "not_found"}
        
        last_tool_msg = tool_messages[-1]
        
        # 获取用户最后一个问题
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if not human_messages:
            return {"retrieval_eval": "not_found"}
        
        last_question = human_messages[-1].content
        
        # 1. 打印评估上下文（全量或大容量预览，供用户确定检索质量）
        max_log_len = 12000 
        content_for_log = last_tool_msg.content[:max_log_len]
        if len(last_tool_msg.content) > max_log_len:
            content_for_log += "... [LOG TRUNCATED]"
            
        logger.info(
            f"[rag_evaluator] Context Check Input:\n"
            f"Question: {last_question}\n"
            f"Context Length: {len(last_tool_msg.content)}\n"
            f"Context Content:\n{content_for_log}\n"
            f"--- [End of Evaluator Context] ---"
        )

        # 2. 调用模型进行评估
        # 深度抑制流式：通过硬编码覆盖 config 字典并显式传参关闭 stream 
        eval_config = {**config, "callbacks": []} if isinstance(config, dict) else config
        
        eval_model = get_model(model_name or config["configurable"].get("model", settings.DEFAULT_MODEL))
        eval_input = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"问题: {last_question}\n\n检索结果: {last_tool_msg.content}")
        ]
        
        try:
            # 显式加入 stream=False（LiteLLM/LangChain 支持此参数）
            response = await eval_model.ainvoke(eval_input, eval_config, stream=False)
            raw_result = response.content if isinstance(response.content, str) else str(response.content)
            raw_result = raw_result.strip().upper()
            
            decision = "not_found"
            if "INSUFFICIENT" in raw_result:
                decision = "insufficient"
            elif "SUFFICIENT" in raw_result:
                decision = "sufficient"
            
            logger.info(f"[rag_evaluator] Result: raw='{raw_result}' -> decision={decision}")
            return {"retrieval_eval": decision}
        except Exception as e:
            logger.error(f"[rag_evaluator] error: {e}")
            return {"retrieval_eval": "insufficient"}  # 出错时默认继续检索
            
    return evaluator_node


def create_memory_vs_kb_router_node(
    model_name: Optional[str] = None,
) -> Callable:
    """
    创建一个「长期记忆 vs 知识库」路由节点。

    作用：
    - 根据用户当前问题与是否绑定 kb_ids，决定本轮是否启用知识库检索：
      - kb_enabled=True  → 允许后续模型节点绑定 search_knowledge 工具；
      - kb_enabled=False → 仅依赖长期记忆与对话上下文，不调用知识库。
    """

    async def router_node(state: MessagesState, config: RunnableConfig) -> dict:
        messages = state.get("messages", [])
        kb_ids = config.get("configurable", {}).get("kb_ids") if isinstance(config, dict) else config.configurable.get("kb_ids")  # type: ignore[attr-defined]
        kb_ids = kb_ids or []

        # 未绑定知识库：直接关闭 kb
        if not kb_ids:
            logger.info("[memory_vs_kb_router] no kb_ids provided, disable KB search")
            return {"kb_enabled": False}

        # 获取用户最后一个问题
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        question = human_messages[-1].content if human_messages else ""
        if not isinstance(question, str):
            question = str(question)
        q = question.strip()

        if not q:
            logger.info("[memory_vs_kb_router] empty question, default to MEMORY")
            return {"kb_enabled": False}

        # 1. 轻量关键词启发式（快速路径，避免每次都调 LLM）
        # 仅作为粗粒度启发式，用于快速分类问题类型，避免每次都调 LLM：
        # - doc_keywords：更像“文档事实型问题”
        # - personal_keywords：更像“个人/会话型问题”
        doc_keywords = ["合同", "文档", "资料", "附件", "论文", "报告", "条款", "作者", "标题"]
        personal_keywords = ["我是谁", "我喜欢", "我爱吃", "我最近", "我的爱好", "我的兴趣"]

        if any(kw in q for kw in doc_keywords):
            logger.info("[memory_vs_kb_router] heuristic: doc-like question -> enable KB")
            return {"kb_enabled": True}
        if any(kw in q for kw in personal_keywords):
            logger.info("[memory_vs_kb_router] heuristic: personal question -> MEMORY only")
            return {"kb_enabled": False}

        # 2. 回退到 LLM 决策（仅基于问题文本做分类）
        try:
            eval_model = get_model(model_name or config["configurable"].get("model", settings.DEFAULT_MODEL))  # type: ignore[index]
        except Exception:
            eval_model = get_model(settings.DEFAULT_MODEL)

        eval_input = [
            SystemMessage(content=MEMORY_VS_KB_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题: {q}"),
        ]

        try:
            eval_config = {**config, "callbacks": []} if isinstance(config, dict) else config  # type: ignore[arg-type]
            response = await eval_model.ainvoke(eval_input, eval_config, stream=False)
        except TypeError:
            # 某些客户端不支持 stream 位置参数，退化为最简调用
            response = await eval_model.ainvoke(eval_input)

        raw = response.content if isinstance(response.content, str) else str(response.content)
        raw_upper = raw.strip().upper()

        decision = "KB" if "KB" in raw_upper else "MEMORY"
        kb_enabled = decision == "KB"
        logger.info(
            "[memory_vs_kb_router] LLM decision: %s (raw=%s, kb_ids=%s)",
            decision,
            raw_upper,
            kb_ids,
        )

        return {"kb_enabled": kb_enabled}

    return router_node


def create_model_to_tools_router(
    max_tool_rounds: int = 2,
    tool_rounds_key: str = "tool_rounds",
) -> Callable[..., Literal["tools", "done", "force_done"]]:
    """
    100% 基于 State 状态机的条件边路由器。
    
    规则：
    1. 优先检查 AIMessage 是否包含 tool_calls。若无，返回 'done'。
    2. 检查评估结果。若 `retrieval_eval` 为 'sufficient'，强制拦截并返回 'done'。
    3. 检查 `state[tool_rounds_key]`。若 >= max_tool_rounds，路由至 'force_done'。
    """
    def router(state: Dict[str, Any]) -> Literal["tools", "done", "force_done"]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        
        # 核心状态读取
        current_rounds = state.get(tool_rounds_key, 0)
        retrieval_eval = state.get("retrieval_eval", "insufficient")
        
        remaining = state.get("remaining_steps", "unknown")
        logger.info(
            "[State Check][Router] tool_rounds=%s, remaining=%s, max=%s, evaluation=%s, keys=%s",
            current_rounds, remaining, max_tool_rounds, retrieval_eval, list(state.keys())
        )
        
        if not isinstance(last, AIMessage) or not last.tool_calls:
            logger.info("[Router] Decision: done (no tool_calls requested)")
            return "done"
        
        # 评估阻断逻辑
        if retrieval_eval == "sufficient":
            logger.info("[Router] Decision: done (evaluation=sufficient, bypassing requested tool_calls)")
            return "done"
            
        # 轮次熔断逻辑
        if current_rounds >= max_tool_rounds:
            logger.warning(
                "[Router] Decision: force_done (tool_rounds reached limit: %s/%s)",
                current_rounds, max_tool_rounds
            )
            return "force_done"
            
        logger.info("[Router] Decision: tools (continuing to next round)")
        return "tools"
    return router


def create_force_done_node(message: Optional[str] = None) -> Callable:
    """
    返回「轮次达上限时」追加一条系统消息的节点；message 默认使用 DEFAULT_FORCE_DONE_MESSAGE。
    """
    content = message or DEFAULT_FORCE_DONE_MESSAGE
    def _node(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=content)]}
    return _node


# 覆盖默认的 MEMORY_VS_KB_SYSTEM_PROMPT，使其更加泛化、避免过拟合具体测试问题。
# Python 模块加载时后定义会覆盖前面的同名常量。
MEMORY_VS_KB_SYSTEM_PROMPT = """你是一个路由决策助手。
你的任务是根据用户的最后一个问题，判断应该优先使用哪种信息来源：

1. MEMORY（长期记忆/对话上下文）：
   - 问题主要关于用户本人、个人偏好、日常行为或历史对话内容。

2. KB（知识库文档）：
   - 问题主要询问外部文档或知识库中的具体事实信息，
     例如合同、规章、技术文档、报告、论文、附件等中的条款、名称、金额、日期、作者、标题等。

请注意：
- 如果问题看起来两者都可能相关，但明显需要查某个文档中的具体事实，请选择 KB。
- 只有当问题主要是个人/会话相关且不依赖外部文档时，才选择 MEMORY。

输出要求：
- 只能输出一个单词：MEMORY 或 KB（全大写），不得包含任何其他内容。"""
