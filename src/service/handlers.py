"""
请求处理器模块，处理具体的业务逻辑

Architecture Overview:
- 该模块是service层的核心，负责处理用户输入并生成响应
- message_generator是流式响应的核心函数，它从LangGraph接收三种事件流：
  1. updates: 节点完成时的状态更新（包含完整消息）
  2. messages: LLM生成的token流（用于流式输出）
  3. custom: 自定义事件

Stream Mode Responsibilities (流模式职责划分):
- updates模式: 仅处理interrupt等需要完整消息的场景，过滤ToolMessage
- messages模式: 处理流式token输出，遵守skip_stream标签
- custom模式: 透传自定义事件

Deduplication Strategy (去重策略):
- 使用StreamState类统一管理流状态
- 所有去重在message_generator层完成，openai_paradigm层不再去重
- updates模式发送的完整消息会检查是否已通过messages模式流式发送
"""
import inspect
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith.utils import LangSmithAuthError

from agents.agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from core import settings

from schema import (
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import convert_message_content_to_string, langchain_to_chat_message, remove_tool_calls
from .task_manager import _build_stop_chat_message, register_active_task, unregister_active_task
from utils.log_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# 工具输出过滤配置
# ============================================================================
# 这些工具的 ToolMessage 输出不应该直接展示给用户
# 原因：这些工具返回的是内部上下文（如 RAG 检索块），应由 LLM 整合后回答
# 
# 注意：不在此列表中的工具（如 WebSearch）的输出会正常展示给用户
TOOLS_WITH_HIDDEN_OUTPUT = {
    "search_knowledge",  # RAG 知识库检索工具 - 返回原始检索块，不适合直接展示
}

# 需要回退机制的工具 - 当模型返回空内容时，使用工具结果生成回退响应
# 这些通常是 RAG 类工具，当模型无法正确处理工具结果时，至少应该展示检索结果
TOOLS_WITH_FALLBACK = {
    "search_knowledge",  # RAG 工具 - 如果模型返回空，展示检索结果
}


@dataclass
class StreamState:
    """
    流状态管理器 - 统一管理消息流的去重和状态跟踪
    
    背景问题:
    ---------
    LangGraph 的 astream 方法会从多个通道（updates, messages, custom）发送相同内容的不同形式：
    - updates 通道：发送完整的 AIMessage 对象（节点完成时）
    - messages 通道：发送 token 流（AIMessageChunk，用于流式输出）
    
    如果不进行去重，前端会收到重复的消息内容。之前的解决方案是在 message_generator 和 
    openai_paradigm 两层都进行去重，导致代码复杂且难以维护。
    
    设计原则:
    ---------
    1. 单一数据源：所有已发送内容的跟踪都在这里完成，openai_paradigm 层不再去重
    2. 明确的发送类型：区分 token 流式发送和完整消息发送
    3. 简单的去重逻辑：基于内容比较和哈希的精确去重
    4. 回退机制：当模型返回空内容但有工具结果时，使用工具结果生成回退响应
    
    使用场景:
    ---------
    - messages 模式发送 token 时，调用 append_token() 记录
    - updates 模式发送完整消息前，调用 should_send_message() 检查是否需要发送
    - 发送完成后，调用 mark_message_sent() 记录
    - 工具结果到达时，调用 record_tool_result() 记录（用于回退）
    - 流结束时，调用 needs_fallback() 检查是否需要回退
    """
    # 已通过 token 流（messages 模式）发送的完整内容
    # 用于 updates 模式去重：如果完整消息已经通过 token 逐字发送过，就不再重复发送
    streamed_content: str = ""
    
    # 已发送的完整消息内容哈希集合
    # 用于防止重复发送完全相同的消息（如多次 interrupt）
    sent_message_hashes: set = field(default_factory=set)
    
    # 当前是否处于活跃的 token 流中（预留，暂未使用）
    is_streaming_tokens: bool = False
    
    # ===== 回退机制相关 =====
    # 记录需要回退的工具结果 {tool_name: content}
    # 当模型返回空内容时，可以使用这些结果生成回退响应
    pending_fallback_results: dict = field(default_factory=dict)
    
    # 是否已发送有效内容（用于判断是否需要回退）
    has_sent_content: bool = False
    
    def append_token(self, token: str) -> None:
        """
        记录已通过 messages 模式流式发送的 token
        
        Args:
            token: 刚发送的 token 字符串
        """
        self.streamed_content += token
        self.is_streaming_tokens = True
    
    def should_send_message(self, content: str) -> bool:
        """
        判断完整消息是否应该发送（用于 updates 模式）
        
        去重规则：
        1. 空内容不发送
        2. 如果内容哈希已存在（完全相同的消息发送过），跳过
        3. 如果内容已经完全通过 token 流发送过，跳过
        
        性能优化：
        - 优先使用哈希检查（O(1)），避免对长内容进行字符串比较
        - 对于流式内容的检查，使用长度和前缀快速判断
        
        Args:
            content: 要发送的消息内容
            
        Returns:
            bool: True 表示应该发送，False 表示应该跳过
        """
        if not content or not content.strip():
            return False
        
        content_stripped = content.strip()
        
        # 1. 首先使用哈希快速检查（O(1) 时间复杂度）
        content_hash = hash(content_stripped)
        if content_hash in self.sent_message_hashes:
            return False
        
        # 2. 检查是否已通过 token 流发送
        # 优化：先检查长度，避免不必要的字符串比较
        if self.streamed_content:
            streamed_stripped = self.streamed_content.strip()
            
            # 如果流式内容长度 >= 待检查内容长度，可能已经发送过
            if len(streamed_stripped) >= len(content_stripped):
                # 完全匹配检查（最常见情况）
                if content_stripped == streamed_stripped:
                    return False
                # 包含检查（内容是流式内容的一部分）
                if content_stripped in streamed_stripped:
                    return False
        
        return True
    
    def mark_message_sent(self, content: str) -> None:
        """
        标记完整消息已发送
        
        Args:
            content: 已发送的消息内容
        """
        if content and content.strip():
            self.sent_message_hashes.add(hash(content.strip()))
    
    def get_unsent_content(self, content: str) -> str:
        """
        获取未发送的内容部分（预留方法，目前未使用）
        
        用于处理 updates 模式下，部分内容已通过 messages 模式发送的情况。
        可以返回尚未发送的剩余部分。
        
        Args:
            content: 完整消息内容
            
        Returns:
            str: 未发送的内容部分
        """
        if not content:
            return ""
        
        # 如果完全没有流式发送过，返回全部内容
        if not self.streamed_content:
            return content
        
        # 如果已发送内容是新内容的前缀，返回剩余部分
        if content.startswith(self.streamed_content):
            return content[len(self.streamed_content):]
        
        # 否则检查新内容是否已发送内容的一部分
        if content.strip() in self.streamed_content:
            return ""
        
        return content
    
    # ===== 回退机制方法 =====
    
    def record_tool_result(self, tool_name: str, content: str) -> None:
        """
        记录工具结果（用于回退机制）
        
        当模型调用了工具但最终返回空内容时，我们可以使用这些工具结果
        生成一个有意义的回退响应。
        
        Args:
            tool_name: 工具名称
            content: 工具返回的内容
        """
        if tool_name in TOOLS_WITH_FALLBACK:
            self.pending_fallback_results[tool_name] = content
            logger.debug(f"记录工具结果用于回退: tool={tool_name}, len={len(content)}")
    
    def mark_content_sent(self) -> None:
        """标记已发送有效内容"""
        self.has_sent_content = True
        # 清空待回退的工具结果（已经有正常响应了，不需要回退）
        self.pending_fallback_results.clear()
    
    def needs_fallback(self) -> bool:
        """
        检查是否需要回退响应
        
        条件：没有发送任何有效内容，但有待处理的工具结果
        """
        return not self.has_sent_content and bool(self.pending_fallback_results)
    
    def get_fallback_response(self) -> str | None:
        """
        生成回退响应
        
        使用 rag.utils 模块中的通用工具格式化回退响应
        
        Returns:
            格式化的回退响应内容，如果没有待处理结果则返回 None
        """
        if not self.pending_fallback_results:
            return None
        
        # 使用 RAG 工具模块生成回退响应
        from rag.utils import format_rag_fallback_response
        
        # 目前主要处理 search_knowledge 的回退
        for tool_name, content in self.pending_fallback_results.items():
            if tool_name == "search_knowledge":
                return format_rag_fallback_response(content)
        
        # 如果有其他工具结果，返回通用消息
        return "抱歉，我执行了相关操作但无法生成完整响应。请尝试重新提问。"


async def _handle_input(
    user_input: UserInput, agent: AgentGraph
) -> tuple[dict[str, Any], UUID, str]:
    """
    处理用户输入，准备agent执行所需的各种配置参数

    该函数负责解析用户输入、设置运行配置、处理中断恢复等操作，
    最终返回一个包含执行参数、运行ID和任务ID的元组

    Args:
        user_input (UserInput): 用户输入对象，包含消息内容、线程ID、用户ID等信息
        agent (AgentGraph): 要执行的agent图对象

    Raises:
        RuntimeError: 当启用LANGFUSE_TRACING但缺少langfuse包时抛出
        HTTPException: 当agent_config包含保留关键字时抛出，状态码为422

    Returns:
        tuple[dict[str, Any], UUID, str]: 包含以下三个元素的元组：
            - 执行参数字典，包含输入数据和配置信息
            - 运行唯一标识符(UUID)
            - 任务ID字符串，格式为"{user_id}_{thread_id}"
    """
    # 生成运行ID并获取线程ID和用户ID
    run_id = uuid4()
    thread_id = user_input.thread_id or str(run_id)
    user_id = user_input.user_id or "user"

    # 构建基础配置参数
    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        configurable["model"] = user_input.model

    # 初始化回调列表
    callbacks = []
    # 如果启用了Langfuse跟踪，则添加Langfuse回调处理器
    if settings.LANGFUSE_TRACING:
        # Langfuse 回调配置 (全局客户端已在 lifespan.py 中初始化)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    # 处理用户提供的额外配置参数
    if user_input.agent_config:
        # 检查是否包含保留关键字（包括即使不在configurable中的'model'）
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    # 添加元数据，用于后续的数据检索和筛选
    metadata = {"user_id": user_id}
    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
        metadata=metadata,
    )

    # 检查是否有需要恢复的中断任务
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    # 根据是否存在中断任务决定输入数据的格式
    if interrupted_tasks:
        # 假设用户输入是用于从中断处恢复agent执行的响应
        input_data: Command[Any] | dict[str, Any] = Command(resume=user_input.message)
    else:
        # 正常情况下将用户消息包装成HumanMessage
        input_data = {"messages": [HumanMessage(content=user_input.message)]}

    # 构建执行参数
    kwargs = {
        "input": input_data,
        "config": config,
    }

    # 构建任务ID
    task_id = f"{user_id}_{thread_id}"
    return kwargs, run_id, task_id


async def invoke_handler(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    调用指定的agent处理用户输入并返回响应结果

    该函数会根据用户输入调用对应的agent进行处理，并根据处理结果构建响应消息
    支持处理正常完成和被中断两种情况，并能处理任务取消的情况

    Args:
        user_input (UserInput): 用户输入对象，包含消息内容、线程ID、用户ID等信息
        agent_id (str, optional): 要调用的agent ID. 默认使用DEFAULT_AGENT

    Raises:
        ValueError: 当遇到未预期的响应类型时抛出
        HTTPException: 当处理过程中发生异常时抛出，状态码为500

    Returns:
        ChatMessage: 处理结果消息对象，可能为正常响应或中断响应
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, task_id = await _handle_input(user_input, agent)
    # 注册 task event
    cancel_event = register_active_task(task_id)

    # 从 config 中提取 user_id 和 thread_id 用于跟踪
    configurable = kwargs["config"]["configurable"]
    user_id = configurable.get("user_id")
    thread_id = configurable.get("thread_id")

    try:
        # 使用 propagate_attributes 追踪用户和会话 (Langfuse 最佳实践)
        with propagate_attributes(user_id=user_id, session_id=thread_id):
            response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        # 如果stop task 接口已经调用，说明要取消当前task
        if cancel_event.is_set():
            logger.info(f"任务 {task_id} 在响应发送前就停止了")
            return _build_stop_chat_message(run_id, "当前对话已被用户停止")
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error")
    finally:
        unregister_active_task(task_id)


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    流式生成聊天消息响应

    该函数通过异步流的方式从agent图中获取响应，并将其格式化为SSE(Server-Sent Events)格式返回。
    它支持处理中断、任务取消、消息过滤等功能，能够实时地将响应发送给客户端。
    可选地使用 RAG 知识库增强回答。

    这是 /stream 端点的核心工作函数。

    架构说明 (Architecture):
    -------------------------
    本函数从 LangGraph agent 接收三种类型的事件流：
    
    1. updates 模式：节点完成时的状态更新
       - 包含完整的消息对象
       - 用于处理 interrupt（中断/人机协同）
       - 核心改进：过滤 ToolMessage，防止工具执行结果（如RAG检索内容）泄露给前端
       
    2. messages 模式：LLM 生成的 token 流
       - 用于实现流式输出效果
       - 遵守 skip_stream 标签（带此标签的节点不进行流式输出）
       
    3. custom 模式：自定义事件
       - 透传自定义事件给前端

    去重策略 (Deduplication Strategy):
    ----------------------------------
    使用 StreamState 类统一管理流状态，解决以下问题：
    - updates 模式和 messages 模式可能发送相同内容的不同形式
    - 避免前端收到重复消息
    - 所有去重在本层完成，openai_paradigm 适配层不再进行去重

    Args:
        user_input (StreamInput): 用户输入对象，包含消息内容、线程ID、用户ID等信息以及流控制选项
        agent_id (str, optional): 要使用的agent ID. 默认使用DEFAULT_AGENT

    Returns:
        AsyncGenerator[str, None]: 异步生成器，产生格式化的SSE消息流

    Yields:
        str: 格式化的SSE事件数据流，包括：
            - type: 'token' - LLM生成的token片段
            - type: 'message' - 完整消息（interrupt、最终回复等）
            - type: 'error' - 错误信息
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, task_id = await _handle_input(user_input, agent)
    
    # 注册 task event，用于支持任务取消功能
    cancel_event = register_active_task(task_id)

    # 从 config 中提取 user_id 和 thread_id 用于 Langfuse 跟踪
    configurable = kwargs["config"]["configurable"]
    user_id = configurable.get("user_id")
    thread_id = configurable.get("thread_id")

    # 使用 StreamState 统一管理流状态（去重账本）
    # 这是解决 updates/messages 双通道重复发送问题的核心
    stream_state = StreamState()

    try:
        # 从图中处理流式事件。使用 propagate_attributes 追踪用户和会话 (Langfuse 最佳实践)
        with propagate_attributes(user_id=user_id, session_id=thread_id):
            async for stream_event in agent.astream(
                **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
            ):
                # 如果 stop_task 接口已经被调用，说明要取消当前 task
                if cancel_event.is_set():
                    logger.info(f"任务 {task_id} 已收到停止指令，正在关闭流")
                    stop_message = _build_stop_chat_message(run_id, "当前对话已被用户手动停止")
                    yield f"data: {json.dumps({'type': 'message', 'content': stop_message.model_dump()})}\n\n"
                    break

                if not isinstance(stream_event, tuple):
                    continue

                # 根据是否使用子图处理不同的流事件结构
                if len(stream_event) == 3:
                    # 使用 subgraphs=True 时: (node_path, stream_mode, event)
                    node_path, stream_mode, event = stream_event
                else:
                    # 不使用 subgraphs 时: (stream_mode, event)
                    stream_mode, event = stream_event

                # ==================== 1. MESSAGES 模式：Token 流式输出 ====================
                # 优先处理 messages 模式，因为它是最频繁的事件类型
                # 更多信息请参见: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
                if stream_mode == "messages":
                    # 如果用户不需要流式 token，跳过
                    if not user_input.stream_tokens:
                        continue
                    
                    msg, metadata = event
                    
                    # DEBUG: 记录收到的消息类型
                    logger.debug(f"[MESSAGES] 收到消息: type={type(msg).__name__}, has_tool_calls={hasattr(msg, 'tool_calls') and bool(msg.tool_calls)}, content_len={len(msg.content) if hasattr(msg, 'content') and msg.content else 0}")
                    
                    # 核心机制：检查 skip_stream 标签
                    # 带有此标签的节点（如信息抽取节点）不进行流式输出
                    # 这是 simple_travel_planner 等智能体控制中间结果不输出的关键
                    if "skip_stream" in metadata.get("tags", []):
                        continue
                    
                    # 由于某些原因，astream("messages") 会导致非 LLM 节点发送额外消息
                    # 只处理 AI 消息块，丢弃其他类型
                    if not isinstance(msg, (AIMessageChunk, AIMessage)):
                        continue
                    
                    # 移除工具调用内容
                    # 目前只有 Anthropic 模型会流式发送工具调用，使用 content item type tool_use
                    content = remove_tool_calls(msg.content)
                    if not content:
                        continue
                    
                    chunk_str = convert_message_content_to_string(content)
                    if not chunk_str:
                        continue
                    
                    # 增量提取：处理累积流模式 (A -> AB -> ABC)
                    # 有些模型返回的是累积内容而非增量，需要提取真正的增量部分
                    current_streamed = stream_state.streamed_content
                    if current_streamed and chunk_str.startswith(current_streamed):
                        delta = chunk_str[len(current_streamed):]
                    else:
                        delta = chunk_str
                    
                    # 空内容在 OpenAI 上下文中通常意味着模型正在请求调用工具
                    # 所以我们只输出非空内容
                    if delta:
                        stream_state.append_token(delta)
                        stream_state.mark_content_sent()  # 标记已发送内容
                        yield f"data: {json.dumps({'type': 'token', 'content': delta}, ensure_ascii=False)}\n\n"
                    continue

                # ==================== 2. UPDATES 模式：节点状态更新 ====================
                # 处理 updates 类型的流事件，主要包含节点更新信息
                if stream_mode == "updates":
                    messages_to_send = []
                    
                    # DEBUG: 记录 updates 事件的节点
                    logger.debug(f"[UPDATES] 收到更新事件: nodes={list(event.keys())}")
                    
                    for node, updates in event.items():
                        # 处理 agent 中断的简单方法
                        # 在更复杂的实现中，我们可以添加一些结构化的 ChatMessage 类型来返回中断值
                        # 中断内容通常为非流式静态文本，需要作为完整消息发送
                        if node == "__interrupt__":
                            for interrupt in updates:
                                val = interrupt.value if hasattr(interrupt, 'value') else str(interrupt)
                                messages_to_send.append(AIMessage(content=val))
                            continue
                        
                        updates = updates or {}
                        update_messages = updates.get("messages", [])
                        
                        # 使用 langgraph-supervisor 库的特殊情况处理
                        if "supervisor" in node or "sub-agent" in node:
                            # 来自实际 agent 的唯一工具是 handoff 和 handback 工具
                            if update_messages and isinstance(update_messages[-1], ToolMessage):
                                if "sub-agent" in node and len(update_messages) > 1:
                                    # 如果这是子 agent，我们希望保留最后2条消息 - handback 工具及其结果
                                    update_messages = update_messages[-2:]
                                else:
                                    # 如果这是 supervisor，我们只想保留最后一条消息 - handoff 结果
                                    # 工具来自 'agent' 节点
                                    update_messages = [update_messages[-1]]
                            else:
                                update_messages = []
                        
                        # 过滤消息
                        for msg in update_messages:
                            # DEBUG: 记录每条消息的详细信息
                            msg_type = type(msg).__name__
                            msg_content_len = len(msg.content) if hasattr(msg, 'content') and msg.content else 0
                            has_tool_calls = hasattr(msg, 'tool_calls') and bool(msg.tool_calls)
                            logger.debug(f"[UPDATES] 处理消息: node={node}, type={msg_type}, content_len={msg_content_len}, has_tool_calls={has_tool_calls}")
                            
                            # 选择性过滤 ToolMessage
                            # 只有在 TOOLS_WITH_HIDDEN_OUTPUT 列表中的工具输出才会被过滤
                            # 其他工具（如 WebSearch）的输出会正常展示给用户
                            if isinstance(msg, ToolMessage):
                                tool_name = getattr(msg, 'name', None)
                                tool_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                                
                                if tool_name and tool_name in TOOLS_WITH_HIDDEN_OUTPUT:
                                    logger.debug(f"过滤工具 '{tool_name}' 的输出（内部上下文）: tool_call_id={msg.tool_call_id}")
                                    # 记录工具结果用于回退机制
                                    stream_state.record_tool_result(tool_name, tool_content)
                                    continue
                                # 其他工具的 ToolMessage 正常展示
                                logger.debug(f"展示工具 '{tool_name}' 的输出: tool_call_id={msg.tool_call_id}")
                            messages_to_send.append(msg)
                    
                    # LangGraph 流可能会发出元组: (field_name, field_value)
                    # 例如 ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), 等等
                    # 我们只累积支持的字段到 `parts` 中并跳过不支持的元数据
                    # 处理收集的消息，将元组格式转换为 AIMessage
                    for message in _process_raw_messages(messages_to_send):
                        chat_message = _safe_convert_to_chat_message(message, run_id)
                        if chat_message is None:
                            continue
                        
                        # LangGraph 会重新发送输入消息，这感觉很奇怪，所以丢弃它
                        if chat_message.type == "human" and (chat_message.content or "").strip() == (user_input.message or "").strip():
                            continue
                        
                        # 保留逻辑：如果标记了跳过流式传输，则不发送
                        if chat_message.response_metadata.get("skip_stream"):
                            continue
                        
                        # --- 去重判定 (Deduplication Logic) ---
                        # AI 消息去重：检查是否已通过 token 流（messages 模式）发送过
                        if chat_message.type == "ai":
                            content = chat_message.content or ""
                            has_tool_calls = bool(chat_message.tool_calls)
                            logger.debug(f"[UPDATES] AI消息详情: content_len={len(content)}, has_tool_calls={has_tool_calls}, tool_calls={[tc.get('name') for tc in chat_message.tool_calls] if chat_message.tool_calls else []}")
                            
                            if not has_tool_calls:
                                # 如果内容已部分或全部通过 messages 通道(token)发过了，则此块忽略
                                if not stream_state.should_send_message(content):
                                    logger.debug(f"跳过已发送的 AI 消息: len={len(content)}")
                                    continue
                                # 如果是新内容（如中断提示），正常发送并更新账本
                                stream_state.mark_message_sent(content)
                                if content.strip():  # 有实际内容时标记
                                    stream_state.mark_content_sent()
                        
                        yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"
                    continue

                # ==================== 3. CUSTOM 模式：自定义事件 ====================
                # 处理 custom 类型的流事件，直接将事件作为消息透传
                if stream_mode == "custom":
                    for message in _process_raw_messages([event]):
                        chat_message = _safe_convert_to_chat_message(message, run_id)
                        if chat_message:
                            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

    except Exception as e:
        logger.error(f"生成器崩溃: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        # ===== 回退机制：检查是否需要生成回退响应 =====
        # 当模型返回空内容但有工具结果时，生成回退响应
        if stream_state.needs_fallback():
            logger.warning(f"检测到模型返回空内容，启用回退机制。待处理工具: {list(stream_state.pending_fallback_results.keys())}")
            fallback_content = stream_state.get_fallback_response()
            if fallback_content:
                # 构造一个 ChatMessage 格式的回退响应
                fallback_message = {
                    "type": "ai",
                    "content": fallback_content,
                    "tool_calls": [],
                    "run_id": str(run_id),
                    "response_metadata": {"fallback": True}
                }
                logger.info(f"发送回退响应: len={len(fallback_content)}")
                yield f"data: {json.dumps({'type': 'message', 'content': fallback_message})}\n\n"
        
        unregister_active_task(task_id)
        yield "data: [DONE]\n\n"


def _process_raw_messages(raw_messages: list) -> list:
    """
    处理原始消息列表，将元组格式的消息转换为 AIMessage
    
    LangGraph 流可能会发出元组格式的消息部分: (field_name, field_value)
    例如:
        - ('content', <str>) - 消息内容
        - ('tool_calls', [ToolCall,...]) - 工具调用列表
        - ('additional_kwargs', {...}) - 额外参数
        
    本函数负责将这些分散的部分累积并组装成完整的 AIMessage 对象。
    
    更多信息请参见: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
    
    Args:
        raw_messages: 原始消息列表，可能包含完整消息对象或元组格式的消息部分
        
    Returns:
        list: 处理后的消息列表，所有元组已转换为 AIMessage 对象
    """
    processed = []
    current_parts: dict[str, Any] = {}
    
    for message in raw_messages:
        if isinstance(message, tuple):
            # 累积消息部分
            key, value = message
            current_parts[key] = value
        else:
            # 遇到完整消息，先处理之前累积的部分
            if current_parts:
                processed.append(_create_ai_message(current_parts))
                current_parts = {}
            processed.append(message)
    
    # 处理剩余的累积部分
    if current_parts:
        processed.append(_create_ai_message(current_parts))
    
    return processed


def _safe_convert_to_chat_message(message: Any, run_id: UUID) -> ChatMessage | None:
    """
    安全地将 LangChain 消息转换为 ChatMessage
    
    如果转换失败，记录错误并返回 None，而不是抛出异常。
    这确保了单个消息的解析失败不会中断整个流。
    
    Args:
        message: LangChain 消息对象（AIMessage, HumanMessage, ToolMessage 等）
        run_id: 运行唯一标识符
        
    Returns:
        ChatMessage | None: 转换后的 ChatMessage，失败时返回 None
    """
    try:
        chat_message = langchain_to_chat_message(message)
        chat_message.run_id = str(run_id)
        return chat_message
    except Exception as e:
        logger.error(f"解析消息时出错: {e}")
        return None


def _create_ai_message(parts: dict) -> AIMessage:
    """
    根据提供的部分创建AIMessage对象

    该函数通过检查AIMessage构造函数的有效参数来过滤输入字典，
    然后使用过滤后的参数创建一个新的AIMessage实例

    Args:
        parts (dict): 包含消息各个部分的字典，键为参数名，值为对应值

    Returns:
        AIMessage: 使用有效参数创建的AIMessage对象
    """
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


async def stream_handler(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """
    处理流式请求并返回SSE响应

    该函数作为流式请求的入口点，接收用户输入和agent ID，
    并返回一个服务器发送事件（SSE）格式的流响应

    Args:
        user_input (StreamInput): 流式输入对象，包含用户消息和流控制选项
        agent_id (str, optional): 要使用的agent ID. 默认使用DEFAULT_AGENT

    Returns:
        StreamingResponse: 服务器发送事件格式的流响应对象
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


async def info_handler() -> ServiceMetadata:
    """
    获取服务元数据信息

    该函数收集并返回服务的相关元数据信息，包括可用的agents、
    支持的模型列表、默认agent和默认模型等信息

    Returns:
        ServiceMetadata: 包含服务元数据信息的对象
    """
    model_names = list(settings.AVAILABLE_MODELS)
    model_names.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=model_names,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def feedback_handler(feedback: Feedback) -> FeedbackResponse:
    """
    处理并提交用户反馈到LangSmith

    该函数是一个简单的LangSmith create_feedback API封装，
    使得可以在服务端存储和管理凭证，而不是在客户端处理
    参见: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post

    Args:
        feedback (Feedback): 包含反馈信息的对象，包括run_id、key、score等

    Raises:
        HTTPException: 当LangSmith认证失败时，状态码为502
        HTTPException: 当提交反馈到LangSmith发生未知错误时，状态码为502

    Returns:
        FeedbackResponse: 表示反馈提交成功的响应对象
    """
    # import os
    
    # api_key = (
    #     settings.LANGCHAIN_API_KEY.get_secret_value()
    #     if settings.LANGCHAIN_API_KEY
    #     else os.getenv("LANGSMITH_API_KEY")
    # )
    # if not api_key:
    #     logger.info("未配置LangSmith API密钥，尝试使用默认客户端初始化")
    #     client = LangsmithClient()
    # else:
    #     client = LangsmithClient(api_key=api_key)
    # kwargs = feedback.kwargs or {}
    # try:
    #     client.create_feedback(
    #         run_id=feedback.run_id,
    #         key=feedback.key,
    #         score=feedback.score,
    #         **kwargs,
    #     )
    # except LangSmithAuthError as exc:
    #     logger.warning("LangSmith认证失败: %s", exc)
    #     raise HTTPException(
    #         status_code=status.HTTP_502_BAD_GATEWAY,
    #         detail="LangSmith认证失败",
    #     ) from exc
    # except Exception as exc:  # pragma: no cover - defensive logging
    #     logger.error("提交反馈时发生未知错误: %s", exc)
    #     raise HTTPException(
    #         status_code=status.HTTP_502_BAD_GATEWAY,
    #         detail="无法向LangSmith提交反馈",
    #     ) from exc

    # return FeedbackResponse()

    # 基于langfuse
    if not settings.LANGFUSE_TRACING:
        raise HTTPException(status_code=400, detail="Langfuse tracing is not enabled")

    langfuse = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value(),
        secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value(),
        host=settings.LANGFUSE_HOST,
    )
    langfuse.create_score(
        trace_id=feedback.run_id,  # Langfuse 用 trace_id，不是 run_id（但通常 run_id == trace_id）
        name=feedback.key,
        value=feedback.score,
        comment=feedback.kwargs.get("comment") if feedback.kwargs else None,
    )
    langfuse.flush()  # 确保同步发送（可选，但 feedback 建议 flush）
    return FeedbackResponse()