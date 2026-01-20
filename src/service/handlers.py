"""
请求处理器模块，处理具体的业务逻辑
"""
import inspect
import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage,HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
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

import logging
logger = logging.getLogger(__name__)


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
        # Langfuse 回调配置
        langfuse_handler = CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value() if settings.LANGFUSE_PUBLIC_KEY else None,
            secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value() if settings.LANGFUSE_SECRET_KEY else None,
            host=settings.LANGFUSE_HOST
        )

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

    try:
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

    该函数通过异步流的方式从agent图中获取响应，并其格式化为SSE(Server-Sent Events)格式返回
    它支持处理中断、任务取消、消息过滤等功能，能够实时地响应发送给客户端
    可选地使用 RAG 知识库增强回答

    Args:
        user_input (StreamInput): 用户输入对象，包含消息内容、线程ID、用户ID等信息以及流控制选项
        agent_id (str, optional): 要使用的agent ID. 默认使用DEFAULT_AGENT

    Returns:
        AsyncGenerator[str, None]: 异步生成器，产生格式化的SSE消息流

    Yields:
        str: 格式化的SSE事件数据流，包括消息、令牌和错误信息
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, task_id = await _handle_input(user_input, agent)
    #  注册 task event
    cancel_event = register_active_task(task_id)

    full_response = ""
    try:
        # 从图中处理流式事件并通过SSE流消息发送给客户端
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            # 如果stop task接口已经被调用，说明要取消当前task
            if cancel_event.is_set():
                logger.info(f"任务 {task_id} 已收到停止指令，正在关闭流")
                stop_message = _build_stop_chat_message(run_id, "当前对话已被用户手动停止")
                yield f"data: {json.dumps({'type': 'message', 'content': stop_message.model_dump()})}\n\n"
                break
            if not isinstance(stream_event, tuple):
                continue
            # 根据是否使用子图处理不同的流事件结构
            if len(stream_event) == 3:
                # 使用subgraphs=True时: (node_path, stream_mode, event)
                node_path, stream_mode, event = stream_event
                # Removed verbose logging: logger.debug(f"LangGraph事件 [SUB]: node={node_path}, mode={stream_mode}")
            else:
                # 不使用subgraphs时: (stream_mode, event)
                stream_mode, event = stream_event
                # Removed verbose logging: logger.debug(f"LangGraph事件 [TOP]: mode={stream_mode}")
            new_messages = []
            # 处理updates类型的流事件，主要包含节点更新信息
            if stream_mode == "updates":
                for node, updates in event.items():
                    # 处理agent中断的简单方法
                    # 在更复杂的实现中，我们可以添加一些结构化的ChatMessage类型来返回中断值
                    if node == "__interrupt__":
                        interrupt_val: Interrupt
                        for interrupt_val in updates:
                            new_messages.append(AIMessage(content=interrupt_val.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # 使用langgraph-supervisor库的特殊情况
                    if "supervisor" in node or "sub-agent" in node:
                        # 来自实际agent的唯二工具是handoff和handback工具
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # 如果这是子agent，我们希望保留最后2条消息 - handback工具及其结果
                                update_messages = update_messages[-2:]
                            else:
                                # 如果这是supervisor，我们只想保留最后一条消息 - handoff结果工具来自'agent'节点
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            # 处理custom类型的流事件，直接事件作为消息
            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph流可能会发出元组: (field_name, field_value)
            # 例如 ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), 等等
            # 我们只累积支持的字段到`parts`中并跳过不支持的元数据
            # 更多信息请参见: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    current_message[key] = value
                else:
                    # 首先处理累积的部分消息
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    
                    # 核心去重逻辑：避免发送与已发送内容完全一致的消息
                    if isinstance(message, AIMessage):
                        msg_content = convert_message_content_to_string(message.content)
                        if msg_content and full_response.endswith(msg_content):
                             logger.info(f"忽略重复的 AI 消息内容: {msg_content[:30]}...")
                             continue
                    
                    processed_messages.append(message)

            # 添加任何剩余的消息部分
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            # 处理消息并发送给客户端
            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"解析消息时出错: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': '意外错误'})}\n\n"
                    continue
                
                logger.debug(f"准备发送消息: type={chat_message.type}, content_len={len(chat_message.content) if chat_message.content else 0}, skip={chat_message.response_metadata.get('skip_stream')}")
                
                # LangGraph重新发送输入消息，这感觉很奇怪，所以丢弃它
                if chat_message.type == "human" and (chat_message.content or "").strip() == (user_input.message or "").strip():
                    continue
                
                # 保留我的逻辑：如果标记了跳过流式传输，则不发送
                if chat_message.response_metadata.get("skip_stream"):
                    continue
                
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            # 处理messages类型的流事件，主要用于流式传输LLM生成的令牌
            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                
                # 核心修复：不但要检查 metadata（来自节点/运行），还要检查消息本身的对象属性
                # 这能过滤掉那种从节点返回、带 skip_stream 标记的静态 AIMessage
                if "skip_stream" in metadata.get("tags", []):
                    continue
                if hasattr(msg, "response_metadata") and msg.response_metadata.get("skip_stream"):
                    continue
                    
                # 按照用户给出的原始代码，这里保留对 AIMessage 或 AIMessageChunk 的支持
                if not isinstance(msg, (AIMessageChunk, AIMessage)):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    token_content = convert_message_content_to_string(content)
                    full_response += token_content
                    # 在OpenAI的上下文中，空内容通常意味着模型要求调用工具
                    # 所以我们只打印非空内容
                    yield f"data: {json.dumps({'type': 'token', 'content': token_content}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error(f"消息生成器中发生错误: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': '内部服务器错误'})}\n\n"
    finally:
        unregister_active_task(task_id)
        # Note: 按照用户建议，在这里不做过多逻辑，只保留原始 stable 结构
        yield "data: [DONE]\n\n"


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