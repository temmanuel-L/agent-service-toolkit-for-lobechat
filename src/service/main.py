"""
主服务模块 - 组合所有路由和应用
"""
from fastapi import APIRouter, Depends, FastAPI, Header, Query
from fastapi.responses import StreamingResponse

from schema import (
    ChatHistoryInput,
    ConversationInput,
    DeleteConversationInput,
    Feedback,
    StopTaskInput,
    StreamInput,
    UserInput,
    OpenAIChatCompletionRequest,
)

from . import conversation as conversation
from . import handlers as handlers
from . import health as health_module
from .auth import verify_bearer
from .lifespan import lifespan
from .responses import _sse_response_example
from .task_manager import stop_task as task_manager_stop_task
from .openai_paradigm import chat_completions_handler


router = APIRouter(dependencies=[Depends(verify_bearer)])

app = FastAPI(
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@router.get("/info")
async def info():
    """
    获取服务信息接口

    该接口返回服务的元数据信息，包括可用的agents列表、支持的模型列表、
    默认agent和默认模型等信息

    Returns:
        ServiceMetadata: 包含服务元数据信息的响应对象
    """
    return await handlers.info_handler()


@app.get("/health")
async def health():
    """
    健康检查接口

    该接口用于检查服务的健康状态，包括基本服务状态和可选的Langfuse连接状态

    Returns:
        dict: 包含健康状态信息的字典
    """
    return await health_module.health_check()


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str | None = None):
    """
    调用指定agent处理用户输入

    该接口接收用户输入并调用指定的agent进行处理，返回处理结果
    如果未指定agent_id，则使用默认agent

    Args:
        user_input (UserInput): 用户输入对象，包含消息内容、线程ID、用户ID等信息
        agent_id (str | None, optional): 要调用的agent ID. 默认为None

    Returns:
        ChatMessage: 处理结果消息对象
    """
    from agents.agents import DEFAULT_AGENT

    if agent_id is None:
        agent_id = DEFAULT_AGENT
    return await handlers.invoke_handler(user_input, agent_id)


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str | None = None):
    """
    流式处理用户输入并返回SSE响应

    该接口接收用户输入并通过指定的agent进行流式处理，返回服务器发送事件(SSE)格式的流响应
    如果未指定agent_id，则使用默认agent

    Args:
        user_input (StreamInput): 流式输入对象，包含用户消息和流控制选项
        agent_id (str | None, optional): 要使用的agent ID. 默认为None

    Returns:
        StreamingResponse: 服务器发送事件格式的流响应对象
    """
    from agents.agents import DEFAULT_AGENT

    if agent_id is None:
        agent_id = DEFAULT_AGENT
    return await handlers.stream_handler(user_input, agent_id)


@router.post("/feedback")
async def feedback(feedback_data: Feedback):
    """
    提交用户反馈到LangSmith

    该接口接收用户反馈数据并提交到LangSmith进行记录和分析

    Args:
        feedback_data (Feedback): 包含反馈信息的对象

    Returns:
        FeedbackResponse: 表示反馈提交成功的响应对象
    """
    return await handlers.feedback_handler(feedback_data)


@router.post("/history")
async def history(input_data: ChatHistoryInput):
    """
    获取聊天历史记录

    该接口根据线程ID获取指定会话的聊天历史记录

    Args:
        input_data (ChatHistoryInput): 包含线程ID的输入对象

    Returns:
        ChatHistory: 包含聊天历史记录的响应对象
    """
    return await conversation.history_handler(input_data)


@router.post("/conversations")
async def conversations(input_data: ConversationInput):
    """
    获取用户对话列表

    该接口根据用户ID获取其所有的对话列表

    Args:
        input_data (ConversationInput): 包含用户ID的输入对象

    Returns:
        list: 用户的对话列表
    """
    return await conversation.conversations_handler(input_data)


@router.post("/delete_conversation")
async def delete_conversation(input_data: DeleteConversationInput):
    """
    删除指定对话

    该接口根据线程ID删除指定的对话及其相关数据

    Args:
        input_data (DeleteConversationInput): 包含要删除对话的线程ID的输入对象

    Returns:
        dict: 表示删除操作结果的响应对象
    """
    return await conversation.delete_conversation_handler(input_data)


@router.post("/stop_task")
async def stop_task(input_data: StopTaskInput):
    """
    停止指定任务

    该接口用于停止正在进行的任务，通过任务ID标识要停止的任务

    Args:
        input_data (StopTaskInput): 包含要停止任务ID的输入对象

    Returns:
        dict: 表示停止操作结果的响应对象
    """
    return await task_manager_stop_task(input_data)


import base64
import json


@app.post("/v1/chat/completions")
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    user_id: str = Header(None, alias="user-id"),
    thread_id: str = Header(None, alias="thread-id"),
    x_user_id: str = Header(None, alias="x-user-id"),
    x_thread_id: str = Header(None, alias="x-thread-id"),
    x_lobe_trace: str = Header(None, alias="x-lobe-trace")
):
    """
    OpenAI兼容的聊天完成接口

    该接口提供与OpenAI API兼容的聊天完成功能，支持流式响应
    从请求头获取user_id和thread_id用于数据存储和检索
    支持从X-lobe-trace头中提取sessionId作为thread_id

    Args:
        request (OpenAIChatCompletionRequest): OpenAI格式的聊天完成请求
        user_id (str, optional): 用户ID，从请求头获取
        thread_id (str, optional): 会话ID，从请求头获取
        x_user_id (str, optional): 备用用户ID，从请求头获取
        x_thread_id (str, optional): 备用会话ID，从请求头获取

    Returns:
        Any: OpenAI格式的聊天完成响应
    """
    from agents.agents import DEFAULT_AGENT
    # 优先使用特定头信息，然后是通用头信息
    effective_user_id = user_id or x_user_id or None
    
    # 从x-lobe-trace头中提取sessionId作为thread_id
    extracted_thread_id = None
    if x_lobe_trace:
        try:
            # 解码base64编码的trace信息
            decoded_trace = base64.b64decode(x_lobe_trace).decode('utf-8')
            trace_data = json.loads(decoded_trace)
            extracted_thread_id = trace_data.get('sessionId')
        except Exception:
            # 如果解析失败，忽略错误
            pass
    
    # 优先级：直接传递的thread_id > 从lobe-trace中提取的sessionId > x_thread_id
    effective_thread_id = thread_id or extracted_thread_id or x_thread_id or None
    return await chat_completions_handler(request, DEFAULT_AGENT, effective_user_id, effective_thread_id)

@app.post("/v1/chat/completions/{agent_id}")
async def openai_chat_completions_with_agent(
    agent_id: str,
    request: OpenAIChatCompletionRequest,
    user_id: str = Header(None, alias="user-id"),
    thread_id: str = Header(None, alias="thread-id"),
    x_user_id: str = Header(None, alias="x-user-id"),
    x_thread_id: str = Header(None, alias="x-thread-id"),
    x_lobe_trace: str = Header(None, alias="x-lobe-trace")
):
    """
    OpenAI兼容的聊天完成接口（指定agent）

    该接口提供与OpenAI API兼容的聊天完成功能，支持指定特定agent，支持流式响应
    从请求头获取user_id和thread_id用于数据存储和检索
    支持从X-lobe-trace头中提取sessionId作为thread_id

    Args:
        agent_id (str): 指定的agent ID
        request (OpenAIChatCompletionRequest): OpenAI格式的聊天完成请求
        user_id (str, optional): 用户ID，从请求头获取
        thread_id (str, optional): 会话ID，从请求头获取
        x_user_id (str, optional): 备用用户ID，从请求头获取
        x_thread_id (str, optional): 备用会话ID，从请求头获取

    Returns:
        Any: OpenAI格式的聊天完成响应
    """
    # 优先使用特定头信息，然后是通用头信息
    effective_user_id = user_id or x_user_id or None
    
    # 从x-lobe-trace头中提取sessionId作为thread_id
    extracted_thread_id = None
    if x_lobe_trace:
        try:
            # 解码base64编码的trace信息
            decoded_trace = base64.b64decode(x_lobe_trace).decode('utf-8')
            trace_data = json.loads(decoded_trace)
            extracted_thread_id = trace_data.get('sessionId')
        except Exception:
            # 如果解析失败，忽略错误
            pass
    
    # 优先级：直接传递的thread_id > 从lobe-trace中提取的sessionId > x_thread_id
    effective_thread_id = thread_id or extracted_thread_id or x_thread_id or None
    return await chat_completions_handler(request, agent_id, effective_user_id, effective_thread_id)


app.include_router(router)