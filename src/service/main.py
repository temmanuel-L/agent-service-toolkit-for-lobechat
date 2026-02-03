"""
主服务模块 - 组合所有路由和应用
"""
import base64
import json
from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request
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
from utils.log_utils import get_logger
from rag.service import rag_service

logger = get_logger(__name__)


router = APIRouter(dependencies=[Depends(verify_bearer)])

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
    # docs_url=None,
    # redoc_url=None,
    # openapi_url=None,
)

from core import settings

# 挂载静态文件目录
try:
    if not settings.STATIC_DIR.exists():
        settings.STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(settings.STATIC_URL, StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
except Exception as e:
    logger.warning(f"Failed to mount static directory: {e}")


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


@router.post("/api/kb/ingest")
async def kb_ingest(request: Request):
    """
    知识库文件入库接口
    接收来自 LobeChat 的文件 URL 和知识库 ID，执行异步向量化入库
    """
    try:
        body = await request.json()
        file_url = body.get("file_url")
        kb_id = body.get("kb_id")
        file_name = body.get("file_name")

        if not file_url or not kb_id:
            return {"status": "error", "message": "Missing file_url or kb_id"}

        logger.info(f"Received KB ingest request: kb_id={kb_id}, file={file_name}")
        
        # 执行入库 (由于 LobeChat 前端可能是 fire-and-forget，我们同步等待还是异步取决于需求)
        # 这里先由于涉及文件下载和处理，我们直接 await 并在完成后返回
        chunks_count = await rag_service.ingest_file(
            file_url=file_url,
            kb_id=kb_id,
            file_name=file_name
        )
        
        return {
            "status": "success",
            "kb_id": kb_id,
            "chunks_count": chunks_count
        }
    except Exception as e:
        logger.error(f"KB Ingest API Error: {str(e)}")
        return {"status": "error", "message": str(e)}


@router.post("/api/kb/delete")
async def kb_delete(request: Request):
    """
    知识库删除接口
    删除 Qdrant 中对应的 Collection
    """
    try:
        body = await request.json()
        kb_id = body.get("kb_id")

        if not kb_id:
            return {"status": "error", "message": "Missing kb_id"}

        logger.info(f"Received KB delete request: kb_id={kb_id}")
        
        success = await rag_service.delete_knowledge_base(kb_id)
        
        if success:
            return {"status": "success", "kb_id": kb_id}
        else:
            return {"status": "error", "message": f"Failed to delete KB or KB not found: {kb_id}"}
    except Exception as e:
        logger.error(f"KB Delete API Error: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/v1/chat/completions")
async def openai_chat_completions(
        request: OpenAIChatCompletionRequest,
        raw_request: Request
):
    """
    OpenAI兼容的聊天完成接口

    该接口提供与OpenAI API兼容的聊天完成功能，支持流式响应
    从请求头获取user_id和thread_id用于数据存储和检索
    支持从X-lobe-trace头中提取sessionId作为thread_id

    Args:
        request (OpenAIChatCompletionRequest): OpenAI格式的聊天完成请求
        raw_request (Request): 原始请求

    Returns:
        Any: OpenAI格式的聊天完成响应
    """

    # ======== 详细记录所有请求信息 ========
    logger.info("=== 请求开始 ===")

    # 记录完整的请求体
    logger.info("=== 完整请求体 ===")
    logger.info(json.dumps({
        "model": request.model,
        "stream": getattr(request, 'stream', None),
        "temperature": getattr(request, 'temperature', None),
        "top_p": getattr(request, 'top_p', None),
        "presence_penalty": getattr(request, 'presence_penalty', None),
        "frequency_penalty": getattr(request, 'frequency_penalty', None),
        "messages_count": len(request.messages) if request.messages else 0,
        "user": request.user,
        "topicId": getattr(request, 'topicId', None),
        "kb_ids": getattr(request, 'kb_ids', None),  # 新增：记录 kb_ids
        "messages": [
            {"role": msg.get("role"), "content": str(msg.get("content", ""))[:200]}
            for msg in (request.messages or [])[-5:]  # 只打印前5条消息的前200字符
        ]
    }, ensure_ascii=False, indent=2))
    logger.info("=== 完整请求体结束 ===")

    # 记录完整的请求头
    logger.info("=== 完整请求头 ===")
    for header_name, header_value in raw_request.headers.items():
        logger.info(f"Header: {header_name} = {header_value}")
    logger.info("=== 完整请求头结束 ===")

    # ======== 从各种来源获取 thread_id ========
    logger.info("=== 获取 thread_id 的过程 ===")

    # 1. 从请求体中获取 topicId（主要来源）
    topicId_from_body = getattr(request, 'topicId', None)
    if topicId_from_body:
        logger.info(f"✅ 从请求体获取 topicId: {topicId_from_body}")

    # 2. 从 X-lobe-trace 头部解析 topicId（备用方案）
    x_lobe_trace = raw_request.headers.get('x-lobe-trace')
    extracted_topic_id = None
    if x_lobe_trace:
        try:
            # 解码base64编码的trace信息
            decoded_trace = base64.b64decode(x_lobe_trace).decode('utf-8')
            trace_data = json.loads(decoded_trace)
            extracted_topic_id = trace_data.get('topicId')
            logger.info(f"✅ 从 X-lobe-trace 解析出 topicId: {extracted_topic_id}")
        except Exception as e:
            logger.warning(f"⚠️ 解析 X-lobe-trace 失败: {str(e)}")
            pass

    # 3. 从 x-thread-id 头部获取 thread_id（备用方案）
    x_thread_id = raw_request.headers.get('x-thread-id')
    if x_thread_id:
        logger.info(f"✅ 从 x-thread-id 头部获取: {x_thread_id}")

    # 4. 从 user_id 头部获取（作为备用）
    user_id_header = raw_request.headers.get('user-id')
    if user_id_header:
        logger.info(f"✅ 从 user-id 头部获取: {user_id_header}")

    # 5. 从 x-user-id 头部获取（作为备用）
    x_user_id = raw_request.headers.get('x-user-id')
    if x_user_id:
        logger.info(f"✅ 从 x-user-id 头部获取: {x_user_id}")

    # ======== 确定最终的 thread_id ========
    # 优先级：请求体中的 topicId > x-thread-id > 从 X-lobe-trace 解析的 topicId > user-id > x-user-id
    effective_thread_id = topicId_from_body or x_thread_id or extracted_topic_id or user_id_header or x_user_id or None

    if effective_thread_id:
        logger.info(f"✅ 最终确定的 thread_id: {effective_thread_id}")
    else:
        logger.warning("⚠️ 未找到有效的 thread_id，将使用 None")

    logger.info("=== 获取 thread_id 的过程结束 ===")

    # ======== 确定 user_id ========
    # 根据你的建议，user_id 直接使用 request.user
    effective_user_id = request.user if hasattr(request, 'user') and request.user else None

    if effective_user_id:
        logger.info(f"✅ 最终确定的 user_id: {effective_user_id}")
    else:
        logger.warning("⚠️ 未找到有效的 user_id，将使用 None")

    # ======== 调用处理函数 ========
    logger.info("=== 调用 chat_completions_handler ===")
    logger.info(
        f"参数: request={type(request)}, agent_id={request.model}, user_id={effective_user_id}, thread_id={effective_thread_id}")

    # 将 request.model 作为 agent_id 传递
    agent_id = request.model if request.model else None

    result = await chat_completions_handler(request, agent_id, effective_user_id, effective_thread_id)

    logger.info("=== chat_completions_handler 调用完成 ===")
    logger.info("=== 请求结束 ===")

    return result


# @app.post("/v1/chat/completions")
# async def openai_chat_completions(
#     request: OpenAIChatCompletionRequest,
#     raw_request: Request,
# ):
#     """
#     OpenAI兼容的聊天完成接口
#     该接口提供与OpenAI API兼容的聊天完成功能，支持流式响应
#
#     Args:
#         request (OpenAIChatCompletionRequest): OpenAI格式的聊天完成请求
#         raw_request: Request,
#
#     Returns:
#         Any: OpenAI格式的聊天完成响应
#     """
#     logger.info(f'所有request信息为: {request}')
#
#     # 1. 检查 URL 参数
#     query_params = dict(raw_request.query_params)
#     logger.info("URL query params: %s", query_params)
#
#     # 2. 检查所有请求头
#     all_headers = dict(raw_request.headers)
#     logger.info("All headers: %s",
#                 {k: v for k, v in all_headers.items() if k.lower() not in ['authorization', 'content-length']})
#
#     # 3. 检查请求体的原始 JSON（看是否有扩展字段）
#     try:
#         body_bytes = await raw_request.body()
#         body_str = body_bytes.decode('utf-8')
#         body_dict = json.loads(body_str)
#         logger.info("Raw request body keys: %s", list(body_dict.keys()))
#         # 检查是否有非标准字段（比如 session_id, conversation_id 等）
#         non_standard_keys = [k for k in body_dict.keys() if k not in [
#             'model', 'messages', 'temperature', 'top_p', 'n', 'stream',
#             'max_tokens', 'stop', 'presence_penalty', 'frequency_penalty', 'user'
#         ]]
#         if non_standard_keys:
#             logger.info("Non-standard keys in request body: %s", non_standard_keys)
#             for key in non_standard_keys:
#                 logger.info("  %s = %s", key, body_dict[key])
#     except Exception as e:
#         logger.warning("Failed to parse raw request body: %s", e)
#
#     # 4. 检查 messages 数组的元数据（有些实现会在消息对象中添加元数据）
#     if request.messages:
#         first_msg = request.messages[0]
#         logger.info("First message keys: %s", list(first_msg.keys()) if isinstance(first_msg, dict) else "not a dict")
#         # 检查是否有 id, session_id, conversation_id 等字段
#         if isinstance(first_msg, dict):
#             metadata_keys = [k for k in first_msg.keys() if k not in ['role', 'content']]
#             if metadata_keys:
#                 logger.info("Metadata keys in first message: %s", metadata_keys)
#
#     # 临时返回，先不处理，等我们看完日志再决定
#     user_id = request.user or "anonymous"
#     agent_id = request.model if request.model else None
#
#     return await chat_completions_handler(
#         request,
#         agent_id=agent_id,
#         user_id=user_id,
#         thread_id=None  # 暂时不传，先看日志
#     )
#
#     # agent_id = request.model if request.model else None
#     # # LobeChat 通过 request.body.user 提供会话 ID
#     # thread_id = getattr(request, 'user', None)
#     # # user_id 可设为匿名（因 LobeChat 不传真实用户 ID）
#     # user_id = "anonymous"  # 或从 auth token 解析（如果你有）
#     #
#     # logger.info("Using thread_id from request.user: %s", thread_id)
#     #
#     # return await chat_completions_handler(
#     #     request,
#     #     agent_id=agent_id,
#     #     user_id=user_id,
#     #     thread_id=thread_id
#     # )


app.include_router(router)