"""
OpenAI API 范式适配层
该模块提供与 OpenAI API 兼容的接口实现，用于与 LobeChat 等前端集成
"""
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from schema import (OpenAIChatCompletionRequest, StreamInput,
                    OpenAIChatMessage, OpenAIChoice, OpenAIChatCompletionResponse,
                    OpenAIChatStreamDelta, OpenAIStreamChoice, OpenAIChatCompletionStreamResponse)
from . import handlers as service_handlers
from utils.log_utils import get_logger

logger = get_logger(__name__)


async def chat_completions_handler(
    request: OpenAIChatCompletionRequest,
    agent_id: str = None,
    user_id: str = None,
    thread_id: str = None
) -> StreamingResponse | OpenAIChatCompletionResponse:
    """
    OpenAI兼容的聊天完成接口处理器
    """
    if request.messages:
        last_message = request.messages[-1]
        user_message = last_message.get("content", "")
        user_id = user_id or request.user or "default_user"
    else:
        user_message = ""
        user_id = user_id or request.user or "default_user"

    stream_input = StreamInput(
        message=user_message,
        user_id=user_id,
        thread_id=thread_id,
        model=None,
        stream_tokens=request.stream,
        agent_config={
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop
        }
    )

    if request.stream:
        async def openai_stream_generator():
            role_sent = False
            # 记录当前 AI 消息已经发送的 token 数量
            tokens_sent_for_current_msg = 0
            # 为整个流生成统一的 ID
            request_id = f"chatcmpl-{uuid.uuid4()}"

            logger.info(f"开启 OpenAI 兼容流式响应: agent={agent_id}, thread={thread_id}")

            async for chunk in service_handlers.message_generator(stream_input, agent_id):
                if not chunk.startswith("data: "):
                    continue

                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    logger.debug("接收到内部 [DONE] 信号")
                    finish_chunk = {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model or (agent_id or "default-model"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                    yield f"data: [DONE]\n\n"
                    continue

                try:
                    internal_data = json.loads(data_str)
                    data_type = internal_data.get("type")
                    
                    # 1. 处理令牌 (Tokens)
                    if data_type == "token":
                        content = internal_data.get("content")
                        if not content or content == "None":
                            continue

                        delta = {"content": content}
                        if not role_sent:
                            delta["role"] = "assistant"
                            role_sent = True
                        
                        # 计数已发送的字符数
                        tokens_sent_for_current_msg += len(content)
                        
                        openai_chunk = OpenAIChatCompletionStreamResponse(
                            id=request_id,
                            created=int(time.time()),
                            model=request.model or (agent_id or "default-model"),
                            choices=[OpenAIStreamChoice(index=0, delta=OpenAIChatStreamDelta(**delta), finish_reason=None)]
                        )
                        # Pydantic model_dump_json doesn't support ensure_ascii in some versions
                        # Use model_dump then json.dumps for maximum compatibility
                        json_data = json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)
                        yield f"data: {json_data}\n\n"

                    elif data_type == "message":
                        chat_msg = internal_data.get("content", {})
                        msg_type = chat_msg.get("type")
                        delta_dict = {}
                        
                        if msg_type == "ai":
                            content = chat_msg.get("content", "")
                            logger.info(f"适配器收到 AI 消息事件: len={len(content)}, tokens_sent={tokens_sent_for_current_msg}")
                            
                            # 只有当有新内容需要补发时才发送
                            # 如果 tokens_sent_for_current_msg 等于 content 长度，说明已经完全发送过了
                            if content and len(content) > tokens_sent_for_current_msg:
                                remaining_content = content[tokens_sent_for_current_msg:]
                                
                                # 额外检查：如果剩余内容非空才发送
                                if remaining_content.strip():
                                    logger.info(f"补发 AI 剩余内容: {remaining_content[:50]}... (剩余长度: {len(remaining_content)})")
                                    
                                    delta_dict = {"content": remaining_content}
                                    if not role_sent:
                                        delta_dict["role"] = "assistant"
                                        role_sent = True
                                        
                                    openai_chunk = OpenAIChatCompletionStreamResponse(
                                        id=request_id,
                                        created=int(time.time()),
                                        model=request.model or (agent_id or "default-model"),
                                        choices=[
                                            OpenAIStreamChoice(
                                                index=0,
                                                delta=OpenAIChatStreamDelta(**delta_dict),
                                                finish_reason=None
                                            )
                                        ]
                                    )
                                    json_data = json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)
                                    yield f"data: {json_data}\n\n"
                            
                            # 重置计数器，为下一条消息做准备
                            tokens_sent_for_current_msg = 0
                            continue
                        
                        elif msg_type == "tool":
                            logger.info(f"发送工具返回内容: {chat_msg.get('tool_call_id')}")
                            delta_dict["role"] = "tool"
                            delta_dict["tool_call_id"] = chat_msg.get("tool_call_id")
                            delta_dict["content"] = chat_msg.get("content", "")

                        # 处理工具调用 (Tool Calls)
                        tool_calls = chat_msg.get("tool_calls")
                        if tool_calls:
                            logger.info(f"发送工具调用请求: {[tc['name'] for tc in tool_calls]}")
                            delta_dict["tool_calls"] = [
                                {
                                    "id": tc.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("name"),
                                        "arguments": json.dumps(tc.get("args"))
                                    }
                                } for tc in tool_calls
                            ]
                        
                        if not delta_dict:
                            continue
                            
                        openai_chunk = OpenAIChatCompletionStreamResponse(
                            id=request_id,
                            created=int(time.time()),
                            model=request.model or (agent_id or "default-model"),
                            choices=[OpenAIStreamChoice(index=0, delta=OpenAIChatStreamDelta(**delta_dict), finish_reason=None)]
                        )
                        json_data = json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)
                        yield f"data: {json_data}\n\n"
                    elif data_type == "error":
                        logger.error(f"内部生成器错误: {internal_data.get('content')}")
                        yield f"data: {json.dumps({'error': {'message': internal_data['content'], 'type': 'internal_server_error', 'code': 500}}, ensure_ascii=False)}\n\n"

                except Exception as e:
                    logger.error(f"OpenAI 适配层处理块出错: {e}")

        return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")
    else:
        result = await service_handlers.invoke_handler(stream_input, agent_id)
        response = OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model or (agent_id or "default-model"),
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChatMessage(role="assistant", content=result.content),
                    finish_reason="stop"
                )
            ]
        )
        return response