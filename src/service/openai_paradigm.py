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


async def chat_completions_handler(
    request: OpenAIChatCompletionRequest,
    agent_id: str = None,
    user_id: str = None,
    thread_id: str = None
) -> StreamingResponse | OpenAIChatCompletionResponse:
    """
    OpenAI兼容的聊天完成接口处理器

    该函数处理OpenAI格式的请求并返回兼容的响应

    Args:
        request (OpenAIChatCompletionRequest): OpenAI格式的聊天完成请求
        agent_id (str): 指定的agent ID
        user_id (str, optional): 用户ID
        thread_id (str, optional): 会话ID

    Returns:
        StreamingResponse: OpenAI格式的流式响应
    """
    # 将 OpenAI 格式的消息转换为内部格式
    if request.messages:
        last_message = request.messages[-1]
        user_message = last_message.get("content", "")
        # 使用传入的 user_id，如果未提供则使用默认值
        user_id = user_id or request.user or "default_user"
    else:
        user_message = ""
        user_id = user_id or request.user or "default_user"

    # 创建内部 StreamInput 对象，包含更多 OpenAI 参数
    stream_input = StreamInput(
        message=user_message,
        user_id=user_id,
        thread_id=thread_id,  # 使用传入的 thread_id
        model=None,
        stream_tokens=request.stream,
        # 将 OpenAI 的参数映射到 agent_config 中
        agent_config={
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "stop": request.stop
        }
    )

    # 根据是否需要流式响应来调用相应的处理器
    if request.stream:
        # 对于流式响应，我们需要创建一个特殊的生成器来输出 OpenAI 格式的 SSE
        async def openai_stream_generator():
            # 调用内部的消息生成器
            async for chunk in service_handlers.message_generator(stream_input, agent_id):
                # 检查是否是 SSE 格式的消息
                if chunk.startswith("data: "):
                    # 解析内部格式的消息
                    data_str = chunk[6:].strip()  # 移除 "data: " 前缀
                    if data_str == "[DONE]":
                        # 发送 OpenAI 格式的完成消息
                        # 添加 finish_reason 来表示完成
                        finish_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model or (agent_id or "default-model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                        yield f"data: [DONE]\n\n"
                    else:
                        try:
                            internal_data = json.loads(data_str)
                            if internal_data["type"] == "message":
                                chat_msg = internal_data["content"]
                                delta = {}
                                
                                # 确定角色
                                if chat_msg["type"] == "ai":
                                    delta["role"] = "assistant"
                                elif chat_msg["type"] == "tool":
                                    delta["role"] = "tool"
                                    delta["tool_call_id"] = chat_msg.get("tool_call_id")
                                
                                # 处理内容 (避免 AI 消息重复推送)
                                content = chat_msg.get("content")
                                if content:
                                    # 如果是流式模式下的 AI 消息，内容通常已经通过 'token' 事件发送过了，所以这里跳过
                                    # 但对于 tool 类型消息（工具执行结果），我们需要发送内容
                                    if chat_msg["type"] == "tool":
                                        delta["content"] = content
                                    elif chat_msg["type"] == "ai" and not request.stream:
                                        delta["content"] = content

                                # 处理工具调用 (Tool Calls)
                                if chat_msg.get("tool_calls"):
                                    delta["tool_calls"] = [
                                        {
                                            "index": i,
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tc["name"],
                                                "arguments": json.dumps(tc["args"], ensure_ascii=False)
                                            }
                                        } for i, tc in enumerate(chat_msg["tool_calls"])
                                    ]
                                
                                if delta:
                                    openai_chunk = OpenAIChatCompletionStreamResponse(
                                        id=f"chatcmpl-{uuid.uuid4()}",
                                        created=int(time.time()),
                                        model=request.model or (agent_id or "default-model"),
                                        choices=[
                                            OpenAIStreamChoice(
                                                index=0,
                                                delta=OpenAIChatStreamDelta(**delta),
                                                finish_reason=None
                                            )
                                        ]
                                    )
                                    yield f"data: {openai_chunk.model_dump_json(exclude_none=True)}\n\n"

                            elif internal_data["type"] == "error":
                                # 处理来自 service handler 的显式错误
                                error_content = internal_data["content"]
                                yield f"data: {json.dumps({'error': {'message': error_content, 'type': 'internal_server_error', 'code': 500}}, ensure_ascii=False)}\n\n"

                            elif internal_data["type"] == "token":
                                content = internal_data["content"]
                                # 转换为 OpenAI 格式
                                openai_chunk = OpenAIChatCompletionStreamResponse(
                                    id=f"chatcmpl-{uuid.uuid4()}",
                                    created=int(time.time()),
                                    model=request.model or (agent_id or "default-model"),
                                    choices=[
                                        OpenAIStreamChoice(
                                            index=0,
                                            delta=OpenAIChatStreamDelta(content=content),
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {openai_chunk.model_dump_json(exclude_none=True)}\n\n"
                        except json.JSONDecodeError:
                            # 如果解析失败，直接传递原始数据
                            yield chunk
                        except Exception as e:
                            # 处理其他错误
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")
    else:
        result = await service_handlers.invoke_handler(stream_input, agent_id)
        # 这里需要将结果转换为 OpenAI 格式
        response = OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model or (agent_id or "default-model"),
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=result.content
                    ),
                    finish_reason="stop"
                )
            ]
        )
        return response