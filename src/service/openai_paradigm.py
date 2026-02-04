"""
OpenAI API 范式适配层

该模块提供与 OpenAI API 兼容的接口实现，用于与 LobeChat 等前端集成。

Architecture:
- 本层是一个纯粹的格式转换层，将内部SSE格式转换为OpenAI兼容格式
- 所有的消息去重、过滤逻辑已在 handlers.py 的 message_generator 中完成
- 本层不再进行任何去重操作，只负责格式转换

Event Types (来自 message_generator):
- token: LLM生成的token流，直接转换为OpenAI delta格式
- message: 完整消息事件（interrupt等），已经过去重，可直接发送
- error: 错误事件
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
from .handlers import TOOLS_WITH_HIDDEN_OUTPUT
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
    
    本函数是OpenAI API兼容层的入口点，负责：
    1. 解析OpenAI格式的请求
    2. 调用内部的message_generator获取流式响应
    3. 将内部格式转换为OpenAI兼容格式
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
            "stop": request.stop,
            "kb_ids": request.kb_ids
        }
    )

    if request.stream:
        return StreamingResponse(
            _openai_stream_adapter(stream_input, agent_id, request.model, thread_id),
            media_type="text/event-stream"
        )
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


async def _openai_stream_adapter(
    stream_input: StreamInput,
    agent_id: str,
    model_name: str,
    thread_id: str
) -> AsyncGenerator[str, None]:
    """
    OpenAI流式响应适配器
    
    将内部的SSE事件流转换为OpenAI兼容的流式格式。
    
    设计原则：
    - 简单直接的格式转换，不进行任何去重操作
    - 所有去重已在 message_generator 层完成
    - 收到的每个token和message事件都应该被转发
    """
    role_sent = False
    request_id = f"chatcmpl-{uuid.uuid4()}"
    effective_model = model_name or agent_id or "default-model"
    
    # 统计发送的内容
    tokens_sent = 0
    messages_sent = 0
    tool_calls_sent = 0
    
    logger.info(f"开启 OpenAI 兼容流式响应: agent={agent_id}, thread={thread_id}")

    async for chunk in service_handlers.message_generator(stream_input, agent_id):
        if not chunk.startswith("data: "):
            continue

        data_str = chunk[6:].strip()
        
        # 处理结束信号
        if data_str == "[DONE]":
            logger.info(f"[OPENAI_ADAPTER] 流结束统计: tokens_sent={tokens_sent}, messages_sent={messages_sent}, tool_calls_sent={tool_calls_sent}, role_sent={role_sent}")
            if not role_sent:
                logger.warning("[OPENAI_ADAPTER] 警告: 流结束但没有发送任何内容到前端!")
            finish_chunk = OpenAIChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=effective_model,
                choices=[OpenAIStreamChoice(index=0, delta=OpenAIChatStreamDelta(), finish_reason="stop")]
            )
            yield f"data: {json.dumps(finish_chunk.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            continue

        try:
            internal_data = json.loads(data_str)
            data_type = internal_data.get("type")
            
            # DEBUG: 记录收到的所有事件
            if data_type == "message":
                chat_msg = internal_data.get("content", {})
                msg_type = chat_msg.get("type")
                msg_content = chat_msg.get("content", "")
                msg_tool_calls = chat_msg.get("tool_calls")
                logger.debug(f"[OPENAI_ADAPTER] 收到message事件: msg_type={msg_type}, content_len={len(msg_content) if msg_content else 0}, has_tool_calls={bool(msg_tool_calls)}")
            elif data_type == "token":
                logger.debug(f"[OPENAI_ADAPTER] 收到token事件: content_len={len(internal_data.get('content', ''))}")
            
            # ========== Token事件：LLM生成的token流 ==========
            if data_type == "token":
                content = internal_data.get("content")
                if not content:
                    continue

                delta = {"content": content}
                if not role_sent:
                    delta["role"] = "assistant"
                    role_sent = True
                
                tokens_sent += 1
                openai_chunk = OpenAIChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=effective_model,
                    choices=[OpenAIStreamChoice(
                        index=0,
                        delta=OpenAIChatStreamDelta(**delta),
                        finish_reason=None
                    )]
                )
                yield f"data: {json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"

            # ========== Message事件：完整消息（已去重） ==========
            elif data_type == "message":
                chat_msg = internal_data.get("content", {})
                msg_type = chat_msg.get("type")
                
                # AI消息：用于interrupt等非流式场景
                if msg_type == "ai":
                    content = chat_msg.get("content", "")
                    if content and content.strip():
                        delta = {"content": content}
                        if not role_sent:
                            delta["role"] = "assistant"
                            role_sent = True
                        
                        messages_sent += 1
                        openai_chunk = OpenAIChatCompletionStreamResponse(
                            id=request_id,
                            created=int(time.time()),
                            model=effective_model,
                            choices=[OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIChatStreamDelta(**delta),
                                finish_reason=None
                            )]
                        )
                        yield f"data: {json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"
                    
                    # 处理工具调用（如果有）
                    # 注意：内部工具（TOOLS_WITH_HIDDEN_OUTPUT 中定义的）的调用不应该发送到前端
                    # 这些工具由后端执行，前端不需要知道
                    tool_calls = chat_msg.get("tool_calls")
                    if tool_calls:
                        # 过滤掉内部工具调用（使用 handlers.py 中定义的 TOOLS_WITH_HIDDEN_OUTPUT）
                        external_tool_calls = [tc for tc in tool_calls if tc.get("name") not in TOOLS_WITH_HIDDEN_OUTPUT]
                        internal_tool_calls = [tc for tc in tool_calls if tc.get("name") in TOOLS_WITH_HIDDEN_OUTPUT]
                        
                        if internal_tool_calls:
                            logger.debug(f"跳过内部工具调用（后端执行）: {[tc.get('name') for tc in internal_tool_calls]}")
                        
                        if external_tool_calls:
                            logger.debug(f"发送外部工具调用请求到前端: {[tc.get('name') for tc in external_tool_calls]}")
                            tool_calls_sent += 1
                            delta_with_tools = {
                                "tool_calls": [
                                    {
                                        "id": tc.get("id"),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("name"),
                                            "arguments": json.dumps(tc.get("args"))
                                        }
                                    } for tc in external_tool_calls
                                ]
                            }
                            if not role_sent:
                                delta_with_tools["role"] = "assistant"
                                role_sent = True
                            
                            openai_chunk = OpenAIChatCompletionStreamResponse(
                                id=request_id,
                                created=int(time.time()),
                                model=effective_model,
                                choices=[OpenAIStreamChoice(
                                    index=0,
                                    delta=OpenAIChatStreamDelta(**delta_with_tools),
                                    finish_reason=None
                                )]
                            )
                            yield f"data: {json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"
                
                # ========== Tool消息：工具执行结果 ==========
                # 某些工具（如 WebSearch）的输出需要展示给用户
                # handlers.py 已经过滤了内部工具（如 search_knowledge），
                # 这里到达的 tool 消息都是应该展示的
                elif msg_type == "tool":
                    content = chat_msg.get("content", "")
                    tool_name = chat_msg.get("name", "tool")
                    
                    if content and content.strip():
                        # 将工具输出格式化后作为 AI 消息发送
                        # 这样前端可以正常显示工具执行结果
                        logger.debug(f"发送工具 '{tool_name}' 的输出到前端: len={len(content)}")
                        
                        messages_sent += 1
                        delta = {"content": content}
                        if not role_sent:
                            delta["role"] = "assistant"
                            role_sent = True
                        
                        openai_chunk = OpenAIChatCompletionStreamResponse(
                            id=request_id,
                            created=int(time.time()),
                            model=effective_model,
                            choices=[OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIChatStreamDelta(**delta),
                                finish_reason=None
                            )]
                        )
                        yield f"data: {json.dumps(openai_chunk.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"
                
                # 自定义消息类型（如有需要可扩展）
                elif msg_type == "custom":
                    custom_data = chat_msg.get("custom_data", {})
                    if custom_data:
                        # 将自定义数据作为特殊格式发送（前端可选择处理）
                        logger.debug(f"发送自定义消息: {list(custom_data.keys())}")

            # ========== Error事件 ==========
            elif data_type == "error":
                error_content = internal_data.get("content", "Unknown error")
                logger.error(f"内部生成器错误: {error_content}")
                yield f"data: {json.dumps({'error': {'message': error_content, 'type': 'internal_server_error', 'code': 500}}, ensure_ascii=False)}\n\n"

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}, data={data_str[:100]}")
        except Exception as e:
            logger.error(f"OpenAI 适配层处理块出错: {e}", exc_info=True)