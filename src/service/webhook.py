"""
Webhook 处理模块
包含处理删除事件的函数，但目前未被激活使用
"""
import hashlib
import hmac
import json
from typing import Dict, Any

from fastapi import HTTPException, Request
from pydantic import ValidationError

from schema import WebhookPayload
from core.settings import settings
from . import conversation as conversation_handlers
from . import service as service_module
from memory.qdrant import adelete_points_by_metadata
from memory.postgres import get_postgres_store


async def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    验证 webhook 请求的签名

    Args:
        payload (bytes): 请求体数据
        signature (str): 请求签名
        secret (str): 验证密钥

    Returns:
        bool: 验证是否成功
    """
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    expected_signature_header = f"sha256={expected_signature}"
    return hmac.compare_digest(signature, expected_signature_header)


async def handle_webhook_event(payload: WebhookPayload) -> Dict[str, Any]:
    """
    处理 webhook 事件

    Args:
        payload (WebhookPayload): Webhook负载数据

    Returns:
        Dict[str, Any]: 处理结果

    Raises:
        HTTPException: 当事件类型未知或参数缺失时
    """
    event_type = payload.event
    
    if event_type == "thread.deleted":
        # 处理会话删除事件
        if payload.thread_id:
            await handle_thread_deletion(payload.thread_id, payload.user_id)
            return {"status": "success", "message": f"Thread {payload.thread_id} deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="thread_id is required for thread deletion")
    
    elif event_type == "user.deleted":
        # 处理用户删除事件
        if payload.user_id:
            await handle_user_deletion(payload.user_id)
            return {"status": "success", "message": f"User {payload.user_id} and all related data deleted successfully"}
        else:
            raise HTTPException(status_code=400, detail="user_id is required for user deletion")
    
    else:
        # 未知事件类型
        raise HTTPException(status_code=400, detail=f"Unknown event type: {event_type}")


async def handle_thread_deletion(thread_id: str, user_id: str = None) -> None:
    """
    处理会话删除

    同时删除 Postgres 和 Qdrant 中的相关数据

    Args:
        thread_id (str): 会话ID
        user_id (str, optional): 用户ID

    Raises:
        HTTPException: 当删除操作失败时
    """
    # 从 Postgres 删除会话数据
    # 使用 conversation_handler 中的逻辑来删除会话
    from agents import get_agent, get_all_agent_info
    from agents.agents import DEFAULT_AGENT
    from langchain_core.runnables import RunnableConfig
    from schema import DeleteConversationInput
    
    # 对所有 agent 执行删除操作
    for agent_info in get_all_agent_info():
        agent_key = agent_info.key
        agent = get_agent(agent_key)
        checkpointer = getattr(agent, "checkpointer", None)
        
        if checkpointer and hasattr(checkpointer, "adelete_thread"):
            try:
                # 删除 Postgres 中的会话数据
                await checkpointer.adelete_thread(thread_id=thread_id)
                
                # 如果提供了 user_id，也删除存储在 store 中的相关数据
                store = getattr(agent, "store", None)
                if store and user_id:
                    try:
                        await store.adelete(
                            namespace=(user_id, "conversation_topic"), key=thread_id
                        )
                    except Exception:
                        # 如果删除话题失败，继续执行，不中断主要流程
                        pass
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete thread from Postgres: {str(e)}")
    
    # 从 Qdrant 删除会话数据
    try:
        # 遍历所有 agent，删除对应的数据
        for agent_info in get_all_agent_info():
            agent_key = agent_info.key
            await adelete_points_by_metadata(
                collection_name=agent_key,
                metadata_filter={"thread_id": thread_id}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete thread from Qdrant: {str(e)}")


async def handle_user_deletion(user_id: str) -> None:
    """
    处理用户删除

    同时删除 Postgres 和 Qdrant 中该用户的所有数据

    Args:
        user_id (str): 用户ID

    Raises:
        HTTPException: 当删除操作失败时
    """
    # 从 Postgres 删除用户的所有会话数据
    from agents import get_agent, get_all_agent_info
    from agents.agents import DEFAULT_AGENT
    
    # 对所有 agent 执行删除操作
    for agent_info in get_all_agent_info():
        agent_key = agent_info.key
        agent = get_agent(agent_key)
        checkpointer = getattr(agent, "checkpointer", None)
        
        if checkpointer and hasattr(checkpointer, "alist"):
            try:
                # 获取用户的所有会话
                conversations_iter = checkpointer.alist(config=None, filter={"user_id": user_id})
                deleted_threads = []
                
                async for checkpoint_tuple in conversations_iter:
                    cp_config = checkpoint_tuple.config or {}
                    configurable = cp_config.get("configurable") or {}
                    thread_id = configurable.get("thread_id")
                    
                    if isinstance(thread_id, str) and thread_id:
                        deleted_threads.append(thread_id)
                        # 删除会话数据
                        await checkpointer.adelete_thread(thread_id=thread_id)
                
                # 删除存储在 store 中的相关数据
                store = getattr(agent, "store", None)
                if store:
                    try:
                        # 删除用户的所有话题数据
                        # 这里我们不能直接删除命名空间，所以需要删除特定用户的所有话题
                        pass  # 实际实现可能需要更具体的逻辑
                    except Exception:
                        # 如果删除话题失败，继续执行，不中断主要流程
                        pass
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete user's threads from Postgres: {str(e)}")
    
    # 从 Qdrant 删除用户的所有数据
    try:
        from agents import get_all_agent_info
        
        # 遍历所有 agent，删除该用户的数据
        for agent_info in get_all_agent_info():
            agent_key = agent_info.key
            await adelete_points_by_metadata(
                collection_name=agent_key,
                metadata_filter={"user_id": user_id}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user's data from Qdrant: {str(e)}")


async def webhook_endpoint(request: Request) -> Dict[str, Any]:
    """
    Webhook 入口点

    处理来自LobeChat的webhook事件，如用户删除、会话删除等

    Args:
        request (Request): HTTP请求对象

    Returns:
        Dict[str, Any]: 处理结果

    Raises:
        HTTPException: 当验证失败或请求无效时
    """
    
    # 获取请求体
    body = await request.body()
    
    # 验证签名（如果配置了 WEBHOOK_SECRET）
    if settings.WEBHOOK_SECRET:
        signature = request.headers.get("x-signature-256") or request.headers.get("X-Signature-256")
        if not signature:
            raise HTTPException(status_code=400, detail="Missing signature header")
        
        secret = settings.WEBHOOK_SECRET.get_secret_value()
        is_valid = await verify_webhook_signature(body, signature, secret)
        if not is_valid:
            raise HTTPException(status_code=403, detail="Invalid signature")
    
    try:
        # 解析请求体
        payload_data = json.loads(body.decode('utf-8'))
        payload = WebhookPayload(**payload_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
    
    # 处理事件
    return await handle_webhook_event(payload)

