"""
任务管理模块，处理异步任务的注册、取消等操作
"""


import asyncio
from threading import Lock
from uuid import UUID

from schema import ChatMessage, StopTaskInput, StopTaskOutput
from utils.log_utils import get_logger

logger = get_logger(__name__)

# 任务取消
active_task_events: dict[str, asyncio.Event] = {}
stop_tasks_lock = Lock()


def _compose_task_id(user_id: str, thread_id: str) -> str:
    """
    构造任务ID

    该函数通过组合用户ID和线程ID来构造唯一的任务标识符

    Args:
        user_id (str): 用户唯一标识符
        thread_id (str): 线程唯一标识符

    Returns:
        str: 由用户ID和线程ID组合而成的任务ID，格式为"{user_id}_{thread_id}"
    """
    return f"{user_id}_{thread_id}"


def register_active_task(task_id: str) -> asyncio.Event:
    """
    注册运行中的任务，并返回其取消事件

    该函数用于注册一个正在运行的任务，将其添加到活动任务事件字典中，
    并返回一个asyncio.Event对象用于任务取消通知

    Args:
        task_id (str): 任务唯一标识符

    Returns:
        asyncio.Event: 与任务关联的事件对象，可用于监听任务取消信号
    """

    event = asyncio.Event()
    with stop_tasks_lock:
        active_task_events[task_id] = event
    return event


def unregister_active_task(task_id: str) -> None:
    """
    在任务结束后清理对应的取消事件

    该函数用于在任务完成后清理与其关联的取消事件，
    从活动任务事件字典中移除对应的任务条目

    Args:
        task_id (str): 任务唯一标识符
    """

    with stop_tasks_lock:
        active_task_events.pop(task_id, None)


def _build_stop_chat_message(run_id: UUID, content: str) -> ChatMessage:
    """
    构建停止聊天消息

    该函数用于创建一个表示对话已停止的ChatMessage对象

    Args:
        run_id (UUID): 运行实例的唯一标识符
        content (str): 消息内容

    Returns:
        ChatMessage: 包含停止信息的聊天消息对象
    """
    message = ChatMessage(type="ai", content=content)
    message.run_id = str(run_id)
    return message


async def stop_task(input: StopTaskInput) -> StopTaskOutput:
    """
    停止指定任务

    该函数用于停止正在运行的任务它会根据输入的用户ID和线程ID构造任务ID，
    查找对应的任务事件并触发取消信号

    Args:
        input (StopTaskInput): 包含用户ID和线程ID的输入参数对象

    Returns:
        StopTaskOutput: 表示停止操作结果的输出对象
    """
    task_id = _compose_task_id(input.user_id, input.thread_id)
    with stop_tasks_lock:
        event = active_task_events.get(task_id)
        if event is None:
            logger.info("请求停止任务但未找到活动任务 %s", task_id)
            return StopTaskOutput(deleted_thread_ids=[])
        event.set()
        logger.info("已记录任务 %s 的停止请求", task_id)

    return StopTaskOutput(deleted_thread_ids=[])
