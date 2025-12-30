"""
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2025-11-04 09:36:33
LastEditTime: 2025-11-04 09:40:18
LastEditors: uyplayer
Description: 时间相关的通用脚本
FilePath: /simulation-intelligent-assistant/common/time_utils.py
@copyright Copyright (c) 2025 by 3040
"""

import time
from datetime import datetime
from functools import wraps

from utils.log_utils import get_logger

logger = get_logger(__name__)


def log_execution_time(msg: str | None = None):
    """
    装饰器计算运行时间，可自定义提示信息
    :param msg: 提示信息
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            message = (
                f"[log_execution_time] 执行  {func.__name__} | 函数开始"
                if not msg
                else f"[log_execution_time] 执行  {func.__name__} | 提示: {msg}"
            )
            logger.info(message)
            result = func(*args, **kwargs)
            end_time = time.time()
            message = (
                f"[log_execution_time] 执行 {func.__name__} 函数结束 | 耗时: {end_time - start_time:.4f} 秒"
                if not msg
                else f"[log_execution_time] 执行 {func.__name__} 函数结束 | 耗时: {end_time - start_time:.4f} 秒 | 提示: {msg}"
            )
            logger.info(message)
            return result

        return wrapper

    return decorator


def current_time(time_format: str = None):
    """
    返回当前时间

    Args:
        time_format (str, optional): 时间格式 %Y-%m-%d %H:%M:%S /  %Y-%m-%d / None

    Returns:
        str : 当前时间（格式化）
    """
    if time_format is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        return datetime.now().strftime(time_format)
