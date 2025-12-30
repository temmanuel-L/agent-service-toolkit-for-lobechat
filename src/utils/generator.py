"""
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2025-09-10 10:58:02
LastEditTime: 2025-10-24 15:58:56
LastEditors: uyplayer
Description: 用户相关的操作 生成随机id / 生成主题等
FilePath: /simulation-intelligent-assistant/app/steam_report/toolkit/user_info.py
@copyright Copyright (c) 2025 by 3040
"""

import uuid
from datetime import datetime


def generate_id(tag=None) -> str:
    """
    生成随机id

    Args:
        tag (str, optional): tag 作为 生成id的前缀，如果不提供不进行插入前缀

    Returns:
        str: 生成的id
    """

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    uid = uuid.uuid4().hex

    return f"{tag}_{timestamp}_{uid}" if tag else f"{timestamp}_{uid}"


def generate_tool_id(length: int = 15):
    """
    生成指定长度的十六进制 tool id

    Args:
        length (int): 截取的字符长度，默认 15
    """
    if length <= 0:
        raise ValueError("length 必须为正整数")
    uid = uuid.uuid4().hex
    return uid[:length]
