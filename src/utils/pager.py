"""
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2025-11-04 13:19:13
LastEditTime: 2025-11-04 14:04:30
LastEditors: uyplayer
Description:
FilePath: /simulation-intelligent-assistant/common/pager.py
@copyright Copyright (c) 2025 by uyplayer
"""


def paginate(items: list[dict], page: int = 1, size: int = 10):
    """
    分页器
    Args:
        items (List[Dict]): 数据
        page (int, optional): 第几页. Defaults to 1.
        size (int, optional): 每页大小. Defaults to 10.

    Returns:
        _type_: _description_
    """
    start = (page - 1) * size
    end = start + size
    return items[start:end], len(items)
