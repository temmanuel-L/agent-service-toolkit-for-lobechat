"""加载 retrieval_query 意图与 statement 的工具函数。"""

import json
from pathlib import Path
from typing import List, Dict, Any

_JSON_PATH = Path(__file__).with_name("query_intention_statement.json")


def load_intention_statements() -> List[Dict[str, Any]]:
    """读取 JSON 文件，返回意图-语句列表。"""
    with _JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_statement_by_intention(intention: str) -> str | None:
    """根据 intention 精确匹配返回对应的 statement。"""
    for item in load_intention_statements():
        if item.get("intention") == intention:
            return item.get("statement")
    return None