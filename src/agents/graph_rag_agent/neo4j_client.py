"""Neo4j 同步 driver 单例管理。

由 lifespan 在服务启动时通过 set_neo4j_driver 注入 driver，
智能体运行时通过 get_neo4j_driver 获取。
开发/测试场景下若未注入，则按环境变量懒加载创建。
"""

from __future__ import annotations

from typing import Any
import os
from functools import cache

from neo4j import GraphDatabase

_driver: Any | None = None


def set_neo4j_driver(driver: Any) -> None:
    """由 lifespan 在启动时调用，注入已创建的 driver 实例。"""
    global _driver
    _driver = driver


@cache
def get_neo4j_driver():
    """获取 Neo4j 同步 driver。

    优先返回 lifespan 注入的实例；
    如果未注入（开发/测试场景），则按环境变量懒加载创建。
    """
    global _driver
    if _driver is not None:
        return _driver

    # 开发/测试回退：环境变量懒加载
    uri = os.getenv("NEO4J_URI", "bolt://192.168.10.51:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "slsltech")

    _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_neo4j_driver() -> None:
    """关闭当前 driver（通常由 lifespan 在 shutdown 时调用）。"""
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        finally:
            _driver = None