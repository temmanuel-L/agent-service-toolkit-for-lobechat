# -*- coding: utf-8 -*-
"""salary-graphrag 专用 Neo4j 同步 driver。

与 graph_rag_salary（默认 7691）隔离：本智能体默认 bolt://192.168.10.51:7680。
环境变量使用 NEO4J_SALARY_GRAPHRAG_* 前缀，避免误读 NEO4J_SALARY_URI（7691）。
"""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase

from utils.log_utils import get_logger

logger = get_logger(__name__)

_driver: Any | None = None

_DEFAULT_URI = "bolt://192.168.10.51:7690"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASSWORD = "slsltech"


def get_salary_graphrag_neo4j_database() -> str | None:
    """数据库名：优先 NEO4J_SALARY_GRAPHRAG_DB，其次 NEO4J_DB。"""
    return os.getenv("NEO4J_SALARY_GRAPHRAG_DB") or os.getenv("NEO4J_DB") or "neo4j"


def get_salary_graphrag_neo4j_driver():
    """获取本智能体专用同步 driver（懒加载单例）。"""
    global _driver
    if _driver is not None:
        return _driver

    uri = os.getenv("NEO4J_SALARY_GRAPHRAG_URI") or _DEFAULT_URI
    user = (
        os.getenv("NEO4J_SALARY_GRAPHRAG_USER")
        or os.getenv("NEO4J_USER")
        or _DEFAULT_USER
    )
    password = (
        os.getenv("NEO4J_SALARY_GRAPHRAG_PASSWORD")
        or os.getenv("NEO4J_PASSWORD")
        or _DEFAULT_PASSWORD
    )

    _driver = GraphDatabase.driver(uri, auth=(user, password))
    logger.info(
        "salary-graphrag Neo4j driver 已创建：%s db=%s",
        uri,
        get_salary_graphrag_neo4j_database(),
    )
    return _driver


def close_salary_graphrag_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        finally:
            _driver = None
