# -*- coding: utf-8 -*-
"""薪酬 GraphRAG 专用 Neo4j 同步 driver。

与 graph_rag_agent（默认 7687 / 10-K 图）隔离：参考项目 kg_graphrag 使用
bolt://192.168.10.51:7691 上的薪酬知识图谱。
"""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase

from utils.log_utils import get_logger

logger = get_logger(__name__)

_driver: Any | None = None

# 与参考 Chatgpt/app/neo4j_python/neo4j_client.get_sync_driver 对齐
_DEFAULT_SALARY_URI = "bolt://192.168.10.51:7691"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASSWORD = "slsltech"


def get_salary_neo4j_database() -> str | None:
    """薪酬图数据库名：优先 NEO4J_SALARY_DB，其次 NEO4J_DB。"""
    return os.getenv("NEO4J_SALARY_DB") or os.getenv("NEO4J_DB") or "neo4j"


def get_salary_neo4j_driver():
    """获取薪酬图谱专用同步 driver（懒加载单例）。"""
    global _driver
    if _driver is not None:
        return _driver

    uri = os.getenv("NEO4J_SALARY_URI") or os.getenv("NEO4J_URI_SALARY") or _DEFAULT_SALARY_URI
    user = os.getenv("NEO4J_SALARY_USER") or os.getenv("NEO4J_USER") or _DEFAULT_USER
    password = (
        os.getenv("NEO4J_SALARY_PASSWORD")
        or os.getenv("NEO4J_PASSWORD")
        or _DEFAULT_PASSWORD
    )

    _driver = GraphDatabase.driver(uri, auth=(user, password))
    logger.info("薪酬 Neo4j driver 已创建：%s db=%s", uri, get_salary_neo4j_database())
    return _driver


def close_salary_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        finally:
            _driver = None
