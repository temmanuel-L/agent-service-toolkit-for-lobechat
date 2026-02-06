"""
Memory 子系统入口。

组件关系图：
  lifespan.py
    ├─ initialize_database()  → saver  (checkpointer, 保存 agent state)
    ├─ initialize_store()     → store  (键值存储, 供 agent 业务逻辑 + 长期记忆)
    ├─ VectorManager          → Qdrant (对话片段向量检索)
    └─ MemoryManager          → 长期记忆编排器 (协调 store + VectorManager)
"""
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.qdrant import get_qdrant_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store
from memory.vector_manager import VectorManager
from memory.long_term import MemoryManager, memory_manager


def initialize_database(
    *, pool_max_size: int | None = None,
) -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | AsyncMongoDBSaver
]:
    """
    创建 checkpointer（saver）。

    Args:
        pool_max_size: Postgres 连接池大小（仅 Postgres 后端生效）。
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver(pool_max_size=pool_max_size)
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    return get_sqlite_saver()


def initialize_store(*, pool_max_size: int | None = None):
    """
    创建 store（键值存储）。

    Args:
        pool_max_size: Postgres 连接池大小（仅 Postgres 后端生效）。
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store(pool_max_size=pool_max_size)
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    if settings.DATABASE_TYPE == DatabaseType.QDRANT:
        return get_qdrant_store()
    return get_sqlite_store()


__all__ = [
    "initialize_database",
    "initialize_store",
    "get_qdrant_store",
    "VectorManager",
    "MemoryManager",
    "memory_manager",
]
