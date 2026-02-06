"""
PostgreSQL 持久化后端。

提供两类资源的连接池工厂：
- saver (AsyncPostgresSaver): LangGraph checkpointer，保存 agent 的 state / checkpoint
- store (AsyncPostgresStore): LangGraph store，保存键值数据（长期记忆摘要、agent 业务数据等）

设计原则：
- 每类资源在整个应用中只创建 **一个** 连接池，由所有 agent 共享
- pool_max_size 可由调用方按需指定（共享场景建议 >=3），默认使用全局配置
- 工厂函数内部已完成 setup()（建表等），调用方无需再次 setup
"""
import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.settings import settings

logger = logging.getLogger(__name__)


def validate_postgres_config() -> None:
    """校验 PostgreSQL 必要配置项，缺失时快速失败。"""
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]
    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )
    if settings.POSTGRES_MIN_CONNECTIONS_PER_POOL > settings.POSTGRES_MAX_CONNECTIONS_PER_POOL:
        raise ValueError(
            f"POSTGRES_MIN_CONNECTIONS_PER_POOL ({settings.POSTGRES_MIN_CONNECTIONS_PER_POOL}) "
            f"must be <= POSTGRES_MAX_CONNECTIONS_PER_POOL ({settings.POSTGRES_MAX_CONNECTIONS_PER_POOL})"
        )


def get_postgres_connection_string() -> str:
    """构建 PostgreSQL 连接字符串。"""
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


@asynccontextmanager
async def get_postgres_saver(*, pool_max_size: int | None = None):
    """
    创建 PostgreSQL checkpointer（AsyncPostgresSaver）。

    Args:
        pool_max_size: 连接池最大连接数。
            - None（默认）：使用 settings.POSTGRES_MAX_CONNECTIONS_PER_POOL
            - 显式指定：用于共享场景（建议 >=3）

    Yields:
        已完成 setup 的 AsyncPostgresSaver 实例，退出时自动关闭连接池。
    """
    validate_postgres_config()
    max_size = pool_max_size or settings.POSTGRES_MAX_CONNECTIONS_PER_POOL
    min_size = min(settings.POSTGRES_MIN_CONNECTIONS_PER_POOL, max_size)
    application_name = settings.POSTGRES_APPLICATION_NAME + "-saver"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=min_size,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "application_name": application_name,
        },
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        yield checkpointer


@asynccontextmanager
async def get_postgres_store(*, pool_max_size: int | None = None):
    """
    创建 PostgreSQL store（AsyncPostgresStore）。

    Args:
        pool_max_size: 连接池最大连接数。
            - None（默认）：使用 settings.POSTGRES_MAX_CONNECTIONS_PER_POOL
            - 显式指定：用于共享场景（建议 >=3）

    Yields:
        已完成 setup 的 AsyncPostgresStore 实例，退出时自动关闭连接池。
    """
    validate_postgres_config()
    max_size = pool_max_size or settings.POSTGRES_MAX_CONNECTIONS_PER_POOL
    min_size = min(settings.POSTGRES_MIN_CONNECTIONS_PER_POOL, max_size)
    application_name = settings.POSTGRES_APPLICATION_NAME + "-store"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=min_size,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "application_name": application_name,
        },
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        store = AsyncPostgresStore(pool)
        await store.setup()
        yield store
