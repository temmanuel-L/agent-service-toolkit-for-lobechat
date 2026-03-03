"""
知识库元数据管理模块

用于记录知识库的元数据信息，包括：
- 知识库ID (kb_id)
- 文档列表
- 创建时间
- 最后更新时间

这些信息存储在 PostgreSQL 中，与 Qdrant 向量库配合使用，
实现知识库的完整生命周期管理。
"""
import json
import logging
from datetime import datetime
from typing import Optional

from utils.log_utils import get_logger

logger = get_logger(__name__)

# 知识库元数据表名
KB_METADATA_TABLE = "kb_metadata"


def _parse_jsonb(value):
    """
    安全解析 JSONB 字段：psycopg 可能已将其反序列化为 list/dict，无需再 json.loads。
    """
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return []


async def _ensure_kb_metadata_table(pool) -> None:
    """
    确保知识库元数据表存在，若不存在则创建。
    使用 IF NOT EXISTS 避免并发创建时的错误。
    """
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {KB_METADATA_TABLE} (
        kb_id VARCHAR(255) PRIMARY KEY,
        file_names JSONB NOT NULL DEFAULT '[]',
        file_urls JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    """
    async with pool.connection() as conn:
        await conn.execute(create_table_sql)
        await conn.commit()
    logger.info(f"知识库元数据表 {KB_METADATA_TABLE} 已就绪")


async def init_kb_metadata_store() -> None:
    """
    初始化知识库元数据存储。
    必须在应用启动时调用。
    """
    from memory.postgres import get_postgres_connection_string, validate_postgres_config
    from psycopg_pool import AsyncConnectionPool

    validate_postgres_config()
    conn_str = get_postgres_connection_string()

    async with AsyncConnectionPool(
        conn_str,
        min_size=1,
        max_size=2,
    ) as pool:
        await _ensure_kb_metadata_table(pool)


async def save_kb_metadata(
    kb_id: str,
    file_names: list[str],
    file_urls: list[str],
) -> bool:
    """
    保存或更新知识库的元数据。
    若 kb_id 已存在则更新，否则插入新记录。

    Args:
        kb_id: 知识库ID (对应 Qdrant collection 名称)
        file_names: 文档文件名列表
        file_urls: 文档URL列表

    Returns:
        bool: 操作是否成功
    """
    from memory.postgres import get_postgres_connection_string, validate_postgres_config
    from psycopg_pool import AsyncConnectionPool

    try:
        validate_postgres_config()
        conn_str = get_postgres_connection_string()

        async with AsyncConnectionPool(
            conn_str,
            min_size=1,
            max_size=2,
        ) as pool:
            await _ensure_kb_metadata_table(pool)

            async with pool.connection() as conn:
                # 使用 upsert 语法：INSERT ... ON CONFLICT DO UPDATE
                await conn.execute(
                    f"""
                    INSERT INTO {KB_METADATA_TABLE} (kb_id, file_names, file_urls, created_at, updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    ON CONFLICT (kb_id) DO UPDATE SET
                        file_names = EXCLUDED.file_names,
                        file_urls = EXCLUDED.file_urls,
                        updated_at = NOW()
                    """,
                    (kb_id, json.dumps(file_names), json.dumps(file_urls)),
                )
                await conn.commit()

            logger.info(
                f"知识库元数据已保存: kb_id={kb_id}, file_count={len(file_names)}"
            )
            return True

    except Exception as e:
        logger.error(f"保存知识库元数据失败: kb_id={kb_id}, error={e}")
        return False


async def add_file_to_kb_metadata(
    kb_id: str,
    file_name: str,
    file_url: str,
) -> bool:
    """
    向现有知识库添加单个文件的元数据。

    Args:
        kb_id: 知识库ID
        file_name: 文档文件名
        file_url: 文档URL

    Returns:
        bool: 操作是否成功
    """
    from memory.postgres import get_postgres_connection_string, validate_postgres_config
    from psycopg_pool import AsyncConnectionPool

    try:
        validate_postgres_config()
        conn_str = get_postgres_connection_string()

        async with AsyncConnectionPool(
            conn_str,
            min_size=1,
            max_size=2,
        ) as pool:
            await _ensure_kb_metadata_table(pool)

            async with pool.connection() as conn:
                # 先查询现有记录
                row = await conn.execute(
                    f"""
                    SELECT file_names, file_urls FROM {KB_METADATA_TABLE}
                    WHERE kb_id = %s
                    """,
                    (kb_id,),
                )
                existing = await row.fetchone()

                if existing:
                    # 已有记录，追加新文件（JSONB 可能已被驱动反序列化为 list）
                    file_names = _parse_jsonb(existing[0])
                    file_urls = _parse_jsonb(existing[1])

                    if file_name not in file_names:
                        file_names.append(file_name)
                    if file_url not in file_urls:
                        file_urls.append(file_url)

                    await conn.execute(
                        f"""
                        UPDATE {KB_METADATA_TABLE}
                        SET file_names = %s, file_urls = %s, updated_at = NOW()
                        WHERE kb_id = %s
                        """,
                        (json.dumps(file_names), json.dumps(file_urls), kb_id),
                    )
                else:
                    # 新建记录
                    await conn.execute(
                        f"""
                        INSERT INTO {KB_METADATA_TABLE} (kb_id, file_names, file_urls, created_at, updated_at)
                        VALUES (%s, %s, %s, NOW(), NOW())
                        """,
                        (kb_id, json.dumps([file_name]), json.dumps([file_url])),
                    )

                await conn.commit()

            logger.info(
                f"知识库文件已添加: kb_id={kb_id}, file_name={file_name}"
            )
            return True

    except Exception as e:
        logger.error(f"添加知识库文件元数据失败: kb_id={kb_id}, file_name={file_name}, error={e}")
        return False


async def delete_kb_metadata(kb_id: str) -> bool:
    """
    删除知识库的所有元数据。

    Args:
        kb_id: 知识库ID

    Returns:
        bool: 操作是否成功
    """
    from memory.postgres import get_postgres_connection_string, validate_postgres_config
    from psycopg_pool import AsyncConnectionPool

    try:
        validate_postgres_config()
        conn_str = get_postgres_connection_string()

        async with AsyncConnectionPool(
            conn_str,
            min_size=1,
            max_size=2,
        ) as pool:
            async with pool.connection() as conn:
                await conn.execute(
                    f"""
                    DELETE FROM {KB_METADATA_TABLE} WHERE kb_id = %s
                    """,
                    (kb_id,),
                )
                await conn.commit()

            logger.info(f"知识库元数据已删除: kb_id={kb_id}")
            return True

    except Exception as e:
        logger.error(f"删除知识库元数据失败: kb_id={kb_id}, error={e}")
        return False


async def delete_file_from_kb_metadata(kb_id: str, file_name: str) -> bool:
    """
    从知识库元数据中删除指定文件的记录。

    Args:
        kb_id: 知识库ID
        file_name: 要删除的文件名

    Returns:
        bool: 操作是否成功
    """
    from memory.postgres import get_postgres_connection_string, validate_postgres_config
    from psycopg_pool import AsyncConnectionPool

    try:
        validate_postgres_config()
        conn_str = get_postgres_connection_string()

        async with AsyncConnectionPool(
            conn_str,
            min_size=1,
            max_size=2,
        ) as pool:
            await _ensure_kb_metadata_table(pool)

            async with pool.connection() as conn:
                # 先查询现有记录
                row = await conn.execute(
                    f"""
                    SELECT file_names, file_urls FROM {KB_METADATA_TABLE}
                    WHERE kb_id = %s
                    """,
                    (kb_id,),
                )
                existing = await row.fetchone()

                if existing:
                    file_names = _parse_jsonb(existing[0])
                    file_urls = _parse_jsonb(existing[1])

                    if file_name in file_names:
                        index = file_names.index(file_name)
                        file_names.pop(index)
                        if index < len(file_urls):
                            file_urls.pop(index)

                        await conn.execute(
                            f"""
                            UPDATE {KB_METADATA_TABLE}
                            SET file_names = %s, file_urls = %s, updated_at = NOW()
                            WHERE kb_id = %s
                            """,
                            (json.dumps(file_names), json.dumps(file_urls), kb_id),
                        )
                        await conn.commit()
                        logger.info(f"知识库文件已从元数据中删除: kb_id={kb_id}, file_name={file_name}")
                        return True

            logger.warning(f"知识库文件不存在: kb_id={kb_id}, file_name={file_name}")
            return True  # 返回成功，因为文件中不存在也算删除成功

    except Exception as e:
        logger.error(f"从知识库元数据删除文件失败: kb_id={kb_id}, file_name={file_name}, error={e}")
        return False


async def get_kb_metadata(kb_id: str) -> Optional[dict]:
    """
    获取知识库的元数据。

    Args:
        kb_id: 知识库ID

    Returns:
        包含元数据的字典，若不存在则返回 None
    """
    from memory.postgres import get_postgres_connection_string, validate_postgres_config
    from psycopg_pool import AsyncConnectionPool

    try:
        validate_postgres_config()
        conn_str = get_postgres_connection_string()

        async with AsyncConnectionPool(
            conn_str,
            min_size=1,
            max_size=2,
        ) as pool:
            await _ensure_kb_metadata_table(pool)

            async with pool.connection() as conn:
                row = await conn.execute(
                    f"""
                    SELECT kb_id, file_names, file_urls, created_at, updated_at
                    FROM {KB_METADATA_TABLE} WHERE kb_id = %s
                    """,
                    (kb_id,),
                )
                result = await row.fetchone()

                if result:
                    return {
                        "kb_id": result[0],
                        "file_names": _parse_jsonb(result[1]),
                        "file_urls": _parse_jsonb(result[2]),
                        "created_at": result[3].isoformat() if result[3] else None,
                        "updated_at": result[4].isoformat() if result[4] else None,
                    }
                return None

    except Exception as e:
        logger.error(f"获取知识库元数据失败: kb_id={kb_id}, error={e}")
        return None
