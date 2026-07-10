"""
应用生命周期管理模块。

资源初始化顺序：
1. Qdrant 客户端 → 同步所有 collection 维度
2. 共享 saver（1 个连接池 max_size=3）→ 所有 agent 的 checkpointer
3. 共享 store （1 个连接池 max_size=3）→ 所有 agent 的 store + MemoryManager
4. 加载所有 agent → 绑定 saver / store
5. 全局 VectorManager → 注入到 vector_search_tool + MemoryManager
6. Neo4j 同步 driver（供 neo4j_graphrag retriever 使用）
7. 启动数据清理调度器 / Langfuse

连接池统计（Postgres 后端）：
  - shared_saver : max_size=3  （替代原来 12 个 agent 各 1 个 = 12 个池）
  - shared_store : max_size=3  （替代原来 12 个 agent 各 1 个 + 1 个 memory 专属 = 13 个池）
  - 总计: 2 个池, 最多 6 个连接（原来 25 个池, 25+ 个连接）
"""
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
import asyncio
import os

from neo4j import GraphDatabase
from qdrant_client import models as qdrant_models
from agents import get_agent, get_all_agent_info, load_agent
from memory import initialize_database, initialize_store
from memory.long_term import memory_manager
from memory.qdrant import get_qdrant_client
from memory.vector_manager import VectorManager
from core import get_embedding_model, settings
from langfuse import Langfuse
from .service import cleanup_manager
from utils.log_utils import get_logger
from agents.graph_rag_agent.neo4j_client import set_neo4j_driver, close_neo4j_driver
from agents.graph_rag_salary.neo4j_client import (
    get_salary_neo4j_driver,
    close_salary_neo4j_driver,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 共享连接池大小：saver / store 各一个池，所有 agent + MemoryManager 共用。
# 设为 3 即可满足单用户场景下的并发需求（同时跑 agent + 异步记忆写入）。
# 如需更高并发，可在 .env 中调大 POSTGRES_MAX_CONNECTIONS_PER_POOL。
# ---------------------------------------------------------------------------
_SHARED_POOL_SIZE = max(3, settings.POSTGRES_MAX_CONNECTIONS_PER_POOL)


# ===== Qdrant collection 管理 =============================================

async def _get_existing_collections(qdrant_client: Any) -> set[str]:
    """获取 Qdrant 中已存在的 collection 列表。"""
    try:
        collections = await qdrant_client.get_collections()
        return {c.name for c in collections.collections}
    except Exception as exc:
        logger.error("获取 Qdrant 集合失败：%s", exc)
        return set()


async def _ensure_qdrant_collection(
    collection_name: str,
    qdrant_client: Any,
    existing_collections: set[str],
    vector_size: int,
) -> None:
    """确保指定 collection 存在且维度正确，不匹配时自动重建。"""
    should_create = collection_name not in existing_collections

    if not should_create:
        try:
            info = await qdrant_client.get_collection(collection_name)
            current_dim = info.config.params.vectors.size
            if current_dim != vector_size:
                logger.warning(
                    "集合 '%s' 维度 (%d) 与当前 embedding 维度 (%d) 不匹配，正在重建...",
                    collection_name, current_dim, vector_size,
                )
                await qdrant_client.delete_collection(collection_name)
                should_create = True
        except Exception as exc:
            if "not found" in str(exc).lower():
                should_create = True
            else:
                logger.error("检查集合 '%s' 维度时出错: %s", collection_name, exc)

    if should_create:
        try:
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            existing_collections.add(collection_name)
            logger.info("已创建 Qdrant 集合：%s (维度: %d)", collection_name, vector_size)
        except Exception as exc:
            if "already exists" in str(exc).lower():
                existing_collections.add(collection_name)
            else:
                logger.error("创建 Qdrant 集合 %s 失败：%s", collection_name, exc)
                raise


async def _sync_all_qdrant_collections(qdrant_client: Any, vector_size: int) -> None:
    """同步 Qdrant 中所有 collection 的维度，确保与当前 embedding 模型一致。"""
    existing = await _get_existing_collections(qdrant_client)

    # 确保 agent_conversations 始终存在
    if "agent_conversations" not in existing:
        await _ensure_qdrant_collection("agent_conversations", qdrant_client, existing, vector_size)

    for name in list(existing):
        await _ensure_qdrant_collection(name, qdrant_client, existing, vector_size)


# ===== Agent 资源绑定 =====================================================

async def _load_and_bind_agent(
    agent_key: str,
    saver: Any,
    store: Any,
) -> None:
    """
    加载 agent 并绑定 checkpointer / store。

    设计说明：
    - saver 和 store 均为全局共享实例，所有 agent 共用同一个连接池
    - saver 数据按 (thread_id, checkpoint_ns) 隔离，共享不会冲突
    - store 数据按 namespace 隔离，共享不会冲突
    """
    try:
        await load_agent(agent_key)
    except Exception as exc:
        logger.error("加载 Agent %s 失败：%s", agent_key, exc)
        return

    agent = get_agent(agent_key)
    agent.checkpointer = saver
    agent.store = store
    logger.info("Agent 初始化完成：%s", agent_key)


# ===== 生命周期入口 =======================================================

@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """
    FastAPI 应用生命周期管理器。

    启动时初始化所有基础设施（Qdrant / Postgres / Agent / 长期记忆），
    关闭时由 AsyncExitStack 自动释放全部资源。
    """
    cleanup_task = None  # 避免异常时 finally 中 UnboundLocalError
    async with AsyncExitStack() as stack:
        try:
            # 1. Embedding 模型 & 向量维度（get_embedding_model 内已用 is_ollama_reachable 等做可达性判断）
            global_embeddings = get_embedding_model()
            vector_size = len(global_embeddings.embed_query("hello"))

            # 2. Qdrant 客户端 → 同步 collection 维度
            qdrant_client = await stack.enter_async_context(get_qdrant_client())
            await _sync_all_qdrant_collections(qdrant_client, vector_size)

            # 3. 共享 saver（checkpointer）：所有 agent 共用 1 个连接池
            shared_saver = await stack.enter_async_context(
                initialize_database(pool_max_size=_SHARED_POOL_SIZE)
            )

            # 4. 共享 store：所有 agent + MemoryManager 共用 1 个连接池
            shared_store = await stack.enter_async_context(
                initialize_store(pool_max_size=_SHARED_POOL_SIZE)
            )

            # 4a. 初始化知识库元数据表
            from rag import kb_metadata
            try:
                await kb_metadata.init_kb_metadata_store()
            except Exception as exc:
                logger.warning(f"知识库元数据表初始化失败（可能 PG 未配置）: {exc}")
            try:
                from rag.service import rag_service
                rag_service.startup_parent_store_self_check()
            except Exception as exc:
                logger.warning(f"parent_store 启动自检失败（不影响启动）: {exc}")

            # 5. 加载所有 agent → 绑定 saver / store
            for agent_info in get_all_agent_info():
                await _load_and_bind_agent(agent_info.key, shared_saver, shared_store)

            # 6. 全局 VectorManager（Qdrant 对话向量管理）
            vector_manager = VectorManager()
            await vector_manager.ainitialize()

            #    6a. 注入到 cleanup_manager
            cleanup_manager.saver = shared_saver
            cleanup_manager.vector_manager = vector_manager

            #    6b. 注入到 vector_search_tool（agent 工具链使用）
            from agents.tools import vector_search_tool
            vector_search_tool.vector_manager = vector_manager

            #    6c. 存入 app.state（供 HTTP 处理器使用）
            app.state.vector_manager = vector_manager

            # 7. Neo4j 同步 driver（供 neo4j_graphrag retriever / VectorRetriever 等使用）
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://192.168.10.51:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "slsltech")
            neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            app.state.neo4j_driver = neo4j_driver
            set_neo4j_driver(neo4j_driver)
            logger.info("Neo4j 同步 driver 初始化完成：%s", neo4j_uri)

            # 7b. 薪酬 GraphRAG 专用 Neo4j（默认 7691，与参考 kg_graphrag 对齐）
            try:
                salary_driver = get_salary_neo4j_driver()
                app.state.salary_neo4j_driver = salary_driver
                logger.info("薪酬 Neo4j driver 初始化完成")
            except Exception as exc:
                logger.error("薪酬 Neo4j driver 初始化失败（graph-rag-salary 将不可用）: %s", exc)

            # 8. 初始化 MemoryManager（长期记忆）
            if settings.LONG_TERM_MEMORY_ENABLED:
                memory_manager.vector_manager = vector_manager  # 复用同一个实例
                memory_manager.set_store(shared_store)
                logger.info(
                    "长期记忆已启用: backend=%s, store=shared(pool=%d), vector=shared",
                    settings.LONG_TERM_MEMORY_BACKEND,
                    _SHARED_POOL_SIZE,
                )

            # 9. 数据清理调度器
            cleanup_task = asyncio.create_task(cleanup_manager.start_cleanup_scheduler())

            # 10. Langfuse（可选）
            if settings.LANGFUSE_TRACING:
                try:
                    Langfuse(
                        public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value() if settings.LANGFUSE_PUBLIC_KEY else None,
                        secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value() if settings.LANGFUSE_SECRET_KEY else None,
                        host=settings.LANGFUSE_HOST,
                    )
                    logger.info("Langfuse 全局客户端初始化成功")
                except Exception as exc:
                    logger.error("Langfuse 初始化失败: %s", exc)

            yield

        except Exception as exc:
            logger.error("应用初始化失败：%s", exc)
            raise
        finally:
            if cleanup_task is not None:
                cleanup_task.cancel()
            cleanup_manager.saver = None
            # 关闭 Neo4j driver
            if hasattr(app.state, "neo4j_driver") and app.state.neo4j_driver is not None:
                try:
                    app.state.neo4j_driver.close()
                    close_neo4j_driver()
                    logger.info("Neo4j 同步 driver 已关闭")
                except Exception as exc:
                    logger.error("关闭 Neo4j driver 失败: %s", exc)
            try:
                close_salary_neo4j_driver()
                logger.info("薪酬 Neo4j driver 已关闭")
            except Exception as exc:
                logger.error("关闭薪酬 Neo4j driver 失败: %s", exc)
