"""
应用生命周期管理模块
"""
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
import asyncio

from qdrant_client import models as qdrant_models
from agents import get_agent, get_all_agent_info, load_agent
from memory import initialize_database, initialize_store
from memory.qdrant import get_qdrant_client
from memory.vector_manager import VectorManager
from .service import cleanup_manager
from utils.log_utils import get_logger

logger = get_logger(__name__)


async def _setup_memory_components(saver: Any, store: Any) -> None:
    """
    初始化短期与长期记忆组件

    该函数负责初始化Agent的记忆组件，包括检查点保存器(checkpointer)和存储器(store)
    如果相应的组件具有setup方法，则调用该方法完成初始化

    Args:
        saver (Any): 检查点保存器对象，用于持久化Agent的运行状态
        store (Any): 存储器对象，用于存储Agent的长期记忆
    """
    if hasattr(saver, "setup"):
        await saver.setup()
    if hasattr(store, "setup"):
        await store.setup()


async def _get_existing_collections(qdrant_client: Any) -> set[str]:
    """
    获取Qdrant中已存在的集合列表

    该函数从Qdrant向量数据库中获取所有已存在的集合名称，用于避免重复创建相同的集合
    如果获取过程中发生异常，则记录错误并返回空集合

    Args:
        qdrant_client (Any): Qdrant客户端实例，用于与向量数据库通信

    Returns:
        set[str]: 包含所有已存在集合名称的集合，如果发生错误则返回空集合
    """
    try:
        collections = await qdrant_client.get_collections()
        return {c.name for c in collections.collections}
    except Exception as exc:
        logger.error("获取 Qdrant 集合失败：%s", exc)
        return set()


async def _ensure_qdrant_collection(
    agent_key: str,
    qdrant_client: Any,
    existing_collections: set[str],
) -> None:
    """
    确保Qdrant集合存在，如果不存在则创建

    该函数检查指定的Agent是否在Qdrant中有对应的集合，如果没有则创建一个新的集合
    集合使用余弦距离度量，向量维度为1536（默认OpenAI embedding维度）

    Args:
        agent_key (str): Agent的唯一标识符，用作集合名称
        qdrant_client (Any): Qdrant客户端实例，用于与向量数据库通信
        existing_collections (set[str]): 已存在的集合名称集合，用于避免重复创建
    """
    if agent_key in existing_collections:
        logger.info("Qdrant collection %s 已存在，跳过创建", agent_key)
        return

    try:
        # 使用更合适的向量维度（根据实际embedding模型调整）
        await qdrant_client.create_collection(
            collection_name=agent_key,
            vectors_config=qdrant_models.VectorParams(
                size=1536,  # OpenAI ada-002 embedding 维度
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        existing_collections.add(agent_key)
        logger.info("已创建 Qdrant 集合：%s", agent_key)
    except Exception as exc:
        if "already exists" in str(exc).lower():
            logger.info(f"Qdrant 集合 {agent_key} 已存在，跳过创建")
            existing_collections.add(agent_key)
        else:
            logger.error(f"创建 Qdrant 集合 {agent_key} 失败：{exc}")
            raise


async def _initialize_agent_resources(
    agent_key: str,
    saver: Any,
    store: Any,
    qdrant_client: Any,
    existing_collections: set[str],
) -> None:
    """
    初始化Agent资源并设置相关属性

    该函数负责加载指定的Agent，并为其设置必要的资源，包括检查点保存器、
    存储器、Qdrant客户端和向量集合等

    Args:
        agent_key (str): Agent的唯一标识符
        saver (Any): 检查点保存器对象
        store (Any): 存储器对象
        qdrant_client (Any): Qdrant客户端实例
        existing_collections (set[str]): 已存在的集合名称集合
    """
    try:
        await load_agent(agent_key)
        logger.info(f"Agent 加载完成：{agent_key}")
    except Exception as exc:
        logger.error(f"加载 Agent {agent_key} 失败：{exc}")

    agent = get_agent(agent_key)
    agent.checkpointer = saver
    agent.store = store
    await _ensure_qdrant_collection(agent_key, qdrant_client, existing_collections)
    
    # 设置 Qdrant 相关属性
    setattr(agent, "qdrant_client", qdrant_client)
    setattr(agent, "vector_collection", agent_key)
    
    # 为向量搜索工具设置向量管理器
    try:
        vector_manager = VectorManager()
        await vector_manager.ainitialize()
        # 更新全局向量搜索工具实例与向量管理器
        from agents.tools import vector_search_tool
        vector_search_tool.vector_manager = vector_manager
    except Exception as e:
        logger.error(f"Error setting up vector manager for agent {agent_key}: {e}")


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理器，在应用启动和关闭时执行必要的初始化和清理工作

    该异步上下文管理器负责在FastAPI应用启动时初始化所有必要的组件，
    包括数据库、向量库、Agent等资源它使用AsyncExitStack确保所有资源
    能够正确释放

    Args:
        app: FastAPI应用实例

    Yields:
        None: 生命周期管理器不产生任何值，仅用于管理资源生命周期
    """
    async with AsyncExitStack() as stack:
        try:
            # 启动数据清理任务
            cleanup_task = asyncio.create_task(cleanup_manager.start_cleanup_scheduler())
            
            # 初始化 Qdrant 客户端
            qdrant_client = await stack.enter_async_context(get_qdrant_client())
            existing_collections = await _get_existing_collections(qdrant_client)

            # 初始化所有 agents
            for agent_info in get_all_agent_info():
                saver = await stack.enter_async_context(initialize_database())
                store = await stack.enter_async_context(initialize_store())
                await _setup_memory_components(saver, store)
                await _initialize_agent_resources(
                    agent_info.key,
                    saver,
                    store,
                    qdrant_client,
                    existing_collections,
                )
                
            # 设置清理管理器的 saver 引用，以便进行数据清理
            # 注意：这里使用最后一个saver，但理想情况下应该有一个机制来访问所有saver以进行完整的清理
            cleanup_manager.saver = saver
            
            # 初始化向量管理器
            vector_manager = VectorManager()
            await vector_manager.ainitialize()
            # 将向量管理器存储在应用状态中，以便在处理请求时使用
            app.state.vector_manager = vector_manager
            
            yield
        except Exception as exc:
            logger.error("初始化数据库 / Store / Agent / 向量库失败：%s", exc)
            raise
        finally:
            # 清理任务
            cleanup_task.cancel()
            # 重置清理管理器的 saver 引用
            cleanup_manager.saver = None