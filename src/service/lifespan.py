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
from core import get_embedding_model, settings
from langfuse import Langfuse
from .service import cleanup_manager
from utils.log_utils import get_logger
import logging

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
    vector_size: int,  # 新增参数：向量维度
) -> None:
    """
    确保Qdrant集合存在，如果不存在则创建

    该函数检查指定的Agent是否在Qdrant中有对应的集合，如果没有则创建一个新的集合
    集合使用余弦距离度量，向量维度为1536（默认OpenAI embedding维度）

    Args:
        agent_key (str): Agent的唯一标识符，用作集合名称
        qdrant_client (Any): Qdrant客户端实例，用于与向量数据库通信
        existing_collections (set[str]): 已存在的集合名称集合，用于避免重复创建
        vector_size (int): 向量维度（例如 768, 1536），由实际 embedding 模型决定
    """
    should_create = agent_key not in existing_collections
    
    if not should_create:
        # 如果集合已存在，检查维度是否匹配
        try:
            collection_info = await qdrant_client.get_collection(agent_key)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != vector_size:
                logger.warning(
                    f"集合 '{agent_key}' 维度 ({current_dim}) 与当前 embedding 维度 ({vector_size}) 不匹配，正在重建..."
                )
                await qdrant_client.delete_collection(agent_key)
                should_create = True
        except Exception as e:
            logger.error(f"检查集合 '{agent_key}' 维度时出错: {e}")
            # 如果 404，说明确实需要创建
            if "not found" in str(e).lower():
                should_create = True

    if should_create:
        try:
            await qdrant_client.create_collection(
                collection_name=agent_key,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            existing_collections.add(agent_key)
            logger.info("已创建 Qdrant 集合：%s (维度: %d)", agent_key, vector_size)
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
    vector_size: int,  # 新增参数：向量维度
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
        vector_size int:  新增参数：向量维度
    """
    try:
        await load_agent(agent_key)
        logger.info(f"Agent 加载完成：{agent_key}")
    except Exception as exc:
        logger.error(f"加载 Agent {agent_key} 失败：{exc}")

    agent = get_agent(agent_key)
    agent.checkpointer = saver
    agent.store = store
    # await _ensure_qdrant_collection(agent_key, qdrant_client, existing_collections, vector_size)
    
    # 设置 Qdrant 相关属性
    setattr(agent, "qdrant_client", qdrant_client)
    # setattr(agent, "vector_collection", agent_key)
    setattr(agent, "vector_collection", "agent_conversations")
    
    # 该函数现在只负责 Agent 基础资源的绑定
    try:
        pass # 前面的 setattr 已完成大部分工作
    except Exception as e:
        logger.error(f"为智能体 {agent_key} 加载资源失败: {e}")


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
    # 使用自定义 get_logger 静默第三方库的 DEBUG 日志，防止刷屏
    # get_logger("httpcore").setLevel(logging.WARNING)
    # get_logger("httpx").setLevel(logging.WARNING)
    # get_logger("openai").setLevel(logging.WARNING)
    # get_logger("urllib3").setLevel(logging.WARNING)

    async with AsyncExitStack() as stack:
        try:
            # --- 新增：获取全局 embedding 实例 ---
            # 这个实例应该和 VectorManager 内部使用的完全相同
            global_embeddings = get_embedding_model()
            vector_size = len(global_embeddings.embed_query("hello"))  # 动态获取维度
            
            # 初始化 Qdrant 客户端
            qdrant_client = await stack.enter_async_context(get_qdrant_client())
            existing_collections = await _get_existing_collections(qdrant_client)

            # --- 确保全局向量集合 'agent_conversations' 存在 ---
            await _ensure_qdrant_collection(
                "agent_conversations",
                qdrant_client,
                existing_collections,
                vector_size=vector_size,
            )

            # 初始化所有 agents
            last_saver = None
            for agent_info in get_all_agent_info():
                saver = await stack.enter_async_context(initialize_database())
                last_saver = saver
                store = await stack.enter_async_context(initialize_store())
                await _setup_memory_components(saver, store)
                await _initialize_agent_resources(
                    agent_info.key,
                    saver,
                    store,
                    qdrant_client,
                    existing_collections,
                    vector_size
                )
                
            # 设置清理管理器的 saver 引用，以便进行数据清理
            cleanup_manager.saver = last_saver
            
            # 初始化全局向量管理器并配置工具 (仅一次)
            vector_manager = VectorManager()
            await vector_manager.ainitialize()
            
            # 设置清理管理器的向量管理引用
            cleanup_manager.vector_manager = vector_manager
            
            # 更新全局向量搜索工具的实例
            from agents.tools import vector_search_tool
            vector_search_tool.vector_manager = vector_manager

            # 将向量管理器存储在应用状态中，以便在处理请求时使用
            app.state.vector_manager = vector_manager

            # --- 初始化完成后，再启动数据清理调度器 ---
            # 确保此时 'agent_conversations' 等集合已经全部创建/校验完成
            cleanup_task = asyncio.create_task(cleanup_manager.start_cleanup_scheduler())
            
            # --- 新增：初始化全局 Langfuse 客户端 ---
            if settings.LANGFUSE_TRACING:
                try:
                    Langfuse(
                        public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value() if settings.LANGFUSE_PUBLIC_KEY else None,
                        secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value() if settings.LANGFUSE_SECRET_KEY else None,
                        host=settings.LANGFUSE_HOST
                    )
                    logger.info("Langfuse 全局客户端初始化成功")
                except Exception as e:
                    logger.error(f"Langfuse 初始化失败: {e}")
            
            yield
        except Exception as exc:
            logger.error("初始化数据库 / Store / Agent / 向量库失败：%s", exc)
            raise
        finally:
            # 清理任务
            cleanup_task.cancel()
            # 重置清理管理器的 saver 引用
            cleanup_manager.saver = None