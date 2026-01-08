from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_qdrant import Qdrant, QdrantVectorStore
from pydantic import SecretStr
from qdrant_client import QdrantClient, models, AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from core.llm import get_embedding_model
import inspect
from core.settings import settings
from schema import ChatMessage
from utils.log_utils import get_logger

logger = get_logger(__name__)


def validate_qdrant_config() -> None:
    """
    验证所有必需的 Qdrant 配置项是否存在

    Raises:
        ValueError: 缺少必要的 Qdrant 配置错误
    """
    from core.settings import settings
    required_vars = [
        "QDRANT_HOST",
        "QDRANT_PORT",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"缺少必要的 Qdrant 配置：{', '.join(missing)}；"
            "要启用 Qdrant 持久化请先设置这些环境变量 "
        )


def get_qdrant_connection_string() -> str:
    """
    构建指向 Qdrant 服务的连接字符串

    Returns:
        str: qdrant 连接字符串
    """
    validate_qdrant_config()
    from core.settings import settings
    return f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"


async def get_qdrant_store(
    collection_name: str = "chat_history",
    embeddings: Optional[Embeddings] = None,
    location: Optional[str] = ":memory:",
    **kwargs: Any,
) -> Qdrant:
    """
    Create a Qdrant instance with the specified parameters.

    Args:
        collection_name: Name of the collection to store vectors in
        embeddings: Embedding function to use for the store
        location: URL for the Qdrant server, defaults to in-memory
        **kwargs: Additional keyword arguments to pass to Qdrant

    Returns:
        Qdrant: A Qdrant instance
    """
    # Use the global embedding model if none is provided
    if embeddings is None:
        embeddings = get_embedding_model()

    # Get connection parameters from settings
    from core.settings import settings
    if settings.QDRANT_HOST and settings.QDRANT_PORT:
        # Use remote Qdrant instance
        location = get_qdrant_connection_string()
        kwargs["api_key"] = settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None
    else:
        # Use in-memory storage for testing/development
        location = ":memory:"

    # 根据location类型选择适当的初始化方式
    if location == ":memory:" or location.startswith("sqlite"):
        # 使用本地模式
        client = QdrantClient(location=location)
    else:
        # 使用远程服务器模式
        client = QdrantClient(url=location)
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        # content_payload_key=kwargs.pop("content_payload_key", Qdrant.CONTENT_KEY),
        # metadata_payload_key=kwargs.pop("metadata_payload_key", Qdrant.METADATA_KEY),
        # distance_strategy=kwargs.pop("distance_strategy", "COSINE"),
        # vector_name=kwargs.pop("vector_name", Qdrant.VECTOR_NAME),
        **kwargs,
    )

@asynccontextmanager
async def get_qdrant_client():
    """
    Create and yield a Qdrant client instance.
    """
    # Determine the appropriate connection parameters
    if settings.QDRANT_HOST and settings.QDRANT_PORT:
        # Use remote Qdrant instance
        client = AsyncQdrantClient(url=get_qdrant_connection_string(),)
    else:
        # Use in-memory storage for testing/development
        client = AsyncQdrantClient(location=":memory:")
    
    try:
        if settings.QDRANT_HOST and settings.QDRANT_PORT:
            logger.debug(f"正在初始化 Qdrant 客户端：{get_qdrant_connection_string()}")
            # 简单健康检查
            await client.get_collections()
        yield client
    except Exception as exc:
        logger.exception(
            f"Qdrant 客户端健康检查失败：{exc}",
        )
        raise RuntimeError("无法连接 Qdrant 服务，请检查配置与网络") from exc
    finally:
        # Close the client connection
        close_method = getattr(client, "close", None)
        if close_method is not None:
            maybe_awaitable = close_method()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable


async def adelete_points_by_metadata(
    collection_name: str,
    metadata_filter: Dict[str, Any],
    client: Optional[QdrantClient] = None
) -> bool:
    """
    Delete points from a Qdrant collection based on metadata filter.

    Args:
        collection_name: Name of the collection to delete from
        metadata_filter: Dictionary containing metadata key-value pairs to filter by
        client: Optional Qdrant client instance to use (if not provided, creates a new one)

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    own_client = client is None
    if own_client:
        if settings.QDRANT_HOST and settings.QDRANT_PORT:
            # Use remote Qdrant instance
            client = QdrantClient(
                url=get_qdrant_connection_string(),
                api_key=settings.QDRANT_API_KEY.get_secret_value() if settings.QDRANT_API_KEY else None,
            )
        else:
            # Use in-memory storage for testing/development
            client = QdrantClient(location=":memory:")
    
    try:
        # Build the filter conditions
        filter_conditions = []
        for key, value in metadata_filter.items():
            filter_conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value)
                )
            )
        
        # Create a filter with all conditions combined (AND operation)
        qdrant_filter = models.Filter(
            must=filter_conditions
        )
        
        # Perform the deletion
        await client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=qdrant_filter
            )
        )
        
        return True
    except Exception as e:
        logger.error(f"Error deleting points from Qdrant: {e}")
        return False
    finally:
        if own_client and client:
            await client.close()