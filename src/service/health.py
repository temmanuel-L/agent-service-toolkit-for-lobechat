"""
健康检查模块
"""
from langfuse import Langfuse

from core import settings
import logging

logger = logging.getLogger(__name__)


async def health_check():
    """
    执行服务健康检查

    该函数检查服务的核心组件是否正常运行，包括基本服务状态和可选的Langfuse连接状态
    如果启用了LANGFUSE_TRACING，还会检查与Langfuse的连接状态

    Returns:
        dict: 包含健康状态信息的字典
              - status: 服务基本状态，正常情况下为"ok"
              - langfuse: (可选) Langfuse连接状态，可能为"connected"(已连接)、"disconnected"(断开连接)或"unavailable"(不可用)
    """
    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            # 使用配置参数创建 Langfuse 实例
            langfuse = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY.get_secret_value() if settings.LANGFUSE_PUBLIC_KEY else None,
                secret_key=settings.LANGFUSE_SECRET_KEY.get_secret_value() if settings.LANGFUSE_SECRET_KEY else None,
                host=settings.LANGFUSE_HOST
            )
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse连接错误: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status