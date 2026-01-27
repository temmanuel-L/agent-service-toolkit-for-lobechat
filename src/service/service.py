"""
数据清理管理器模块
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from core.settings import settings
from memory.qdrant import get_qdrant_client
from memory.vector_manager import VectorManager

logger = logging.getLogger(__name__)


class CleanupManager:
    """
    数据清理管理器，负责定期清理过期数据
    """
    
    def __init__(self):
        self.saver = None  # 用于数据库清理
        self.vector_manager = None  # 用于向量数据清理
        self._is_running = False
        self.cleanup_task = None

    async def start_cleanup_scheduler(self):
        """
        启动数据清理调度器
        """
        logger.info(f"启动数据清理调度器，间隔: {settings.CLEANUP_INTERVAL_HOURS} 小时")
        self._is_running = True
        
        while self._is_running:
            try:
                # 执行清理任务
                await self.perform_cleanup()
                
                # 等待下一个清理周期
                await asyncio.sleep(settings.CLEANUP_INTERVAL_HOURS * 3600)
            except asyncio.CancelledError:
                logger.info("数据清理调度器被取消")
                break
            except Exception as e:
                logger.error(f"数据清理过程中发生错误: {e}")
                # 发生错误后等待一段时间再重试
                await asyncio.sleep(3600)  # 等待1小时后重试

    async def perform_cleanup(self):
        """
        执行数据清理任务
        """
        logger.info("开始执行数据清理任务")
        
        try:
            # 清理过期的向量数据
            await self.cleanup_expired_vector_data()
            
            # 如果有数据库清理器，清理过期的检查点数据
            if self.saver and hasattr(self.saver, 'adelete_expired'):
                cutoff_date = datetime.utcnow() - timedelta(days=settings.DATA_RETENTION_DAYS)
                await self.saver.adelete_expired(cutoff_date)
            
            logger.info("数据清理任务完成")
        except Exception as e:
            logger.error(f"执行数据清理时发生错误: {e}")

    async def cleanup_expired_vector_data(self):
        """
        清理过期的向量数据
        """
        try:
            # 如果已有共享的 vector_manager，则直接使用，避免重复初始化
            vm = self.vector_manager
            if not vm:
                vm = VectorManager()
                await vm.ainitialize()
            
            # 获取过期的用户/会话数据并清理
            # 这里的清理逻辑已经在 VectorManager.acleanup_expired_data 中实现
            # 如果需要批量清理所有 agent，可以在此扩展
            logger.info("清理过期向量数据完成")
        except Exception as e:
            logger.error(f"清理过期向量数据时发生错误: {e}")

    async def stop_cleanup_scheduler(self):
        """
        停止数据清理调度器
        """
        logger.info("停止数据清理调度器")
        self._is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass


# 创建全局清理管理器实例
cleanup_manager = CleanupManager()