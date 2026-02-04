"""
Author: uyplayer
Email: uyplayer@outlook.com
Date: 2025-11-04 09:36:33
LastEditTime: 2025-11-04 09:37:32
LastEditors: uyplayer
Description: 项目log 模块
FilePath: /simulation-intelligent-assistant/common/log_utils.py
@copyright Copyright (c) 2025 by uyplayer
"""

import logging
import os
import sys
from pathlib import Path

from logging.handlers import TimedRotatingFileHandler

try:
    from config.path_config import log_dir
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "log"
    os.makedirs(log_dir, exist_ok=True)

# 从 settings 读取日志级别，实现统一的日志级别控制
# 在 .env 中设置 LOG_LEVEL=DEBUG/INFO/WARNING/ERROR 来控制日志输出
def _get_log_level() -> str:
    """获取日志级别，优先从 settings 读取"""
    try:
        from core.settings import settings
        return settings.LOG_LEVEL.value.upper()
    except Exception:
        # 如果 settings 不可用（如启动早期），使用默认值
        return os.environ.get("LOG_LEVEL", "INFO").upper()


class Logger:
    def __init__(self, logger_name="framework", path=log_dir):
        self._logger = logging.getLogger(logger_name)
        self._logger.propagate = False
        logging.root.setLevel(logging.NOTSET)

        self.log_path = path
        self.log_file_name = "agent_local.log"  # 日志文件
        self.backup_count = 14  # 保留的日志数量
        # 日志输出级别 - 从环境变量或 settings 动态获取
        log_level = _get_log_level()
        self.console_output_level = log_level
        self.file_output_level = log_level
        # 设置 logger 自身的级别，否则 INFO 级别的日志可能被默认的 WARNING 级别过滤掉
        self._logger.setLevel(self.console_output_level)
        # 日志输出格式
        pattern = "%(asctime)s - %(filename)s [Line:%(lineno)d] - %(levelname)s - %(message)s"
        self.formatter = logging.Formatter(pattern)

    def get_logger(self):
        """在logger中添加日志句柄并返回，如果logger已有句柄，则直接返回
        我们这里添加两个句柄，一个输出日志到控制台，另一个输出到日志文件
        两个句柄的日志级别不同，在配置文件中可设置
        """
        if not self._logger.handlers:  # 避免重复日志
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            self._logger.addHandler(console_handler)

            # 每天重新创建一个日志文件，最多保留backup_count份
            file_handler = TimedRotatingFileHandler(
                filename=os.path.join(self.log_path, self.log_file_name),
                when="midnight",
                interval=1,
                backupCount=self.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self._logger.addHandler(file_handler)
        return self._logger


def get_logger(name):
    return Logger(logger_name=name).get_logger()


if __name__ == "__main__":
    my_log = get_logger("test")
    my_log.info("hello test ")
