# Author: Lx
# Date: 2020/1/4 10:53
# 通用工具

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from timeit import default_timer as timer
from typing import Any, Callable, TypeVar

from utils.log_utils import get_logger
from config.path_config import topo_scada_res_dir

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def print_time_consume(func: F) -> F:
    def wrapper(*args, **kwargs):
        start = timer()
        res = func(*args, **kwargs)
        info = f"{func.__name__}耗时: {(timer() - start) * 1000} ms"
        print(info)
        return res

    return wrapper  # type: ignore[return-value]


def Spend_Time_Log(log):
    """
    Decorator factory that logs the execution time of sync/async functions.
    """
    def decorator(func: F) -> F:
        if hasattr(func, '__qualname__') and '.' in func.__qualname__:
            # This is a class method, use qualname
            func_name = func.__qualname__
        else:
            # This is a regular function
            func_name = func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start = perf_counter()
                log.debug("%s start", func_name)
                try:
                    return await func(*args, **kwargs)
                finally:
                    log.debug(
                        "%s finished in %.4fs", func_name, perf_counter() - start
                    )

            return async_wrapper  # type: ignore[return-value]

        def sync_wrapper(*args, **kwargs):
            start = perf_counter()
            log.debug("%s start", func.__name__)
            try:
                return func(*args, **kwargs)
            finally:
                log.debug("%s finished in %.4fs", func.__name__, perf_counter() - start)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _record_path(conversation_id: str) -> Path:
    path = topo_scada_res_dir / conversation_id
    path.mkdir(parents=True, exist_ok=True)
    return path / "record.json"


def load_record(
    conversation_id: str, key: str, default: Any | None = None
) -> Any | None:
    record_file = _record_path(conversation_id)
    if not record_file.exists():
        return default

    try:
        data = json.loads(record_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return data.get(key, default)


def write_record(conversation_id: str, key: str, value: Any) -> None:
    record_file = _record_path(conversation_id)
    if record_file.exists():
        try:
            data = json.loads(record_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[key] = value
    record_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
    )
