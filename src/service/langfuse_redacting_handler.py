from __future__ import annotations

import copy
from typing import Any

from langfuse.langchain import CallbackHandler


def _truncate_image_url(url: str, keep_chars: int = 100) -> str:
    # data URI 若被截断会触发 Langfuse media 解析报错，直接替换为占位符
    if url.startswith("data:image/"):
        return "[redacted_data_uri]"
    if len(url) <= keep_chars:
        return url
    return f"{url[:keep_chars]}...[truncated:{len(url)-keep_chars}]"


def _sanitize_any_data(obj: Any, keep_chars: int = 100) -> Any:
    """递归清理对象中的 image_url，避免 trace 中携带超大 base64。"""
    if isinstance(obj, dict):
        # 快路径：直接命中 image_url block
        if obj.get("type") == "image_url":
            image_url = obj.get("image_url")
            if isinstance(image_url, dict):
                raw_url = image_url.get("url")
                if isinstance(raw_url, str):
                    out = obj.copy()
                    out_image_url = image_url.copy()
                    out_image_url["url"] = _truncate_image_url(raw_url, keep_chars)
                    out["image_url"] = out_image_url
                    return out
        # 常规递归
        changed = False
        out: dict[str, Any] = {}
        for k, v in obj.items():
            nv = _sanitize_any_data(v, keep_chars)
            out[k] = nv
            if nv is not v:
                changed = True
        return out if changed else obj

    if isinstance(obj, list):
        changed = False
        out: list[Any] = []
        for item in obj:
            ni = _sanitize_any_data(item, keep_chars)
            out.append(ni)
            if ni is not item:
                changed = True
        return out if changed else obj

    if isinstance(obj, tuple):
        changed = False
        out_items: list[Any] = []
        for item in obj:
            ni = _sanitize_any_data(item, keep_chars)
            out_items.append(ni)
            if ni is not item:
                changed = True
        return tuple(out_items) if changed else obj

    if hasattr(obj, "content"):
        sanitized_content = _sanitize_any_data(getattr(obj, "content"), keep_chars)
        if sanitized_content is getattr(obj, "content"):
            return obj
        msg_copy = copy.copy(obj)
        msg_copy.content = sanitized_content
        return msg_copy

    return obj


class RedactingLangfuseCallbackHandler(CallbackHandler):
    """仅用于 Langfuse trace 脱敏，不影响真实模型输入。"""

    def __init__(self, keep_image_url_chars: int = 100, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keep_image_url_chars = keep_image_url_chars

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        sanitized_inputs = _sanitize_any_data(inputs, self._keep_image_url_chars)
        return super().on_chain_start(
            serialized,
            sanitized_inputs,
            *args,
            **kwargs,
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        sanitized_messages = _sanitize_any_data(messages, self._keep_image_url_chars)
        return super().on_chat_model_start(
            serialized,
            sanitized_messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # 仅定向清理 image_url/data:image 字段，保留普通文本输出（如 itinerary）可观测性
        sanitized_outputs = _sanitize_any_data(outputs, self._keep_image_url_chars)
        return super().on_chain_end(sanitized_outputs, *args, **kwargs)
