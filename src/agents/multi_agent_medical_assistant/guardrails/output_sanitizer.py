# -*- coding: utf-8 -*-
"""
医疗智能体输出清洗：剥离不应暴露给前端的内部控制文本。

处理形态：
- SAFE{"agent": "...", "reasoning": "...", "confidence": 0.95} 实际回答
- SAFE```json\n{...}\n``` 实际回答
- /think ... /think ORIGINAL TEXT 实际回答
- REVISED RESPONSE: 实际回答
"""
from __future__ import annotations

import re


def sanitize_output_for_frontend(text: str) -> str:
    """
    清洗输出文本，移除 SAFE、路由决策 JSON、/think 块、ORIGINAL TEXT 等内部控制信息。

    Args:
        text: 原始输出文本

    Returns:
        清洗后的文本
    """
    if not text or not isinstance(text, str):
        return text or ""

    result = text.strip()

    # 1. 剥离 SAFE + JSON 前缀（含 SAFE{...} 和 SAFE```json...```）
    result = _strip_safe_json_prefix(result)

    # 2. 剥离 /think ... /think 块（含可能的中英变体）
    result = _strip_think_blocks(result)

    # 3. 剥离 "ORIGINAL TEXT" / "REVISED RESPONSE:" 及其前导内容，保留其后真实回答
    result = _extract_actual_response(result)

    return result.strip() if result else ""


def _strip_safe_json_prefix(text: str) -> str:
    """剥离 SAFE{...} 或 SAFE```json...``` 前缀。"""
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.upper().startswith("SAFE"):
        return text

    after_safe = stripped[4:].lstrip()

    # 形态: SAFE{"agent": ...} 或 SAFE```json\n{...}\n```
    if after_safe.startswith("{"):
        end = _find_json_object_end(after_safe)
        if end >= 0:
            return after_safe[end + 1 :].lstrip()
        return after_safe
    if after_safe.upper().startswith("```JSON"):
        after_safe = after_safe[7:].lstrip()
    elif after_safe.startswith("```"):
        after_safe = after_safe[3:].lstrip()
        if after_safe.upper().startswith("JSON"):
            after_safe = after_safe[4:].lstrip()

    idx = after_safe.find("```")
    if idx >= 0:
        return after_safe[idx + 3 :].lstrip()
    return after_safe


def _find_json_object_end(s: str) -> int:
    """从第一个 { 开始，找到匹配的 } 的位置。"""
    if not s or s[0] != "{":
        return -1
    depth = 0
    in_string = False
    escape = False
    quote_char = None
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
            elif c in ('"', "'"):
                in_string = True
                quote_char = c
        elif c == quote_char:
            in_string = False
    return -1


def _strip_think_blocks(text: str) -> str:
    """剥离 /think ... /think 块。"""
    if not text:
        return text
    pattern = re.compile(
        r"/think\s*[\s\S]*?/think\s*",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.sub("", text).strip()


def _extract_actual_response(text: str) -> str:
    """
    若存在 "ORIGINAL TEXT" 或 "REVISED RESPONSE:"，则保留其后内容；
    否则返回原文。
    最后移除结果中残留的 ORIGINAL TEXT / REVISED RESPONSE 标记（含中英标点变体）。
    """
    if not text:
        return text

    markers = [
        "ORIGINAL TEXT",
        "REVISED RESPONSE:",
        "REVISED RESPONSE：",
        "ORIGINAL TEXT/think",
    ]
    text_upper = text.upper()
    best_idx = -1
    best_len = 0
    for marker in markers:
        idx = text_upper.find(marker.upper())
        if idx >= 0 and (best_idx < 0 or idx < best_idx):
            best_idx = idx
            best_len = len(marker)
    if best_idx >= 0:
        after = text[best_idx + best_len :].strip()
        if after:
            text = after

    # 移除结果中残留的 guardrails 标记（LLM 可能在多处输出）
    strip_patterns = [
        r"\s*ORIGINAL\s+TEXT\s*",
        r"\s*REVISED\s+RESPONSE\s*:?\s*",
        r"\s*REVISED\s+RESPONSE\s*：?\s*",
    ]
    for pat in strip_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text.strip()
