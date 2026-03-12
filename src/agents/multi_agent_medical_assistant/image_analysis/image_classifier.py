# -*- coding: utf-8 -*-
"""
医疗图像分类器：使用 Vision LLM 判断图片类型，用于路由决策。

需使用支持多模态的模型（如 gpt-4o、gemini-pro-vision 等）。
vision_model 可从 agent_config 或 settings 读取，便于后续扩展 provider 枚举。
"""

from __future__ import annotations

import json
from mimetypes import guess_type
from typing import Any

from langchain_core.language_models import BaseChatModel

from utils.log_utils import get_logger

logger = get_logger(__name__)

CLASSIFICATION_PROMPT = """Determine if this is a medical image. If it is, classify it as:
'BRAIN MRI SCAN', 'CHEST X-RAY', 'SKIN LESION', or 'OTHER'. If it's not a medical image, return 'NON-MEDICAL'.
You must provide your answer in JSON format:
{"image_type": "IMAGE TYPE", "reasoning": "Your reasoning", "confidence": 0.95}"""


def image_path_to_data_url(image_path: str) -> str:
    """将本地图片路径转为 data URL（base64），供 Vision LLM 使用。"""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    with open(image_path, "rb") as f:
        b64 = f.read()
    import base64

    b64_str = base64.b64encode(b64).decode("utf-8")
    return f"data:{mime_type};base64,{b64_str}"


class ImageClassifier:
    """使用 Vision LLM 分析图片，判断医学图像类型。"""

    def __init__(self, vision_model: BaseChatModel | None = None):
        """vision_model 未提供时，由 classify_image 的 config 传入。"""
        self.vision_model = vision_model

    def classify_image(
        self,
        image_path: str,
        vision_model: BaseChatModel | None = None,
    ) -> dict[str, Any]:
        """
        分析图片，返回 {image_type, reasoning, confidence}。

        vision_model 优先使用参数，其次 self.vision_model。
        需支持多模态输入（image_url）。
        """
        model = vision_model or self.vision_model
        if not model:
            logger.warning("ImageClassifier: no vision model, returning NON-MEDICAL")
            return {
                "image_type": "NON-MEDICAL",
                "reasoning": "Vision model not configured",
                "confidence": 0.0,
            }

        data_url = image_path_to_data_url(image_path)
        logger.info("[ImageClassifier] Analyzing image: %s", image_path)

        messages = [
            {"role": "system", "content": "You are an expert in medical imaging."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        try:
            resp = model.invoke(messages)
            content = getattr(resp, "content", "") or str(resp)
            # 尝试解析 JSON
            if "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                obj = json.loads(content[start:end])
                return {
                    "image_type": str(obj.get("image_type", "unknown")).strip().upper(),
                    "reasoning": str(obj.get("reasoning", "")),
                    "confidence": float(obj.get("confidence", 0.0)),
                }
        except Exception as e:
            logger.warning("[ImageClassifier] Parse failed: %s", e)
            return {
                "image_type": "UNKNOWN",
                "reasoning": str(e),
                "confidence": 0.0,
            }

        return {
            "image_type": "UNKNOWN",
            "reasoning": "Could not parse JSON from response",
            "confidence": 0.0,
        }
