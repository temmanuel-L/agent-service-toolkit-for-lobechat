import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage

class MultimodalInputProcessor:
    """通用多模态输入处理器：解析/归一化/图转文。"""

    IMAGE_TAG_URL_RE = re.compile(r"<image[^>]+?\burl=[\"']([^\"']+)[\"']", re.IGNORECASE | re.DOTALL)
    IMAGE_TAG_RE = re.compile(r"<image\b[^>]*>", re.IGNORECASE | re.DOTALL)

    @classmethod
    def strip_lobe_system_context(cls, text: str) -> str:
        for marker in ("<!-- SYSTEM CONTEXT", "<files_info>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text.strip()

    @classmethod
    def text_parts_from_content(cls, content: str | list[Any]) -> str:
        if isinstance(content, str):
            return content
        text: list[str] = []
        for item in content:
            if isinstance(item, str):
                text.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text.append(str(item.get("text", "")))
        return "".join(text)

    @classmethod
    def extract_image_urls_from_lobe_image_tags(cls, text: str) -> list[str]:
        urls = cls.IMAGE_TAG_URL_RE.findall(text or "")
        ordered: list[str] = []
        seen: set[str] = set()
        for u in urls:
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                ordered.append(u)
        return ordered

    @classmethod
    def has_openai_image_block(cls, content: Any) -> bool:
        if isinstance(content, list):
            return any(
                isinstance(block, dict) and block.get("type") in ("image_url", "image")
                for block in content
            )
        return False

    @classmethod
    def normalize_human_content(cls, content: Any) -> dict[str, Any]:
        visible_text = ""
        has_visual_input = False
        if isinstance(content, list):
            visible_text = cls.strip_lobe_system_context(cls.text_parts_from_content(content)).strip()
            has_visual_input = cls.has_openai_image_block(content)
        elif isinstance(content, str):
            visible_text = cls.strip_lobe_system_context(content).strip()
            has_visual_input = "<image" in content.lower()
        return {
            "current_user_text": visible_text,
            "has_visual_input": has_visual_input,
            "extraction_input_text": visible_text,
        }

    @classmethod
    def build_visual_content_for_model(cls, content: Any) -> Any:
        if isinstance(content, list):
            return content
        if isinstance(content, str):
            urls = cls.extract_image_urls_from_lobe_image_tags(content)
            if not urls:
                return cls.strip_lobe_system_context(content).strip()
            visible = cls.strip_lobe_system_context(content).strip()
            parts: list[dict[str, Any]] = []
            if visible:
                parts.append({"type": "text", "text": visible})
            for u in urls:
                parts.append({"type": "image_url", "image_url": {"url": u}})
            return parts if parts else visible
        return ""

    @classmethod
    def summarize_content_for_log(cls, content: Any) -> str:
        if isinstance(content, list):
            text_preview = cls.strip_lobe_system_context(cls.text_parts_from_content(content)).strip()[:300]
            image_blocks = 0
            data_url_blocks = 0
            block_types: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block_type = str(item.get("type", "unknown"))
                block_types.append(block_type)
                if block_type == "image_url":
                    image_blocks += 1
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict) and str(image_url.get("url", "")).startswith("data:image/"):
                        data_url_blocks += 1
            return (
                f"type=list block_types={block_types} image_blocks={image_blocks} "
                f"data_url_blocks={data_url_blocks} text_preview={text_preview!r}"
            )
        if isinstance(content, str):
            preview = cls.strip_lobe_system_context(content).strip()[:300]
            return f"type=str has_image_tag={'<image' in content.lower()} text_preview={preview!r}"
        return f"type={type(content).__name__}"

    @classmethod
    def extract_text_only(cls, content: Any) -> str:
        if isinstance(content, str):
            text = cls.strip_lobe_system_context(content)
            return cls.IMAGE_TAG_RE.sub("", text).strip()
        if isinstance(content, list):
            return cls.strip_lobe_system_context(cls.text_parts_from_content(content)).strip()
        return ""

    @classmethod
    def normalize_openai_user_content(cls, content: object) -> str | list[dict[str, object]]:
        if isinstance(content, list):
            return content
        if not isinstance(content, str):
            return ""
        text = cls.strip_lobe_system_context(content)
        urls = [u.strip() for u in cls.IMAGE_TAG_URL_RE.findall(text) if u and u.strip()]
        visible_text = cls.IMAGE_TAG_RE.sub("", text).strip()
        if not urls:
            return visible_text
        parts: list[dict[str, object]] = []
        if visible_text:
            parts.append({"type": "text", "text": visible_text})
        for u in urls:
            parts.append({"type": "image_url", "image_url": {"url": u}})
        return parts

    @classmethod
    async def vision_to_text(
        cls,
        llm: Any,
        config: Any,
        vision_system_prompt: str,
        content: Any,
        *,
        logger: Optional[Any] = None,
        log_prefix: str = "[视觉理解]",
    ) -> str:
        human_for_model = cls.build_visual_content_for_model(content)
        if not human_for_model:
            return ""
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = await llm.with_config(tags=["skip_stream"]).ainvoke(
                [SystemMessage(content=vision_system_prompt), HumanMessage(content=human_for_model)],
                config,
            )
            vision_text = (response.content or "").strip()
            if logger:
                logger.info(f"{log_prefix} 模型输出长度: {len(vision_text)}")
            return vision_text
        except Exception as exc:
            if logger:
                logger.error(f"{log_prefix} 调用失败: {exc}", exc_info=True)
            return ""

    @classmethod
    def apply_interrupt_image_only_fallback(
        cls,
        update: dict[str, Any],
        *,
        has_visual_input: bool,
        fallback_text: str = "[本轮仅收到图片，暂未识别出可用文本]",
        messages_field: str = "messages",
        extraction_field: str = "extraction_input_text",
    ) -> tuple[dict[str, Any], bool]:
        """
        interrupt 回合“仅图且识别失败”兜底：
        - 若本轮含图但未产生 messages，补一条哨兵 HumanMessage
        - 清空 extraction 字段，避免误抽取
        Returns: (update, whether_fallback_applied)
        """
        if not has_visual_input:
            return update, False

        has_messages = bool(update.get(messages_field))
        if has_messages:
            return update, False

        update[messages_field] = [HumanMessage(content=fallback_text)]
        update[extraction_field] = ""
        return update, True
