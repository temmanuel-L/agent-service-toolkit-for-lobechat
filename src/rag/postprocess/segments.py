"""
RAG 检索结果段落拼装工具。

职责：
- 将检索得到的 NodeWithScore 列表转换为带有统一 Header 的文本段落；
- simple 模式下一条节点对应一条段落；
- parent_child 模式下，按文档来源聚合同一文档的多个命中 chunk，合并为一个更大的上下文段。
"""

from __future__ import annotations

from typing import List, Tuple, Dict

from llama_index.core.schema import NodeWithScore

from core.settings import settings
from memory.utils import is_low_quality_text
from rag.utils import RAG_SEGMENT_SEPARATOR
from utils.log_utils import get_logger

logger = get_logger(__name__)


def build_segments_from_nodes(
    nodes: List[NodeWithScore],
    start_index: int = 1,
) -> Tuple[List[str], int]:
    """
    将检索到的节点列表转换为文本段落列表，并返回下一个可用的段落编号。

    - simple 分块: 一条节点 → 一条段落。
    - parent_child 分块: 期望 nodes 已代表父级上下文（由检索阶段提升叶子命中）；
      此处与 simple 一致，一条节点 → 一条段落，仅展示父节点文本本身。
    """
    segments: List[str] = []
    segment_index = start_index

    strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()

    # simple / parent_child: 一条节点 → 一条段落
    # 在 parent_child 策略下，RagService 会在检索阶段将叶子命中提升为父节点，
    # 因此这里无需再做 doc 级聚合，直接按照节点本身作为一个段落输出。
    for nws in nodes:
        content = (nws.text or "").strip()
        if not content:
            continue

        if is_low_quality_text(content):
            logger.warning(
                "检测到 RAG 检索结果包含脏数据 (Score: %.4f, 已剔除): %s...",
                nws.score,
                content[:50].replace("\n", " "),
            )
            continue

        meta = getattr(nws.node, "metadata", {}) or {}
        source_label = meta.get("doc_title") or meta.get("file_name") or "unknown_source"

        preview = (
            content[:100].replace("\n", " ") + "..."
            if len(content) > 100
            else content.replace("\n", " ")
        )

        logger.debug(
            "  [Segment %d] Score: %.4f | Source: %s | %s",
            segment_index,
            nws.score,
            source_label,
            preview,
        )

        segment_header = (
            f"[Knowledge Segment {segment_index}] "
            f"Source: {source_label} | Score: {nws.score:.4f}"
        )
        segments.append(f"{segment_header}\n{content}")
        segment_index += 1

    return segments, segment_index


__all__ = [
    "build_segments_from_nodes",
    "RAG_SEGMENT_SEPARATOR",
]

