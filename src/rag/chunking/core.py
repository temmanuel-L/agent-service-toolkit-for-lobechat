"""
RAG 分块（Chunking）核心工具。

当前提供：
- 基于 LlamaIndex SentenceSplitter 的固定窗口分块策略；
- 通过 settings.RAG_CHUNKING_STRATEGY 预留父子分块等扩展点（目前仅 simple）。
"""

from __future__ import annotations

import tiktoken
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser

from core.settings import settings


def build_default_sentence_splitter() -> SentenceSplitter:
    """
    基于当前 settings 构建一个默认的 SentenceSplitter。

    - 使用 cl100k_base 的 tiktoken 编码做 token 级分块；
    - chunk_size / chunk_overlap 来自 RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP。
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = SentenceSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        tokenizer=encoding.encode,
    )
    return splitter


def build_chunking_transformations() -> list:
    """
    根据 RAG_CHUNKING_STRATEGY 返回用于 LlamaIndex 的 transformations 列表。

    当前支持：
    - "simple": 仅使用 SentenceSplitter 做固定窗口分块（默认）；
    - "parent_child": 使用 HierarchicalNodeParser 做父子分块（大块作为 parent，小块作为 child）。
    """
    strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()

    if strategy == "parent_child":
        # 父子分块策略：使用 HierarchicalNodeParser 生成两层节点
        #
        # 设计：
        # - child 层：使用当前 RAG_CHUNK_SIZE 作为较小粒度，用于向量检索；
        # - parent 层：使用 ~3x 的窗口作为较大粒度，用于提供更完整的上下文。
        #
        # HierarchicalNodeParser 会自动为 child 节点打上 parent 关联信息，
        # 后续若需要可以在检索侧利用这些元数据做 parent 级别的上下文展开。
        base_size = settings.RAG_CHUNK_SIZE
        parent_size = max(base_size * 3, base_size + settings.RAG_CHUNK_OVERLAP)
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_size, base_size]
        )
        return [parser]

    # 默认 simple 策略：单层 SentenceSplitter 固定窗口
    splitter = build_default_sentence_splitter()
    return [splitter]


__all__ = [
    "build_default_sentence_splitter",
    "build_chunking_transformations",
]

