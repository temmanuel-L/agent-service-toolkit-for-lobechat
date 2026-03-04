"""
RAG 分块（Chunking）核心工具。

当前提供：
- 基于 LlamaIndex SentenceSplitter 的固定窗口分块策略；
- 基于 HierarchicalNodeParser 的父子分块策略（叶子向量索引 + 父节点上下文）。
"""

from __future__ import annotations

import tiktoken
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.node_parser.relational.hierarchical import get_leaf_nodes

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
        # 父子分块策略：使用 HierarchicalNodeParser 生成两层节点。
        #
        # 在「摄入阶段」我们会显式调用同样的配置，将 parent/child
        # 全部写入 docstore，仅对 leaf nodes 建立向量索引。
        # 这里保留返回 parser 的接口，便于未来在需要时直接用于
        # LlamaIndex 的 transformations 管线（向后兼容）。
        base_size = settings.RAG_CHUNK_SIZE
        parent_size = max(base_size * 3, base_size + settings.RAG_CHUNK_OVERLAP)
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_size, base_size]
        )
        return [parser]

    # 默认 simple 策略：单层 SentenceSplitter 固定窗口
    splitter = build_default_sentence_splitter()
    return [splitter]


def build_parent_child_nodes(documents: list) -> tuple[list, list]:
    """
    使用 HierarchicalNodeParser 将 Document 列表切分为父子两层节点。

    返回:
        (leaf_nodes, all_nodes)
        - leaf_nodes: 叶子节点列表，仅这些节点会进入向量索引；
        - all_nodes:  包含父子在内的所有节点，用于写入 docstore，
                      便于检索阶段按 parent 展开上下文。
    """
    base_size = settings.RAG_CHUNK_SIZE
    parent_size = max(base_size * 3, base_size + settings.RAG_CHUNK_OVERLAP)
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[parent_size, base_size]
    )
    all_nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    return leaf_nodes, all_nodes


__all__ = [
    "build_default_sentence_splitter",
    "build_chunking_transformations",
    "build_parent_child_nodes",
]

