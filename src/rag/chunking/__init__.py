"""
RAG 分块（Chunking）子模块。

职责边界：
- 接收解析后的文档（包含统一的 DocumentMetadata）；
- 根据策略（固定窗口、父子分块等）生成 chunk / node 级结构；
- 不直接关心底层向量库，只产出结构化文本块与其元数据。

当前提供：
- build_default_sentence_splitter: 基于 settings 的固定窗口分块器构造函数；
- build_chunking_transformations: 返回给 LlamaIndex 使用的 transformations 列表（预留父子分块扩展点）；
- build_parent_child_nodes: 基于 HierarchicalNodeParser 生成父子节点（叶子向量索引 + 父节点上下文）；
- build_simple_nodes: simple 策略下的统一 token 分块入口；
- build_title_aware_nodes: 标题感知 + token 级两阶段分块策略。
"""

from .core import (
    build_default_sentence_splitter,
    build_chunking_transformations,
    build_parent_child_nodes,
    build_simple_nodes,
    build_title_aware_nodes,
)

__all__ = [
    "build_default_sentence_splitter",
    "build_chunking_transformations",
    "build_parent_child_nodes",
    "build_simple_nodes",
    "build_title_aware_nodes",
]

