"""
RAG Query 改写 / HyDE 子模块。

HyDE（Hypothetical Document Embeddings）的核心思想：
- 先让 LLM 根据用户问题生成一段「假想文档」或多种等价表述；
- 对这些假想文档进行向量化，用于检索时作为查询向量；
- 相比直接对原始 query 向量化，更容易命中语义相关但表述差异较大的文档。

本子模块负责：
- Query 改写策略（是否启用、多路改写的数量与并发）；
- 与 core.get_model 的异步集成（ainvoke）；
- 将改写结果封装为统一的数据模型（见 schema_hyde.py）。
"""

from .hyde_generator import generate_hyde_variants

__all__ = [
    "generate_hyde_variants",
]

