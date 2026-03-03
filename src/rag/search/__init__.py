"""
RAG 检索（Search）子模块。

职责边界：
- 统一封装向量检索、关键词检索（BM25）及其融合逻辑；
- 负责构造针对 Qdrant 的检索请求（包含 metadata filter）；
- 对上层只暴露与存储无关的 SearchRequest / SearchResult 数据模型。
"""

from .core import hybrid_search_single_kb

__all__ = [
    "hybrid_search_single_kb",
]

