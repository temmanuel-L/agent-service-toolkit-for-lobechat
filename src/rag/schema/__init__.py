"""
RAG 子模块的数据模型包。

按照职责拆分为多个 schema_xxx.py 文件，例如：
- schema_parsing.py   : 原始文件解析与文档级元数据
- schema_chunking.py  : 分块/节点级数据结构
- schema_hyde.py      : Query 改写与 HyDE
- schema_search.py    : 检索请求/结果
- schema_rerank.py    : 重排相关数据结构
- segments_postprocess.py : 检索结果后处理
"""

from .schema_parsing import DocumentMetadata, ParsedDocument

__all__ = [
    "DocumentMetadata",
    "ParsedDocument",
]

