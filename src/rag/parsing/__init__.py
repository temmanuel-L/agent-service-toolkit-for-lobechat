"""
RAG 原始数据解析子模块。

职责边界：
- 接收「物理文件」或本地临时文件路径；
- 使用扩展性良好的解析器（当前基于 LlamaIndex SimpleDirectoryReader）抽取文本；
- 在 **分块之前** 统一生成并注入文档级元数据（作者、标题、时间等）；
- 返回带有完整 metadata 的文档对象，供后续 chunking / 向量化复用。
"""

from .file_parsing import parse_file_to_documents

__all__ = [
    "parse_file_to_documents",
]

