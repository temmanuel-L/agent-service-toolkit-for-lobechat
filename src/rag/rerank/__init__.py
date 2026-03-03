"""
RAG 重排（Rerank）子模块。

职责边界：
- 聚焦于「候选段落列表 → 重新排序后的段落列表」这一变换；
- 与具体的 Rerank 服务（如 TEI、智谱等）解耦，通过适配器集成；
- 可以在内部处理分数归一化、阈值裁剪等细节，对上层隐藏实现复杂度。
"""

from .core import rerank_nodes

__all__ = [
    "rerank_nodes",
]

