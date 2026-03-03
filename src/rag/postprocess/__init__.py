"""
RAG 检索结果后处理（Postprocess）子模块。

职责边界：
- 将检索 + 重排后得到的少量 segment 进行拼装与截断；
- 负责 token/字符级长度控制、分段标记（如 [Knowledge Segment N]）、回退策略等；
- 对上游工具（如 SearchKnowledgeTool）提供统一的「可直接给 LLM 使用的上下文字符串」。

当前的大部分后处理逻辑分散在 rag.utils 和 rag.service.query_knowledge 中，
后续会迁移至此处，并统一通过 segments_postprocess.py 中的数据模型约束。
"""

__all__: list[str] = []

