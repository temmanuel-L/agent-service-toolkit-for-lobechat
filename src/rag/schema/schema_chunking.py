"""
RAG 分块（Chunking）阶段的数据模型。

这些模型用于描述：
- 分块策略配置（固定窗口、父子分块等）；
- 分块后的逻辑单元（Chunk），以及与 DocumentMetadata 的关联关系。

当前仅提供骨架定义，具体字段会在迁移分块逻辑时逐步完善。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schema_parsing import DocumentMetadata


class ChunkingConfig(BaseModel):
    """分块策略的通用配置。"""

    strategy: str = Field(
        default="fixed",
        description="分块策略名称，例如 fixed / parent_child 等。",
    )
    chunk_size: int = Field(
        default=512,
        ge=1,
        description="基础 chunk 大小（单位：token 或字符，具体由实现决定）。",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="相邻 chunk 的重叠大小。",
    )
    enable_parent_child: bool = Field(
        default=False,
        description="是否启用父子分块策略（小块检索 + 大块提供上下文）。",
    )


class Chunk(BaseModel):
    """
    分块后的最小检索单元。

    注意：
    - 这里的 text 对应的是将来存入向量库/Qdrant 的内容；
    - metadata 会在解析阶段的 DocumentMetadata 基础上附加 chunk 级别的信息（如页码、段落号）。
    """

    text: str = Field(description="chunk 文本内容。")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="chunk 级别的元数据，至少应包含 DocumentMetadata 中的字段副本。",
    )
    document_metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="可选的文档级元数据对象引用，便于在应用层直接访问结构化信息。",
    )


__all__ = [
    "ChunkingConfig",
    "Chunk",
]

