"""
RAG 解析与文档级元数据的数据模型。

设计目标：
- 为不同格式的原始文件提供统一的、高度结构化的元数据视图；
- 在「解析 → 分块」之间就确定好通用字段，并在后续所有 chunk 的 metadata 中复用；
- 允许部分字段缺失（例如某些格式拿不到作者），但字段本身保持稳定存在。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    单个物理文档（例如一个 PDF/Word 文件）的通用元数据。

    注意：
    - 各字段尽量保持「语义稳定」，即使底层存储或解析方式变化也不改名。
    - 尽量在解析阶段就把能拿到的信息补齐；拿不到的字段保持为 None。
    """

    # 基本定位信息
    kb_id: Optional[str] = Field(
        default=None,
        description="所属知识库 ID（即 Qdrant collection 名称），可选。",
    )
    file_name: str = Field(
        description="原始文件名（不含路径）。",
        examples=["contract_v1.pdf"],
    )
    file_path: Optional[str] = Field(
        default=None,
        description="解析时的本地临时路径，仅用于调试和日志，不保证长期有效。",
    )
    file_url: Optional[str] = Field(
        default=None,
        description="文件的远程 URL（如 MinIO/S3 预签名地址），若有。",
    )
    file_type: str = Field(
        description="文件类型/扩展名（如 pdf/docx/xlsx），统一为小写不带点。",
        examples=["pdf", "docx", "pptx", "txt"],
    )

    # 内容与语义相关元数据
    doc_title: Optional[str] = Field(
        default=None,
        description="文档标题（从元数据或首页内容/文件名中推断）。",
    )
    author: Optional[str] = Field(
        default=None,
        description="作者（若底层格式支持则尝试解析，否则为 None）。",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="文档创建时间（优先使用文件元数据，其次为解析/摄入时间）。",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="文档最近更新时间（若可用）。",
    )
    summary: Optional[str] = Field(
        default=None,
        description="文档整体摘要，可由后续批处理/离线任务填充。",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="与文档相关的关键词/标签列表。",
    )

    # 结构与组织信息（可选：目录、章节等）
    table_of_contents: Optional[Dict[str, Any]] = Field(
        default=None,
        description="文档目录结构，格式不限（例如树形结构），后续可逐步完善。",
    )

    # 预留扩展字段：用于存放解析器特有的信息，避免 schema 频繁变更
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="解析器特定或临时性的额外字段（例如分页信息、版本号等）。",
    )


class ParsedDocument(BaseModel):
    """
    解析阶段输出的统一结构：一个逻辑文档 + 其元数据。

    对接上游：
    - 上游可以是 LlamaIndex 的 Document，也可以是自定义的文本块；
    - 对于后续分块/入库来说，只要 text + metadata 完整即可。
    """

    text: str = Field(
        description="文档的原始文本内容（尚未分块）。",
    )
    metadata: DocumentMetadata = Field(
        description="与该文档绑定的通用元数据。",
    )


__all__ = [
    "DocumentMetadata",
    "ParsedDocument",
]

