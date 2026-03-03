"""
文件解析与文档级元数据注入。

说明：
- 目前基于 LlamaIndex 的 SimpleDirectoryReader，一次性支持 PDF/DOCX/PPTX/HTML 等常见格式；
- 若后续需要针对特定格式做更细致的解析（如表格结构、图片 OCR），可以在本模块内按格式拆分实现，
  但对外仍暴露统一的 parse_file_to_documents 接口。
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

from llama_index.core import SimpleDirectoryReader

from rag.schema.schema_parsing import DocumentMetadata
from utils.log_utils import get_logger

logger = get_logger(__name__)


def _infer_file_type(file_path: str, file_name: Optional[str]) -> str:
    """根据文件路径/文件名推断文件类型（扩展名），统一为小写不带点。"""
    name = file_name or os.path.basename(file_path)
    ext = os.path.splitext(name)[1].lower()
    if ext.startswith("."):
        ext = ext[1:]
    return ext or "unknown"


def _extract_doc_title(file_path: str, documents: list, file_name: Optional[str]) -> str:
    """
    从文档中提取标题，用于注入到每个 chunk 的元数据。

    提取优先级：
    1. PDF 元数据中的 title 字段（pypdf）
    2. 文档首页前若干行中最适合做标题的行（短、非空、非页码）
    3. 文件名去扩展名（兜底）
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ---- 策略 1: PDF 元数据 ----
    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            meta = reader.metadata
            if meta and meta.title and meta.title.strip():
                title = meta.title.strip()
                logger.debug(f"从 PDF 元数据提取标题: '{title[:80]}'")
                return title
        except Exception:
            # PDF 元数据不可用，继续尝试其他策略
            pass

    # ---- 策略 2: 从首页内容中启发式提取 ----
    if documents:
        first_text = documents[0].text[:1000]
        lines = [line.strip() for line in first_text.split("\n") if line.strip()]
        # 跳过纯数字行（页码）、过长行（正文段落）
        for line in lines[:8]:
            # 好的标题特征：长度适中（5-300 字符），不以数字开头（排除页码）
            if 5 < len(line) < 300 and not line[0].isdigit():
                logger.debug(f"从首页内容提取标题: '{line[:80]}'")
                return line

    # ---- 策略 3: 从文件名推断 ----
    if file_name:
        name_without_ext = os.path.splitext(file_name)[0]
        # 替换常见分隔符为空格
        title = re.sub(r"[_\-]+", " ", name_without_ext).strip()
        if title:
            logger.debug(f"从文件名推断标题: '{title[:80]}'")
            return title

    return ""


def parse_file_to_documents(
    file_path: str,
    *,
    kb_id: Optional[str] = None,
    file_name: Optional[str] = None,
    file_url: Optional[str] = None,
) -> Tuple[List, DocumentMetadata]:
    """
    解析本地文件为文档列表，并在「分块前」统一注入文档级元数据。

    Args:
        file_path: 本地临时文件路径。
        kb_id:     所属知识库 ID（Qdrant collection 名称），可选。
        file_name: 原始文件名（不含路径），可选；未提供则从 file_path 推断。
        file_url:  文件远程 URL（如 S3 预签名地址），用于追踪来源。

    Returns:
        (documents, metadata):
            - documents: LlamaIndex Document 列表，已填充 metadata / excluded_*_metadata_keys
            - metadata:  统一的 DocumentMetadata 对象，供上层记录或持久化
    """
    logger.info(f"[parsing] Loading document from file: {file_path}")
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    if not documents:
        logger.warning(f"[parsing] No content extracted from {file_name or file_path}")
        # 构造一个最小化的元数据对象，便于上层记录失败原因
        meta = DocumentMetadata(
            kb_id=kb_id,
            file_name=file_name or os.path.basename(file_path),
            file_path=file_path,
            file_url=file_url,
            file_type=_infer_file_type(file_path, file_name),
        )
        return [], meta

    # 文档级元数据（作者/时间等部分字段目前可能为空，后续可逐步补全）
    inferred_file_name = file_name or os.path.basename(file_path)
    file_type = _infer_file_type(file_path, inferred_file_name)
    doc_title = _extract_doc_title(file_path, documents, inferred_file_name)

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    try:
        stat = os.stat(file_path)
        # Windows 下 st_ctime 是「创建时间」，类 Unix 下是 inode 变更时间；这里仅作为近似参考
        created_at = datetime.fromtimestamp(stat.st_ctime)
        updated_at = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        # 文件可能位于只读或特殊文件系统，失败时不影响主流程
        pass

    # 尝试补充作者信息（仅对 PDF 等支持元数据的格式有效）
    author: Optional[str] = None
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            meta = reader.metadata
            if meta and getattr(meta, "author", None):
                a = str(meta.author).strip()
                if a:
                    author = a
        except Exception:
            # 作者元数据获取失败不影响主流程
            pass

    metadata = DocumentMetadata(
        kb_id=kb_id,
        file_name=inferred_file_name,
        file_path=file_path,
        file_url=file_url,
        file_type=file_type,
        doc_title=doc_title or None,
        author=author,
        created_at=created_at,
        updated_at=updated_at,
    )
    base_meta_dict = metadata.model_dump(exclude_none=True)

    # 将统一的文档级元数据注入到每一个文档对象中
    for doc in documents:
        existing = getattr(doc, "metadata", {}) or {}
        # 统一元数据优先级更高，避免同名字段冲突
        merged = {**existing, **base_meta_dict}
        doc.metadata = merged

        # 确保所有元数据参与 embedding 和 LLM 上下文（不排除任何 key）
        doc.excluded_embed_metadata_keys = []
        doc.excluded_llm_metadata_keys = []

    if metadata.doc_title:
        logger.info(f"[parsing] 文档标题: '{metadata.doc_title[:80]}'")

    logger.info(
        f"[parsing] Extracted {len(documents)} document pages/segments for "
        f"file={inferred_file_name}, kb_id={kb_id}"
    )

    return documents, metadata


__all__ = [
    "parse_file_to_documents",
]

