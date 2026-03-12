"""
PDF 解析模块（使用 docling）。

当安装 pdf-docling optional 依赖时，用于解析含图片、表格的 PDF，
使用 docling 内置 SmolVLM 生成图片描述，导出为 Markdown 后转为 LlamaIndex Document。

依赖：pip install .[pdf-docling]
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core.schema import Document

from rag.schema.schema_parsing import DocumentMetadata
from utils.log_utils import get_logger

logger = get_logger(__name__)


def parse_pdf_to_documents(
    file_path: str,
    *,
    kb_id: Optional[str] = None,
    file_name: Optional[str] = None,
    file_url: Optional[str] = None,
    do_picture_description: bool = True,
) -> Tuple[List[Document], DocumentMetadata]:
    """
    使用 docling 解析 PDF，支持表格、OCR、图片描述。

    Args:
        file_path: 本地 PDF 文件路径。
        kb_id: 所属知识库 ID，可选。
        file_name: 原始文件名，可选。
        file_url: 文件远程 URL，可选。
        do_picture_description: 是否用 SmolVLM 生成图片描述，默认 True。

    Returns:
        (documents, metadata): 与 parse_file_to_documents 相同格式。
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        smolvlm_picture_description,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    inferred_file_name = file_name or os.path.basename(file_path)

    pipeline_options = PdfPipelineOptions(
        generate_page_images=False,
        generate_picture_images=False,
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=True,
        do_picture_description=do_picture_description,
    )
    if do_picture_description:
        pipeline_options.picture_description_options = smolvlm_picture_description

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(file_path)
    markdown_text = result.document.export_to_markdown()

    if not markdown_text or not markdown_text.strip():
        logger.warning(f"[pdf_parsing] No content extracted from {inferred_file_name}")
        meta = DocumentMetadata(
            kb_id=kb_id,
            file_name=inferred_file_name,
            file_path=file_path,
            file_url=file_url,
            file_type="pdf",
        )
        return [], meta

    # 提取标题（从首行或文件名）
    first_lines = [l.strip() for l in markdown_text.split("\n")[:10] if l.strip()]
    doc_title: Optional[str] = None
    for line in first_lines[:5]:
        if 5 < len(line) < 300 and not (line and line[0].isdigit()):
            doc_title = line
            break
    if not doc_title and inferred_file_name:
        doc_title = (
            Path(inferred_file_name).stem.replace("_", " ").replace("-", " ").strip()
            or None
        )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    try:
        stat = os.stat(file_path)
        created_at = datetime.fromtimestamp(stat.st_ctime)
        updated_at = datetime.fromtimestamp(stat.st_mtime)
    except Exception:
        pass

    author: Optional[str] = None
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        meta = reader.metadata
        if meta and getattr(meta, "author", None):
            a = str(meta.author).strip()
            if a:
                author = a
    except Exception:
        pass

    metadata = DocumentMetadata(
        kb_id=kb_id,
        file_name=inferred_file_name,
        file_path=file_path,
        file_url=file_url,
        file_type="pdf",
        doc_title=doc_title,
        author=author,
        created_at=created_at,
        updated_at=updated_at,
    )
    base_meta_dict = metadata.model_dump(exclude_none=True)

    doc = Document(text=markdown_text, metadata=base_meta_dict)
    doc.excluded_embed_metadata_keys = []
    doc.excluded_llm_metadata_keys = []

    if metadata.doc_title:
        logger.info("[pdf_parsing] 文档标题: '%s'", metadata.doc_title[:80])

    logger.info(
        f"[pdf_parsing] Extracted 1 document (docling) for file={inferred_file_name}, kb_id={kb_id}"
    )

    return [doc], metadata


__all__ = ["parse_pdf_to_documents"]
