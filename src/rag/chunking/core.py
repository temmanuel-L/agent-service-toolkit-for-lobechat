"""
RAG 分块（Chunking）核心工具。

当前提供：
- 基于 LlamaIndex SentenceSplitter 的固定窗口分块策略；
- 基于 HierarchicalNodeParser 的父子分块策略（叶子向量索引 + 父节点上下文）；
- 标题感知 + token 级两阶段分块策略（对标 RagFlow 的 Token + Title Chunker）；
- RecursiveCharacterTextSplitter（LangChain）经 LangchainNodeParser 接入，支持 token/sentence/recursive 三种 splitter。
"""

from __future__ import annotations

import re

import tiktoken
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    LangchainNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.node_parser.relational.hierarchical import get_leaf_nodes
from llama_index.core.schema import Document

from core.settings import settings

# RecursiveCharacterTextSplitter 分隔符：CJK 混合文档
_SEPARATORS_CJK = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    ". ",
    "! ",
    "? ",
    "；",
    "; ",
    "，",
    ", ",
    " ",
    "",
]
# 非 CJK 文档
_SEPARATORS_DEFAULT = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _resolve_parent_size(base_size: int) -> int:
    ratio = max(1, int(getattr(settings, "RAG_FATHER_SON_RATIO", 3) or 3))
    return max(base_size * ratio, base_size + settings.RAG_CHUNK_OVERLAP)


def _estimate_cjk_ratio(text: str, sample_size: int = 2000) -> float:
    """采样估算 CJK 字符占比，用于 RecursiveCharacterTextSplitter 的字符换算。"""
    if not text:
        return 0.0
    sample = text[:sample_size]
    cjk_count = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    return cjk_count / len(sample) if sample else 0.0


def _build_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int | None = None,
    *,
    sample_text: str = "",
) -> LangchainNodeParser:
    """
    构建 RecursiveCharacterTextSplitter（经 LangchainNodeParser 包装）。

    chunk_size/overlap 为 token 级目标；按 CJK 比例换算为字符数，纯字符级切分，不调用 tiktoken。
    sample_text 为空时默认按 CJK（1.5 chars/token）换算，适合中英混合文档。
    """
    if chunk_overlap is None:
        chunk_overlap = settings.RAG_CHUNK_OVERLAP
    cjk_ratio = _estimate_cjk_ratio(sample_text) if sample_text else 0.5
    is_cjk = cjk_ratio >= 0.30
    chars_per_token = 1.5 if is_cjk else 4.0
    chunk_size_chars = int(chunk_size * chars_per_token)
    overlap_chars = int(chunk_overlap * chars_per_token)
    separators = _SEPARATORS_CJK if is_cjk else _SEPARATORS_DEFAULT

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    lc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=separators,
        length_function=len,
    )
    return LangchainNodeParser(lc_splitter=lc_splitter)


def _build_sentence_splitter(
    chunk_size: int,
    chunk_overlap: int,
    encoding,
) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=encoding.encode,
    )


def _build_token_splitter(
    chunk_size: int,
    chunk_overlap: int | None = None,
) -> TokenTextSplitter:
    """
    内部工具：根据给定的 chunk_size/overlap 构建 TokenTextSplitter。

    说明：
    - TokenTextSplitter 仍是“最大长度约束”，最后一个块可能小于 chunk_size；
    - 相比 SentenceSplitter（句子优先），TokenTextSplitter 更接近固定 token 窗口。
    """
    if chunk_overlap is None:
        chunk_overlap = settings.RAG_CHUNK_OVERLAP
    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=encoding.encode,
        backup_separators=["\n", " "],
    )
    return splitter


def _build_splitter(
    chunk_size: int,
    chunk_overlap: int | None = None,
    *,
    sample_text: str = "",
):
    if chunk_overlap is None:
        chunk_overlap = settings.RAG_CHUNK_OVERLAP
    splitter_type = (getattr(settings, "RAG_SPLITTER_TYPE", "token") or "token").lower()
    if splitter_type == "sentence":
        encoding = tiktoken.get_encoding("cl100k_base")
        return _build_sentence_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding=encoding,
        )
    if splitter_type == "recursive":
        return _build_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sample_text=sample_text,
        )
    return _build_token_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_default_sentence_splitter():
    """
    基于当前 settings 构建默认分块器（token/sentence）。

    - 使用 cl100k_base 的 tiktoken 编码做 token 级分块；
    - chunk_size / chunk_overlap 来自 RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP；
    - 分块器类型由 RAG_SPLITTER_TYPE 控制（token / sentence）。
    """
    return _build_splitter(chunk_size=settings.RAG_CHUNK_SIZE)


def build_chunking_transformations() -> list:
    """
    根据 RAG_CHUNKING_STRATEGY 返回用于 LlamaIndex 的 transformations 列表。

    当前支持：
    - "simple": 仅使用 TokenTextSplitter 做固定窗口分块（默认）；
    - "parent_child": 使用 HierarchicalNodeParser 做父子分块（大块作为 parent，小块作为 child）。

    注意：更高级的「标题感知」分块在 build_title_aware_nodes 中实现，
    不通过 transformations 暴露，避免对现有 LlamaIndex 管线造成破坏性变更。
    """
    strategy = (getattr(settings, "RAG_CHUNKING_STRATEGY", "simple") or "simple").lower()

    if strategy == "parent_child":
        base_size = settings.RAG_CHUNK_SIZE
        parent_size = _resolve_parent_size(base_size)
        parser_ids = ["pc_parent", "pc_child"]
        parser_map = {
            parser_ids[0]: _build_splitter(parent_size, settings.RAG_CHUNK_OVERLAP),
            parser_ids[1]: _build_splitter(base_size, settings.RAG_CHUNK_OVERLAP),
        }
        parser = HierarchicalNodeParser.from_defaults(
            node_parser_ids=parser_ids,
            node_parser_map=parser_map,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
        return [parser]

    splitter = build_default_sentence_splitter()
    return [splitter]


def build_parent_child_nodes(
    documents: list,
    *,
    title_aware: bool | None = None,
) -> tuple[list, list]:
    """
    使用 HierarchicalNodeParser 将 Document 列表切分为父子两层节点。

    若 title_aware 为 True，则在父子分块前先按标题粗粒度切分为若干 section。

    返回:
        (leaf_nodes, all_nodes)
    """
    expanded_docs: list[Document] = []
    if title_aware is None:
        title_aware = getattr(settings, "RAG_CHUNKING_TITLE_AWARE", False)
    if title_aware:
        for doc in documents:
            if isinstance(doc, Document):
                expanded_docs.extend(_split_document_by_headings(doc))
            else:
                expanded_docs.append(doc)
    else:
        expanded_docs = list(documents)

    base_size = settings.RAG_CHUNK_SIZE
    parent_size = _resolve_parent_size(base_size)
    parser_ids = ["pc_parent", "pc_child"]
    sample_text = ""
    if expanded_docs:
        sample_text = " ".join(
            (getattr(d, "text", "") or "")[:3000] for d in expanded_docs[:3]
        )
    parser_map = {
        parser_ids[0]: _build_splitter(
            parent_size, settings.RAG_CHUNK_OVERLAP, sample_text=sample_text
        ),
        parser_ids[1]: _build_splitter(
            base_size, settings.RAG_CHUNK_OVERLAP, sample_text=sample_text
        ),
    }
    parser = HierarchicalNodeParser.from_defaults(
        node_parser_ids=parser_ids,
        node_parser_map=parser_map,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    )
    all_nodes = parser.get_nodes_from_documents(expanded_docs)
    leaf_nodes = get_leaf_nodes(all_nodes)
    return leaf_nodes, all_nodes


def _looks_like_heading(line: str) -> bool:
    """
    简单启发式：判断一行文本是否更像「小节标题」而不是正文。

    设计思路（对标 RagFlow 的 Title Chunker，但保持轻量级、无额外依赖）：
    - 行长适中（3~80 字符），排除空行、单双字噪声和超长段落；
    - 恰好 3 字符的行：要求全部是中文或字母
      （匹配中文常见章节标题如"半导体""管理层""薪酬表"，排除数字/标点噪声）；
    - 不能以句号/问号/感叹号结尾（排除完整句子）；
    - 标点符号占比不过高（标题通常标点较少）；
    - 排除包含大量数字的行（表格数据行）。
    """
    text = line.strip()
    if not text:
        return False

    length = len(text)
    if length < 3 or length > 80:
        return False

    # 恰好 3 字符的行：必须全是中文或字母（匹配"半导体""管理层"，排除数字/标点噪声）
    if length == 3:
        if not re.fullmatch(r"[\u4e00-\u9fffa-zA-Z]+", text):
            return False

    if text[-1] in "。？！!?":
        return False

    puncts = re.findall(r"[，,；;。？！?!]", text)
    if len(puncts) >= 3:
        return False

    # 表格行：包含大量数字
    digit_chars = re.findall(r"\d", text)
    if len(digit_chars) >= 6:
        return False
    # 包含 2 个以上数值区间模式
    ranges = re.findall(r"\d[\d,]*\s*[-\u2013\u2014]\s*\d[\d,]*", text)
    if len(ranges) >= 2:
        return False

    return True


def _split_document_by_headings(document: Document) -> list[Document]:
    """
    粗粒度标题感知切分：
    - 按行扫描文档内容；
    - 遇到「疑似标题」行时开启新的 section；
    - 每个 section 形成一个新的 Document，并注入 section_title / section_index 元数据。
    """
    text = getattr(document, "text", "") or ""
    if not text:
        return [document]

    lines = text.splitlines()

    MAX_HEADINGS_PER_DOC = 200
    heading_count = 0
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and _looks_like_heading(stripped):
            heading_count += 1
            if heading_count > MAX_HEADINGS_PER_DOC:
                return [document]

    sections: list[tuple[str | None, str]] = []
    current_lines: list[str] = []
    current_heading: str | None = None
    first_heading: str | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and _looks_like_heading(stripped):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
                current_lines = []
            if first_heading is None:
                first_heading = stripped
            current_heading = stripped
        current_lines.append(raw_line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    if not sections or (len(sections) == 1 and sections[0][0] is None):
        return [document]

    base_meta = getattr(document, "metadata", {}) or {}
    section_docs: list[Document] = []
    for idx, (heading, section_text) in enumerate(sections):
        meta = {**base_meta}
        meta["section_index"] = idx
        if heading:
            # 将首标题（页面级标题）作为前缀传播到所有子 section，
            # 例如页面标题"半导体"下的子 section"管理层" → "半导体 > 管理层"，
            # 保证所有半导体薪酬数据 chunk 的 section_title 都包含"半导体"。
            if first_heading and heading != first_heading:
                meta["section_title"] = f"{first_heading} > {heading}"
            else:
                meta["section_title"] = heading
        section_docs.append(
            Document(
                text=section_text,
                metadata=meta,
            )
        )

    return section_docs


def build_title_aware_nodes(
    documents: list,
    *,
    chunk_size: int | None = None,
) -> list:
    """
    标题感知 + token 级分块的两阶段策略（对标 RagFlow 的 Token + Title Chunker）：

    1. 先按「疑似小节标题」切分 Document，得到若干 section 级 Document；
    2. 再对每个 section 使用 SentenceSplitter 做 token 级分块（带 overlap）。
    """
    if chunk_size is None:
        splitter = build_default_sentence_splitter()
    else:
        splitter = _build_splitter(
            chunk_size=chunk_size,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
    expanded_docs: list[Document] = []
    for doc in documents:
        if isinstance(doc, Document):
            expanded_docs.extend(_split_document_by_headings(doc))
        else:
            expanded_docs.append(doc)

    nodes = splitter.get_nodes_from_documents(expanded_docs)
    return nodes


def build_simple_nodes(
    documents: list,
    *,
    title_aware: bool | None = None,
) -> list:
    """
    simple 策略统一分块入口（token / sentence / recursive）。

    - title_aware=False: 直接按 RAG_SPLITTER_TYPE 分块；
    - title_aware=True : 先按标题切 section，再按 RAG_SPLITTER_TYPE 分块。
    """
    if title_aware is None:
        title_aware = getattr(settings, "RAG_CHUNKING_TITLE_AWARE", False)
    if title_aware:
        return build_title_aware_nodes(documents, chunk_size=settings.RAG_CHUNK_SIZE)
    sample_text = ""
    if documents:
        sample_text = " ".join(
            (getattr(d, "text", "") or "")[:3000] for d in documents[:3]
        )
    splitter = _build_splitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        sample_text=sample_text,
    )
    return splitter.get_nodes_from_documents(documents)


__all__ = [
    "build_default_sentence_splitter",
    "build_chunking_transformations",
    "build_parent_child_nodes",
    "build_title_aware_nodes",
    "build_simple_nodes",
]
