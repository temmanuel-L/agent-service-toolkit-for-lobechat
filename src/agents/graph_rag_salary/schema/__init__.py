# -*- coding: utf-8 -*-
"""薪酬知识图谱本体：节点、关系、别名。"""

from agents.graph_rag_salary.schema.aliases import (
    INDUSTRY_ALIASES,
    INDUSTRY_ALIAS_LOOKUP,
    JOB_ALIASES,
    JOB_ALIAS_LOOKUP,
    SALARY_LEVEL_ALIASES,
    SALARY_LEVEL_ALIAS_LOOKUP,
    SALARY_LEVEL_SORT_ORDER,
    SKILL_ALIASES,
    SKILL_ALIAS_LOOKUP,
    SUB_SECTOR_ALIASES,
    SUB_SECTOR_ALIAS_LOOKUP,
    SUB_SECTOR_HINTS,
    SUB_SECTOR_PARENT_INDUSTRY,
    extract_region_from_text,
    is_sub_sector_hint,
    normalize_salary_level,
    normalize_sub_sector_hint,
    parent_industry_for_sub_sector,
    resolve_alias,
)
from agents.graph_rag_salary.schema.nodes import NODE_TYPES
from agents.graph_rag_salary.schema.relationships import PATTERNS, RELATIONSHIP_TYPES

__all__ = [
    "NODE_TYPES",
    "RELATIONSHIP_TYPES",
    "PATTERNS",
    "INDUSTRY_ALIASES",
    "INDUSTRY_ALIAS_LOOKUP",
    "SUB_SECTOR_ALIASES",
    "SUB_SECTOR_ALIAS_LOOKUP",
    "SUB_SECTOR_PARENT_INDUSTRY",
    "SUB_SECTOR_HINTS",
    "JOB_ALIASES",
    "JOB_ALIAS_LOOKUP",
    "SKILL_ALIASES",
    "SKILL_ALIAS_LOOKUP",
    "SALARY_LEVEL_ALIASES",
    "SALARY_LEVEL_ALIAS_LOOKUP",
    "SALARY_LEVEL_SORT_ORDER",
    "resolve_alias",
    "is_sub_sector_hint",
    "normalize_sub_sector_hint",
    "parent_industry_for_sub_sector",
    "normalize_salary_level",
    "extract_region_from_text",
]


def format_ontology_summary(max_node_desc_chars: int = 80) -> str:
    """轻量本体摘要，供 L1 prompt 等使用，避免塞入整份 schema JSON。"""
    lines: list[str] = ["节点类型:"]
    for node in NODE_TYPES:
        label = node.get("label", "")
        desc = (node.get("description") or "").replace("\n", " ").strip()
        if len(desc) > max_node_desc_chars:
            desc = desc[:max_node_desc_chars] + "…"
        lines.append(f"- {label}: {desc}" if desc else f"- {label}")
    lines.append("合法关系模式:")
    for src, rel, dst in PATTERNS:
        lines.append(f"- ({src})-[:{rel}]->({dst})")
    return "\n".join(lines)
