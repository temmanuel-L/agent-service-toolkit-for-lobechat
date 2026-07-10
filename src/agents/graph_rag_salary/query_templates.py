# -*- coding: utf-8 -*-
"""参数化 Cypher 检索模板 A/B/C/D + Chunk 兜底 E。

业务结果不做条数截断：岗位/薪资/技能/趋势全量返回，避免 overview 等意图漏岗。
同一岗位可挂多条 SalaryRange（全国平均/华东/华北/华南），检索侧按 level 去重后全量返回。
"""

from __future__ import annotations

import json
from typing import Any

from agents.graph_rag_salary.schema.aliases import (
    SALARY_LEVEL_SORT_ORDER,
    normalize_salary_level,
)
from agents.graph_rag_salary.neo4j_client import get_salary_neo4j_database
from core import get_embedding_model


def _salary_band_width(s: dict) -> float:
    try:
        return float(s.get("max_value") or s.get("max") or 0) - float(
            s.get("min_value") or s.get("min") or 0
        )
    except (TypeError, ValueError):
        return float("inf")


def _salary_min(s: dict) -> float:
    try:
        return float(s.get("min_value") or s.get("min") or 0)
    except (TypeError, ValueError):
        return 0.0


def _salary_max(s: dict) -> float:
    try:
        return float(s.get("max_value") or s.get("max") or 0)
    except (TypeError, ValueError):
        return 0.0


def _level_sort_key(level: str) -> tuple:
    try:
        idx = SALARY_LEVEL_SORT_ORDER.index(level)
    except ValueError:
        idx = len(SALARY_LEVEL_SORT_ORDER)
    return (idx, level)


def dedupe_salaries(salaries: list | None) -> list:
    """同一岗位多条薪资：按规范化 level+(min,max) 去重，保留全部不同地区档。"""
    items = [x for x in (salaries or []) if x is not None]
    if not items:
        return []

    def as_dict(x) -> dict:
        if isinstance(x, dict):
            return dict(x)
        return {
            "min_value": getattr(x, "min_value", None),
            "max_value": getattr(x, "max_value", None),
            "min": getattr(x, "min", None),
            "max": getattr(x, "max", None),
            "level": getattr(x, "level", None),
            "unit": getattr(x, "unit", None),
            "position_title": getattr(x, "position_title", None),
        }

    by_level: dict[str, list[dict]] = {}
    for raw in items:
        d = as_dict(raw)
        level = normalize_salary_level(d.get("level"))
        d["level"] = level or d.get("level") or ""
        if d.get("min") is None and d.get("min_value") is not None:
            d["min"] = d["min_value"]
        if d.get("max") is None and d.get("max_value") is not None:
            d["max"] = d["max_value"]
        by_level.setdefault(level, []).append(d)

    result: list[dict] = []
    seen_exact: set[tuple[str, float, float]] = set()
    for level in sorted(by_level.keys(), key=_level_sort_key):
        cands = by_level[level]
        unique: dict[tuple[float, float], dict] = {}
        for c in cands:
            key = (_salary_min(c), _salary_max(c))
            if key not in unique:
                unique[key] = c
        if len(unique) == 1:
            chosen = next(iter(unique.values()))
        else:
            chosen = min(
                unique.values(),
                key=lambda x: (_salary_band_width(x), _salary_min(x)),
            )
        exact = (level, _salary_min(chosen), _salary_max(chosen))
        if exact in seen_exact:
            continue
        seen_exact.add(exact)
        result.append(chosen)
    return result


# 兼容旧名
pick_preferred_salary = dedupe_salaries


class OntologyQueryTemplates:
    def __init__(self, driver, database: str | None = None):
        self.driver = driver
        self.db = database or get_salary_neo4j_database()

    def _query(self, cypher: str, params: dict | None = None) -> list[dict]:
        result = self.driver.execute_query(
            cypher,
            parameters_=params or {},
            database_=self.db,
        )
        return [dict(r) for r in (result.records or [])]

    # ---------- A: salary_lookup ----------
    def salary_lookup(
        self,
        industry_ids: list[str] | None = None,
        job_ids: list[str] | None = None,
        sub_sector: str | None = None,
        sub_sector_ids: list[str] | None = None,
    ) -> list[dict]:
        industry_ids = industry_ids or []
        job_ids = job_ids or []
        sub_sector_ids = sub_sector_ids or []
        cypher = """
        MATCH (j:JobPosition)
        WHERE (size($job_ids) = 0 OR elementId(j) IN $job_ids)
          AND (
            size($sub_sector_ids) = 0 OR EXISTS {
                MATCH (ss:SubSector)-[:HAS_POSITION]->(j)
                WHERE elementId(ss) IN $sub_sector_ids
            }
          )
          AND (
            $sub_sector IS NULL OR $sub_sector = ''
            OR EXISTS {
                MATCH (ss:SubSector)-[:HAS_POSITION]->(j)
                WHERE toLower(ss.name) CONTAINS toLower($sub_sector)
            }
            OR (j.sub_sector IS NOT NULL AND toLower(j.sub_sector) CONTAINS toLower($sub_sector))
            OR (j.title IS NOT NULL AND toLower(j.title) CONTAINS toLower($sub_sector))
          )
        OPTIONAL MATCH (i_direct:Industry)-[:HAS_POSITION]->(j)
        OPTIONAL MATCH (i_via:Industry)-[:HAS_SUB_SECTOR]->(:SubSector)-[:HAS_POSITION]->(j)
        WITH j,
             [x IN (collect(DISTINCT i_direct) + collect(DISTINCT i_via))
              WHERE x IS NOT NULL
                AND (size($industry_ids) = 0 OR elementId(x) IN $industry_ids) | x] AS industries
        WHERE size($industry_ids) = 0 OR size(industries) > 0
        OPTIONAL MATCH (ss:SubSector)-[:HAS_POSITION]->(j)
        OPTIONAL MATCH (j)-[:HAS_SALARY_RANGE]->(s:SalaryRange)
        OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(k:Skill)
        OPTIONAL MATCH (j)-[:FROM_CHUNK]->(c:Chunk)
        WITH j, industries,
             collect(DISTINCT ss.name) AS sub_sectors,
             collect(DISTINCT s) AS salaries,
             collect(DISTINCT k.name) AS skills,
             collect(DISTINCT c.text) AS evidence
        WITH j, industries, sub_sectors, skills, evidence,
             [x IN salaries WHERE x IS NOT NULL | x] AS all_salaries
        RETURN
            [x IN industries | x.name] AS industry_names,
            sub_sectors,
            j.title AS job_title,
            j.category AS category,
            coalesce(sub_sectors[0], j.sub_sector) AS sub_sector,
            [s IN all_salaries | {
                min: s.min_value,
                max: s.max_value,
                unit: coalesce(s.unit, '千元人民币'),
                level: s.level,
                position_title: s.position_title
            }] AS salary_ranges,
            skills,
            evidence
        """
        rows = self._query(
            cypher,
            {
                "industry_ids": industry_ids,
                "job_ids": job_ids,
                "sub_sector": sub_sector or "",
                "sub_sector_ids": sub_sector_ids,
            },
        )
        for row in rows:
            row["salary_ranges"] = dedupe_salaries(row.get("salary_ranges"))
        return rows

    # ---------- B: compare（应用层对两组各调 A） ----------
    def compare(self, groups: list[dict[str, Any]]) -> list[dict]:
        """
        groups 元素形如:
        {"industry_ids": [...], "job_ids": [...], "sub_sector": "保险",
         "sub_sector_ids": [...], "label": "..."}
        """
        rows = []
        for g in groups:
            items = self.salary_lookup(
                industry_ids=g.get("industry_ids") or [],
                job_ids=g.get("job_ids") or [],
                sub_sector=g.get("sub_sector"),
                sub_sector_ids=g.get("sub_sector_ids") or [],
            )
            if not items and (
                g.get("job_ids") or g.get("industry_ids") or g.get("sub_sector_ids")
            ):
                items = self.salary_lookup(
                    industry_ids=g.get("industry_ids") or [],
                    job_ids=g.get("job_ids") or [],
                    sub_sector=None,
                    sub_sector_ids=None,
                )
            rows.append({
                "group_label": g.get("label") or "",
                "results": items,
            })
        return rows

    # ---------- C: industry_overview ----------
    def industry_overview(
        self,
        industry_ids: list[str],
        sub_sector: str | None = None,
        sub_sector_ids: list[str] | None = None,
    ) -> list[dict]:
        sub_sector_ids = sub_sector_ids or []
        cypher = """
        MATCH (i:Industry)
        WHERE elementId(i) IN $industry_ids
        OPTIONAL MATCH (i)-[:HAS_SUB_SECTOR]->(ss_all:SubSector)
        OPTIONAL MATCH (i)-[:HAS_POSITION]->(j_direct:JobPosition)
        OPTIONAL MATCH (i)-[:HAS_SUB_SECTOR]->(ss:SubSector)-[:HAS_POSITION]->(j_via:JobPosition)
        WITH i, collect(DISTINCT ss_all) AS all_sub_sectors,
             collect(DISTINCT j_direct) + collect(DISTINCT j_via) AS jobs_raw,
             collect(DISTINCT ss) AS via_sub_sectors
        UNWIND (CASE WHEN size(jobs_raw) = 0 THEN [null] ELSE jobs_raw END) AS j
        WITH i, all_sub_sectors, j
        WHERE j IS NULL OR (
            (
                size($sub_sector_ids) = 0 OR EXISTS {
                    MATCH (ss:SubSector)-[:HAS_POSITION]->(j)
                    WHERE elementId(ss) IN $sub_sector_ids
                }
            )
            AND (
                $sub_sector IS NULL OR $sub_sector = ''
                OR EXISTS {
                    MATCH (ss:SubSector)-[:HAS_POSITION]->(j)
                    WHERE toLower(ss.name) CONTAINS toLower($sub_sector)
                }
                OR (j.sub_sector IS NOT NULL AND toLower(j.sub_sector) CONTAINS toLower($sub_sector))
                OR (j.title IS NOT NULL AND toLower(j.title) CONTAINS toLower($sub_sector))
            )
        )
        OPTIONAL MATCH (ss_j:SubSector)-[:HAS_POSITION]->(j)
        OPTIONAL MATCH (j)-[:HAS_SALARY_RANGE]->(s:SalaryRange)
        WITH i, all_sub_sectors, j, collect(DISTINCT ss_j.name) AS j_sub_sectors,
             collect(DISTINCT s) AS salaries
        WITH i, all_sub_sectors, j, j_sub_sectors,
             [x IN salaries WHERE x IS NOT NULL | x] AS all_salaries
        WITH i, all_sub_sectors,
             [p IN collect(DISTINCT {
                title: j.title,
                category: j.category,
                sub_sector: coalesce(j_sub_sectors[0], j.sub_sector),
                salaries: [x IN all_salaries | {
                    min: x.min_value, max: x.max_value, level: x.level
                }]
             }) WHERE p.title IS NOT NULL | p] AS positions
        OPTIONAL MATCH (i)-[:REQUIRES_SKILL]->(k:Skill)
        OPTIONAL MATCH (i)-[:HAS_TREND]->(t:MarketTrend)
        RETURN
            i.name AS industry,
            i.overview AS overview,
            [x IN all_sub_sectors WHERE x IS NOT NULL | x.name] AS sub_sectors,
            $sub_sector AS sub_sector_filter,
            positions,
            collect(DISTINCT k.name) AS skills,
            collect(DISTINCT {
                name: t.trend_name,
                description: t.description
            }) AS trends
        """
        rows = self._query(
            cypher,
            {
                "industry_ids": industry_ids,
                "sub_sector": sub_sector or "",
                "sub_sector_ids": sub_sector_ids,
            },
        )
        for row in rows:
            positions = row.get("positions") or []
            for pos in positions:
                if isinstance(pos, dict):
                    pos["salaries"] = dedupe_salaries(pos.get("salaries"))
        return rows

    # ---------- D: skill_trend ----------
    def skill_trend(
        self,
        skill_ids: list[str] | None = None,
        industry_ids: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[dict]:
        skill_rows = []
        if skill_ids:
            skill_rows = self._query(
                """
                MATCH (k:Skill)
                WHERE elementId(k) IN $skill_ids
                OPTIONAL MATCH (j:JobPosition)-[:REQUIRES_SKILL]->(k)
                OPTIONAL MATCH (i:Industry)-[:REQUIRES_SKILL]->(k)
                RETURN k.name AS skill, k.type AS type,
                       collect(DISTINCT j.title) AS jobs,
                       collect(DISTINCT i.name) AS industries
                """,
                {"skill_ids": skill_ids},
            )
        elif query_text:
            skill_rows = self._query(
                """
                CALL db.index.fulltext.queryNodes('skillNameFulltext', $q)
                YIELD node, score
                WITH node AS k, score
                OPTIONAL MATCH (j:JobPosition)-[:REQUIRES_SKILL]->(k)
                OPTIONAL MATCH (i:Industry)-[:REQUIRES_SKILL]->(k)
                RETURN k.name AS skill, k.type AS type, score,
                       collect(DISTINCT j.title) AS jobs,
                       collect(DISTINCT i.name) AS industries
                ORDER BY score DESC
                """,
                {"q": query_text},
            )

        if industry_ids:
            trend_rows = self._query(
                """
                MATCH (i:Industry)-[:HAS_TREND]->(t:MarketTrend)
                WHERE elementId(i) IN $industry_ids
                RETURN i.name AS industry, t.trend_name AS trend,
                       t.description AS description
                """,
                {"industry_ids": industry_ids},
            )
        else:
            trend_rows = self._query(
                """
                CALL db.index.fulltext.queryNodes('trendNameFulltext', $q)
                YIELD node, score
                RETURN node.trend_name AS trend, node.description AS description, score
                ORDER BY score DESC
                """,
                {"q": query_text or ""},
            )

        return [{"skills": skill_rows, "trends": trend_rows}]

    # ---------- E: chunk fallback ----------
    def build_chunk_fallback_retriever(self):
        from neo4j_graphrag.retrievers import VectorCypherRetriever
        from neo4j_graphrag.types import RetrieverResultItem

        embeddings = get_embedding_model()

        # 禁止 Industry/SubSector → 全量 Job/Salary 扩展，否则大行业 chunk 会笛卡尔积卡死
        retrieval_query = """
        MATCH (node)-[:FROM_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (d)-[:FROM_REPORT]->(r:Report)
        OPTIONAL MATCH (d)-[:IS_OF_TYPE]->(rt:ReportType)

        OPTIONAL MATCH (i:Industry)-[:FROM_CHUNK]->(node)
        OPTIONAL MATCH (ss:SubSector)-[:FROM_CHUNK]->(node)
        OPTIONAL MATCH (j:JobPosition)-[:FROM_CHUNK]->(node)
        OPTIONAL MATCH (j)-[:HAS_SALARY_RANGE]->(s:SalaryRange)
        OPTIONAL MATCH (k:Skill)-[:FROM_CHUNK]->(node)
        OPTIONAL MATCH (t:MarketTrend)-[:FROM_CHUNK]->(node)

        WITH node, d, r, rt, score,
             collect(DISTINCT i.name) AS industries,
             collect(DISTINCT ss.name) AS sub_sectors,
             collect(DISTINCT j.title) AS job_titles,
             collect(DISTINCT {
                job: j.title,
                min: s.min_value,
                max: s.max_value,
                level: s.level
             }) AS salary_facts,
             collect(DISTINCT k.name) AS skills,
             collect(DISTINCT t.trend_name) AS market_trends

        RETURN
            elementId(node) AS chunk_element_id,
            node.text AS text,
            score,
            d.path AS document_path,
            r.title AS report_title,
            r.year AS report_year,
            rt.name AS report_type,
            [x IN industries WHERE x IS NOT NULL] AS industries,
            [x IN sub_sectors WHERE x IS NOT NULL] AS sub_sectors,
            [x IN job_titles WHERE x IS NOT NULL] AS job_titles,
            [x IN salary_facts WHERE x.job IS NOT NULL] AS salary_facts,
            [x IN skills WHERE x IS NOT NULL] AS skills,
            [x IN market_trends WHERE x IS NOT NULL] AS market_trends
        """

        def format_record(record):
            return RetrieverResultItem(
                content=record["text"],
                metadata={
                    "chunk_element_id": record.get("chunk_element_id"),
                    "score": record["score"],
                    "document_path": record.get("document_path"),
                    "report_title": record.get("report_title"),
                    "report_year": record.get("report_year"),
                    "report_type": record.get("report_type"),
                    "industries": record.get("industries"),
                    "sub_sectors": record.get("sub_sectors"),
                    "job_titles": record.get("job_titles"),
                    "salary_facts": record.get("salary_facts"),
                    "skills": record.get("skills"),
                    "market_trends": record.get("market_trends"),
                },
            )

        return VectorCypherRetriever(
            driver=self.driver,
            neo4j_database=self.db,
            index_name="chunkEmbedding",
            embedder=embeddings,
            retrieval_query=retrieval_query,
            result_formatter=format_record,
        )


def format_context_for_llm(intent: str, payload: Any) -> str:
    """把模板结果格式化为给 LLM 的上下文文本。"""
    requested = None
    if isinstance(payload, dict):
        requested = payload.get("requested_region")
    header = f"检索意图: {intent}\n"
    if requested:
        header += (
            f"用户关注地区口径: {requested}"
            f"（上下文可能含其它地区档，请优先采用该口径，勿编造缺失地区）\n"
        )
    header += "结构化检索结果如下（请严格依据这些事实回答，勿编造数字）：\n"
    try:
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        body = str(payload)
    return header + body
