# -*- coding: utf-8 -*-
"""两个通用 Cypher 模板 + chunk 混合检索。"""

from __future__ import annotations

from typing import Any

from agents.salary_graphrag_agent.neo4j_client import get_salary_graphrag_neo4j_database
from agents.salary_graphrag_agent.schema.aliases import (
    area_sort_key,
    normalize_salary_level,
)
from core import get_embedding_model
from utils.log_utils import get_logger

logger = get_logger(__name__)


def _salary_min(s: dict) -> float:
    try:
        return float(s.get("min") or s.get("min_value") or 0)
    except (TypeError, ValueError):
        return 0.0


def _salary_max(s: dict) -> float:
    try:
        return float(s.get("max") or s.get("max_value") or 0)
    except (TypeError, ValueError):
        return 0.0


def _salary_band_width(s: dict) -> float:
    return _salary_max(s) - _salary_min(s)


def dedupe_salaries(salaries: list | None) -> list:
    """同一岗位多条薪资：按 area+(min,max) 去重，同 area 多组取最窄。"""
    items = [x for x in (salaries or []) if x is not None]
    if not items:
        return []

    by_area: dict[str, list[dict]] = {}
    for raw in items:
        d = (
            raw
            if isinstance(raw, dict)
            else {
                "min": getattr(raw, "min", None),
                "max": getattr(raw, "max", None),
                "area": getattr(raw, "area", None) or getattr(raw, "level", None),
                "unit": getattr(raw, "unit", None),
            }
        )
        area = normalize_salary_level(d.get("area")) or d.get("area") or ""
        d["area"] = area
        if d.get("min") is None and d.get("min_value") is not None:
            d["min"] = d["min_value"]
        if d.get("max") is None and d.get("max_value") is not None:
            d["max"] = d["max_value"]
        by_area.setdefault(area, []).append(d)

    result: list[dict] = []
    seen: set[tuple[str, float, float]] = set()
    for area in sorted(by_area.keys(), key=area_sort_key):
        cands = by_area[area]
        unique: dict[tuple, dict] = {}
        for c in cands:
            key = (_salary_min(c), _salary_max(c))
            if key not in unique:
                unique[key] = c
        chosen = (
            min(unique.values(), key=lambda x: (_salary_band_width(x), _salary_min(x)))
            if len(unique) > 1
            else next(iter(unique.values()))
        )
        exact = (area, _salary_min(chosen), _salary_max(chosen))
        if exact not in seen:
            seen.add(exact)
            result.append(chosen)
    return result


class OntologyQueryTemplates:
    def __init__(self, driver, database: str | None = None):
        self.driver = driver
        self.db = database or get_salary_graphrag_neo4j_database()

    def _query(self, cypher: str, params: dict | None = None) -> list[dict]:
        result = self.driver.execute_query(cypher, parameters_=params or {}, database_=self.db)
        return [dict(r) for r in (result.records or [])]

    def salary_chain_recall(
        self,
        industry_ids: list[str] | None = None,
        sub_sector_ids: list[str] | None = None,
        job_ids: list[str] | None = None,
        area: str | None = None,
    ) -> list[dict]:
        industry_ids = industry_ids or []
        sub_sector_ids = sub_sector_ids or []
        job_ids = job_ids or []
        area_norm = normalize_salary_level(area) if area else None

        cypher = """
        MATCH (i:Industry)
        WHERE size($industry_ids) = 0 OR elementId(i) IN $industry_ids
        OPTIONAL MATCH (i)-[:HAS_SUB_SECTOR]->(ss:SubSector)
        WITH i, collect(DISTINCT ss) AS all_sub_sectors,
             CASE WHEN size($sub_sector_ids) = 0 THEN collect(DISTINCT ss)
                  ELSE [x IN collect(DISTINCT ss) WHERE elementId(x) IN $sub_sector_ids] END AS scoped_subs
        OPTIONAL MATCH (i)-[:HAS_SUB_SECTOR]->(ss_a:SubSector)-[:HAS_POSITION]->(j_a:JobPosition)
        WHERE CASE
          WHEN size($job_ids) > 0 THEN elementId(j_a) IN $job_ids
          ELSE size($sub_sector_ids) = 0 OR elementId(ss_a) IN $sub_sector_ids
        END
        OPTIONAL MATCH (i)-[:HAS_POSITION]->(j_b:JobPosition)
        WHERE size($job_ids) = 0 OR elementId(j_b) IN $job_ids
        WITH i, all_sub_sectors, scoped_subs,
             collect(DISTINCT j_a) + collect(DISTINCT j_b) AS jobs_raw
        UNWIND (CASE WHEN size(jobs_raw) = 0 THEN [null] ELSE jobs_raw END) AS j
        WITH i, all_sub_sectors, scoped_subs, j
        WHERE j IS NULL OR (
            (size($job_ids) = 0 OR elementId(j) IN $job_ids)
        )
        OPTIONAL MATCH (ss_j:SubSector)-[:HAS_POSITION]->(j)
        OPTIONAL MATCH (j)-[sal:HAS_SALARY_IN]->(a:Area)
        WHERE $area IS NULL OR $area = '' OR a.name = $area
        WITH i, all_sub_sectors, scoped_subs, j, collect(DISTINCT ss_j.name) AS j_sub_sectors,
             collect(DISTINCT {min: sal.min_value, max: sal.max_value,
                                unit: coalesce(sal.unit, '千元人民币'), area: a.name}) AS salaries
        WITH i, all_sub_sectors, scoped_subs, j, j_sub_sectors,
             [x IN salaries WHERE x.min IS NOT NULL AND x.max IS NOT NULL | x] AS all_salaries
        WITH i, all_sub_sectors, scoped_subs,
             [p IN collect(DISTINCT {
                 title: j.title, category: j.category,
                 sub_sector: coalesce(j_sub_sectors[0], j.sub_sector),
                 salaries: [x IN all_salaries | {min: x.min, max: x.max, unit: x.unit, area: x.area}]
             }) WHERE p.title IS NOT NULL | p] AS positions
        RETURN
            i.name AS industry,
            i.overview AS overview,
            [x IN all_sub_sectors WHERE x IS NOT NULL | x.name] AS sub_sectors,
            [x IN scoped_subs WHERE x IS NOT NULL | x.name] AS scoped_sub_sectors,
            positions
        ORDER BY i.name
        """
        rows = self._query(
            cypher,
            {
                "industry_ids": industry_ids,
                "sub_sector_ids": sub_sector_ids,
                "job_ids": job_ids,
                "area": area_norm or "",
            },
        )
        for row in rows:
            for pos in (row.get("positions") or []):
                if isinstance(pos, dict):
                    pos["salaries"] = dedupe_salaries(pos.get("salaries"))
        return rows

    def industry_supplement_recall(
        self,
        industry_ids: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> list[dict]:
        industry_ids = industry_ids or []
        categories = categories or []

        cypher = """
        MATCH (i:Industry)
        WHERE elementId(i) IN $industry_ids
        OPTIONAL MATCH (i)-[:HAS_POSITION]->(j_direct:JobPosition)
        OPTIONAL MATCH (i)-[:HAS_SUB_SECTOR]->(ss:SubSector)-[:HAS_POSITION]->(j_via:JobPosition)
        WITH i, collect(DISTINCT j_direct) + collect(DISTINCT j_via) AS jobs_raw,
             collect(DISTINCT ss) AS all_sub_sectors
        UNWIND (CASE WHEN size(jobs_raw) = 0 THEN [null] ELSE jobs_raw END) AS j
        WITH i, all_sub_sectors, j
        WHERE j IS NULL OR (
            size($categories) = 0 OR j.category IN $categories OR j.category IS NULL
        )
        OPTIONAL MATCH (ss_j:SubSector)-[:HAS_POSITION]->(j)
        OPTIONAL MATCH (j)-[sal:HAS_SALARY_IN]->(a:Area)
        WITH i, all_sub_sectors, j,
             collect(DISTINCT ss_j.name) AS j_sub_sectors,
             collect(DISTINCT {min: sal.min_value, max: sal.max_value, area: a.name}) AS salaries
        WITH i, all_sub_sectors, j, j_sub_sectors,
             [x IN salaries WHERE x.min IS NOT NULL AND x.max IS NOT NULL | x] AS all_salaries
        WITH i, all_sub_sectors,
             [p IN collect(DISTINCT {
                 title: j.title, category: j.category,
                 sub_sector: coalesce(j_sub_sectors[0], j.sub_sector),
                 salaries: [x IN all_salaries | {min: x.min, max: x.max, area: x.area}]
             }) WHERE p.title IS NOT NULL | p] AS positions
        RETURN
            i.name AS industry,
            i.overview AS overview,
            i.skills AS skills,
            i.hot_positions AS hot_positions,
            i.high_paying_positions AS high_paying_positions,
            i.trends AS trends,
            [x IN all_sub_sectors WHERE x IS NOT NULL | x.name] AS sub_sectors,
            positions
        ORDER BY i.name
        """
        rows = self._query(
            cypher,
            {
                "industry_ids": industry_ids,
                "categories": categories,
            },
        )
        for row in rows:
            for pos in (row.get("positions") or []):
                if isinstance(pos, dict):
                    pos["salaries"] = dedupe_salaries(pos.get("salaries"))
        return rows

    def chunk_hybrid_recall(self, query_text: str, top_k: int = 5) -> list[dict]:
        if not query_text:
            return []
        vec_rows = self._chunk_vector_search(query_text, top_k=top_k)
        bm25_rows = self._chunk_fulltext_search(query_text, top_k=top_k)
        fused = self._rrf_fuse(vec_rows, bm25_rows, k=60)
        return fused[:top_k]

    def _chunk_vector_search(self, query_text: str, top_k: int = 5) -> list[dict]:
        try:
            vector = self._embed_query(query_text)
        except Exception as e:
            logger.warning("chunk vector embed 失败: %s", e)
            return []
        cypher = """
        CALL db.index.vector.queryNodes('chunkEmbedding', $top_k, $vector)
        YIELD node, score
        WITH node AS c, score
        OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (d)-[:FROM_REPORT]->(r:Report)
        OPTIONAL MATCH (i:Industry)-[:FROM_CHUNK]->(c)
        OPTIONAL MATCH (ss:SubSector)-[:FROM_CHUNK]->(c)
        OPTIONAL MATCH (j:JobPosition)-[:FROM_CHUNK]->(c)
        OPTIONAL MATCH (j)-[sal:HAS_SALARY_IN]->(a:Area)
        WITH c, d, r, score,
             collect(DISTINCT i.name) AS industries,
             collect(DISTINCT ss.name) AS sub_sectors,
             collect(DISTINCT j.title) AS job_titles,
             collect(DISTINCT {job: j.title, min: sal.min_value, max: sal.max_value, area: a.name}) AS salary_facts
        RETURN
            elementId(c) AS chunk_element_id,
            c.text AS text,
            score,
            d.path AS document_path,
            r.name AS report_title,
            r.year AS report_year,
            [x IN industries WHERE x IS NOT NULL] AS industries,
            [x IN sub_sectors WHERE x IS NOT NULL] AS sub_sectors,
            [x IN job_titles WHERE x IS NOT NULL] AS job_titles,
            [x IN salary_facts WHERE x.job IS NOT NULL] AS salary_facts
        ORDER BY score DESC
        LIMIT $top_k
        """
        try:
            return self._query(cypher, {"vector": vector, "top_k": top_k})
        except Exception as e:
            logger.warning("chunk vector 检索失败: %s", e)
            return []

    def _chunk_fulltext_search(self, query_text: str, top_k: int = 5) -> list[dict]:
        safe = (query_text or "").replace('"', " ").replace("~", " ").strip()
        if not safe:
            return []
        cypher = """
        CALL db.index.fulltext.queryNodes('chunkTextFulltext', $q) YIELD node, score
        WITH node AS c, score
        OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (d)-[:FROM_REPORT]->(r:Report)
        OPTIONAL MATCH (i:Industry)-[:FROM_CHUNK]->(c)
        OPTIONAL MATCH (ss:SubSector)-[:FROM_CHUNK]->(c)
        OPTIONAL MATCH (j:JobPosition)-[:FROM_CHUNK]->(c)
        OPTIONAL MATCH (j)-[sal:HAS_SALARY_IN]->(a:Area)
        WITH c, d, r, score,
             collect(DISTINCT i.name) AS industries,
             collect(DISTINCT ss.name) AS sub_sectors,
             collect(DISTINCT j.title) AS job_titles,
             collect(DISTINCT {job: j.title, min: sal.min_value, max: sal.max_value, area: a.name}) AS salary_facts
        RETURN
            elementId(c) AS chunk_element_id,
            c.text AS text,
            score,
            d.path AS document_path,
            r.name AS report_title,
            r.year AS report_year,
            [x IN industries WHERE x IS NOT NULL] AS industries,
            [x IN sub_sectors WHERE x IS NOT NULL] AS sub_sectors,
            [x IN job_titles WHERE x IS NOT NULL] AS job_titles,
            [x IN salary_facts WHERE x.job IS NOT NULL] AS salary_facts
        ORDER BY score DESC
        LIMIT $top_k
        """
        try:
            return self._query(cypher, {"q": safe, "top_k": top_k})
        except Exception as e:
            logger.warning("chunk fulltext 检索失败: %s", e)
            return []

    @staticmethod
    def _rrf_fuse(vec_rows: list[dict], bm25_rows: list[dict], k: int = 60) -> list[dict]:
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}
        for rank, row in enumerate(vec_rows):
            cid = row.get("chunk_element_id") or row.get("text") or str(rank)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in meta:
                meta[cid] = dict(row)
        for rank, row in enumerate(bm25_rows):
            cid = row.get("chunk_element_id") or row.get("text") or f"b{rank}"
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in meta:
                meta[cid] = dict(row)
        fused = []
        for cid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            row = dict(meta[cid])
            row["fused_score"] = sc
            fused.append(row)
        return fused

    @staticmethod
    def _embed_query(text: str) -> list[float]:
        embeddings = get_embedding_model()
        return embeddings.embed_query(text)


def format_context_for_llm(intent: str, payload: Any) -> str:
    """把检索 payload 格式化为给 LLM 的上下文文本。"""
    import json

    header = f"检索意图: {intent}\n"
    if isinstance(payload, dict):
        requested = payload.get("requested_area") or payload.get("requested_region")
        if requested:
            header += (
                f"用户关注区域口径: {requested}"
                f"（优先采用该口径，勿编造缺失区域）\n"
            )
    header += "结构化检索结果如下（严格依据事实回答，勿编造数字）：\n"
    try:
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        body = str(payload)
    return header + body
