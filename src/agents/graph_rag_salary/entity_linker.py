# -*- coding: utf-8 -*-
"""实体链接：别名 → fulltext → 名称向量。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.graph_rag_salary.schema.aliases import (
    INDUSTRY_ALIAS_LOOKUP,
    JOB_ALIAS_LOOKUP,
    SKILL_ALIAS_LOOKUP,
    SUB_SECTOR_ALIAS_LOOKUP,
    is_sub_sector_hint,
    normalize_sub_sector_hint,
    parent_industry_for_sub_sector,
    resolve_alias,
)
from agents.graph_rag_salary.neo4j_client import get_salary_neo4j_database
from core import get_embedding_model
from utils.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LinkedEntity:
    kind: str  # industry | sub_sector | job | skill
    query_text: str
    canonical: str | None
    element_id: str | None
    name: str | None
    score: float
    method: str  # alias|fulltext|vector|none


class EntityLinker:
    def __init__(self, driver, database: str | None = None):
        self.driver = driver
        self.db = database or get_salary_neo4j_database()

    def link_industry(self, text: str) -> LinkedEntity:
        """链接一级行业。若输入是二级行业 hint，先链 SubSector 再取父 Industry。"""
        text = (text or "").strip()
        if not text:
            return LinkedEntity("industry", text, None, None, None, 0.0, "none")

        if is_sub_sector_hint(text):
            ss = self.link_sub_sector(text)
            if ss.element_id:
                parent = self._parent_industry_of_sub_sector(ss.element_id)
                if parent:
                    return LinkedEntity(
                        kind="industry",
                        query_text=text,
                        canonical=parent["name"],
                        element_id=parent["eid"],
                        name=parent["name"],
                        score=ss.score,
                        method=f"via_sub_sector:{ss.method}",
                    )
            # 图中尚无 SubSector 时，用映射表解析父行业名再链
            parent_name = parent_industry_for_sub_sector(
                normalize_sub_sector_hint(text)
            )
            if parent_name:
                return self._link(
                    text=parent_name,
                    kind="industry",
                    alias_lookup=INDUSTRY_ALIAS_LOOKUP,
                    fulltext_index="industryNameFulltext",
                    vector_index="industryNameEmbedding",
                    name_prop="name",
                    label="Industry",
                )

        return self._link(
            text=text,
            kind="industry",
            alias_lookup=INDUSTRY_ALIAS_LOOKUP,
            fulltext_index="industryNameFulltext",
            vector_index="industryNameEmbedding",
            name_prop="name",
            label="Industry",
        )

    def link_sub_sector(self, text: str) -> LinkedEntity:
        return self._link(
            text=text,
            kind="sub_sector",
            alias_lookup=SUB_SECTOR_ALIAS_LOOKUP,
            fulltext_index="subSectorNameFulltext",
            vector_index="subSectorNameEmbedding",
            name_prop="name",
            label="SubSector",
        )

    def link_job(
        self,
        text: str,
        industry_element_ids: list[str] | None = None,
        sub_sector_hint: str | None = None,
        sub_sector_element_ids: list[str] | None = None,
    ) -> LinkedEntity:
        return self._link(
            text=text,
            kind="job",
            alias_lookup=JOB_ALIAS_LOOKUP,
            fulltext_index="jobTitleFulltext",
            vector_index="jobTitleEmbedding",
            name_prop="title",
            label="JobPosition",
            industry_element_ids=industry_element_ids,
            sub_sector_hint=sub_sector_hint,
            sub_sector_element_ids=sub_sector_element_ids,
        )

    def link_skill(self, text: str) -> LinkedEntity:
        return self._link(
            text=text,
            kind="skill",
            alias_lookup=SKILL_ALIAS_LOOKUP,
            fulltext_index="skillNameFulltext",
            vector_index="skillNameEmbedding",
            name_prop="name",
            label="Skill",
        )

    def link_many_industries(self, texts: list[str]) -> list[LinkedEntity]:
        return [self.link_industry(t) for t in texts if t]

    def link_many_sub_sectors(self, texts: list[str]) -> list[LinkedEntity]:
        return [self.link_sub_sector(t) for t in texts if t]

    def link_many_jobs(
        self,
        texts: list[str],
        industry_element_ids: list[str] | None = None,
        sub_sector_hint: str | None = None,
        sub_sector_element_ids: list[str] | None = None,
    ) -> list[LinkedEntity]:
        return [
            self.link_job(
                t,
                industry_element_ids,
                sub_sector_hint,
                sub_sector_element_ids,
            )
            for t in texts
            if t
        ]

    def _parent_industry_of_sub_sector(self, sub_sector_eid: str) -> dict | None:
        try:
            result = self.driver.execute_query(
                """
                MATCH (i:Industry)-[:HAS_SUB_SECTOR]->(ss:SubSector)
                WHERE elementId(ss) = $eid
                RETURN elementId(i) AS eid, i.name AS name
                LIMIT 1
                """,
                parameters_={"eid": sub_sector_eid},
                database_=self.db,
            )
            if result.records:
                return dict(result.records[0])
        except Exception as e:
            logger.warning("parent industry 查询失败: %s", e)
        return None

    def _link(
        self,
        text: str,
        kind: str,
        alias_lookup: dict,
        fulltext_index: str,
        vector_index: str,
        name_prop: str,
        label: str,
        industry_element_ids: list[str] | None = None,
        sub_sector_hint: str | None = None,
        sub_sector_element_ids: list[str] | None = None,
    ) -> LinkedEntity:
        text = (text or "").strip()
        if not text:
            return LinkedEntity(kind, text, None, None, None, 0.0, "none")

        # 1) 别名
        canonical = resolve_alias(text, alias_lookup)
        search_text = canonical or text
        job_scope = kind == "job"

        # 2) fulltext
        ft = self._fulltext_search(
            index_name=fulltext_index,
            query=search_text,
            name_prop=name_prop,
            industry_element_ids=industry_element_ids if job_scope else None,
            sub_sector_hint=sub_sector_hint if job_scope else None,
            sub_sector_element_ids=sub_sector_element_ids if job_scope else None,
        )
        if ft:
            return LinkedEntity(
                kind=kind,
                query_text=text,
                canonical=canonical or ft["name"],
                element_id=ft["eid"],
                name=ft["name"],
                score=float(ft.get("score") or 1.0),
                method="alias+fulltext" if canonical else "fulltext",
            )

        # 若别名规范名与原文不同，再试原文 fulltext
        if canonical and canonical != text:
            ft2 = self._fulltext_search(
                index_name=fulltext_index,
                query=text,
                name_prop=name_prop,
                industry_element_ids=industry_element_ids if job_scope else None,
                sub_sector_hint=sub_sector_hint if job_scope else None,
                sub_sector_element_ids=sub_sector_element_ids if job_scope else None,
            )
            if ft2:
                return LinkedEntity(
                    kind=kind,
                    query_text=text,
                    canonical=canonical,
                    element_id=ft2["eid"],
                    name=ft2["name"],
                    score=float(ft2.get("score") or 1.0),
                    method="fulltext",
                )

        # 3) 向量
        vec = self._vector_search(
            index_name=vector_index,
            text=search_text,
            name_prop=name_prop,
            industry_element_ids=industry_element_ids if job_scope else None,
            sub_sector_hint=sub_sector_hint if job_scope else None,
            sub_sector_element_ids=sub_sector_element_ids if job_scope else None,
        )
        if vec:
            return LinkedEntity(
                kind=kind,
                query_text=text,
                canonical=canonical or vec["name"],
                element_id=vec["eid"],
                name=vec["name"],
                score=float(vec.get("score") or 0.0),
                method="alias+vector" if canonical else "vector",
            )

        # 4) 宽松 CONTAINS
        loose = self._contains_search(
            label=label,
            name_prop=name_prop,
            text=search_text,
            industry_element_ids=industry_element_ids if job_scope else None,
            sub_sector_hint=sub_sector_hint if job_scope else None,
            sub_sector_element_ids=sub_sector_element_ids if job_scope else None,
        )
        if loose:
            return LinkedEntity(
                kind=kind,
                query_text=text,
                canonical=canonical or loose["name"],
                element_id=loose["eid"],
                name=loose["name"],
                score=0.5,
                method="contains",
            )

        return LinkedEntity(kind, text, canonical, None, None, 0.0, "none")

    @staticmethod
    def _job_scope_where(name_prop: str) -> str:
        """职位归属过滤：优先 SubSector 边，其次 Industry 直连；兼容属性兜底。"""
        return f"""
                (
                    size($sub_sector_ids) = 0 OR EXISTS {{
                        MATCH (ss:SubSector)-[:HAS_POSITION]->(node)
                        WHERE elementId(ss) IN $sub_sector_ids
                    }}
                )
                AND (
                    size($industry_ids) = 0 OR EXISTS {{
                        MATCH (i:Industry)-[:HAS_POSITION]->(node)
                        WHERE elementId(i) IN $industry_ids
                    }} OR EXISTS {{
                        MATCH (i:Industry)-[:HAS_SUB_SECTOR]->(:SubSector)-[:HAS_POSITION]->(node)
                        WHERE elementId(i) IN $industry_ids
                    }}
                )
                AND (
                    $sub_sector IS NULL OR $sub_sector = ''
                    OR EXISTS {{
                        MATCH (ss:SubSector)-[:HAS_POSITION]->(node)
                        WHERE toLower(ss.name) CONTAINS toLower($sub_sector)
                    }}
                    OR (node.sub_sector IS NOT NULL AND toLower(node.sub_sector) CONTAINS toLower($sub_sector))
                    OR toLower(node.{name_prop}) CONTAINS toLower($sub_sector)
                )
        """

    def _fulltext_search(
        self,
        index_name: str,
        query: str,
        name_prop: str,
        industry_element_ids: list[str] | None = None,
        sub_sector_hint: str | None = None,
        sub_sector_element_ids: list[str] | None = None,
    ) -> dict | None:
        safe = query.replace('"', " ").replace("~", " ").strip()
        if not safe:
            return None
        industry_ids = industry_element_ids or []
        sub_sector_ids = sub_sector_element_ids or []
        scoped = bool(industry_ids or sub_sector_ids or sub_sector_hint)
        try:
            if scoped:
                cypher = f"""
                CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
                WHERE {self._job_scope_where(name_prop)}
                RETURN elementId(node) AS eid, node.{name_prop} AS name, score
                ORDER BY score DESC
                LIMIT 1
                """
                result = self.driver.execute_query(
                    cypher,
                    parameters_={
                        "index": index_name,
                        "q": safe,
                        "industry_ids": industry_ids,
                        "sub_sector_ids": sub_sector_ids,
                        "sub_sector": sub_sector_hint or "",
                    },
                    database_=self.db,
                )
            else:
                cypher = f"""
                CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
                RETURN elementId(node) AS eid, node.{name_prop} AS name, score
                ORDER BY score DESC
                LIMIT 1
                """
                result = self.driver.execute_query(
                    cypher,
                    parameters_={"index": index_name, "q": safe},
                    database_=self.db,
                )
            if result.records:
                return dict(result.records[0])
        except Exception as e:
            logger.warning("fulltext 失败 (%s): %s", index_name, e)
        return None

    def _vector_search(
        self,
        index_name: str,
        text: str,
        name_prop: str,
        industry_element_ids: list[str] | None = None,
        sub_sector_hint: str | None = None,
        sub_sector_element_ids: list[str] | None = None,
    ) -> dict | None:
        industry_ids = industry_element_ids or []
        sub_sector_ids = sub_sector_element_ids or []
        scoped = bool(industry_ids or sub_sector_ids or sub_sector_hint)
        try:
            vector = self._embed_query(text)
            if scoped:
                cypher = f"""
                CALL db.index.vector.queryNodes($index, 8, $vector)
                YIELD node, score
                WHERE {self._job_scope_where(name_prop)}
                RETURN elementId(node) AS eid, node.{name_prop} AS name, score
                ORDER BY score DESC
                LIMIT 1
                """
                result = self.driver.execute_query(
                    cypher,
                    parameters_={
                        "index": index_name,
                        "vector": vector,
                        "industry_ids": industry_ids,
                        "sub_sector_ids": sub_sector_ids,
                        "sub_sector": sub_sector_hint or "",
                    },
                    database_=self.db,
                )
            else:
                cypher = f"""
                CALL db.index.vector.queryNodes($index, 5, $vector)
                YIELD node, score
                RETURN elementId(node) AS eid, node.{name_prop} AS name, score
                ORDER BY score DESC
                LIMIT 1
                """
                result = self.driver.execute_query(
                    cypher,
                    parameters_={"index": index_name, "vector": vector},
                    database_=self.db,
                )
            if result.records:
                return dict(result.records[0])
        except Exception as e:
            logger.warning("vector 失败 (%s): %s", index_name, e)
        return None

    def _contains_search(
        self,
        label: str,
        name_prop: str,
        text: str,
        industry_element_ids: list[str] | None = None,
        sub_sector_hint: str | None = None,
        sub_sector_element_ids: list[str] | None = None,
    ) -> dict | None:
        industry_ids = industry_element_ids or []
        sub_sector_ids = sub_sector_element_ids or []
        scoped = bool(industry_ids or sub_sector_ids or sub_sector_hint)
        try:
            if scoped:
                cypher = f"""
                MATCH (n:{label})
                WHERE toLower(n.{name_prop}) CONTAINS toLower($text)
                WITH n AS node
                WHERE {self._job_scope_where(name_prop)}
                RETURN elementId(node) AS eid, node.{name_prop} AS name
                LIMIT 1
                """
                result = self.driver.execute_query(
                    cypher,
                    parameters_={
                        "industry_ids": industry_ids,
                        "sub_sector_ids": sub_sector_ids,
                        "text": text,
                        "sub_sector": sub_sector_hint or "",
                    },
                    database_=self.db,
                )
            else:
                cypher = f"""
                MATCH (n:{label})
                WHERE toLower(n.{name_prop}) CONTAINS toLower($text)
                RETURN elementId(n) AS eid, n.{name_prop} AS name
                LIMIT 1
                """
                result = self.driver.execute_query(
                    cypher,
                    parameters_={"text": text},
                    database_=self.db,
                )
            if result.records:
                return dict(result.records[0])
        except Exception as e:
            logger.warning("contains 失败 (%s): %s", label, e)
        return None

    @staticmethod
    def _embed_query(text: str) -> list[float]:
        embeddings = get_embedding_model()
        return embeddings.embed_query(text)
