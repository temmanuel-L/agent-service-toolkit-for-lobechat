# -*- coding: utf-8 -*-
"""实体链接：embedding 优先 + fulltext 兜底。"""

from __future__ import annotations

from dataclasses import dataclass

from agents.salary_graphrag_agent.neo4j_client import get_salary_graphrag_neo4j_database
from agents.salary_graphrag_agent.schema.aliases import normalize_salary_level
from core import get_embedding_model
from utils.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LinkedEntity:
    kind: str  # industry | sub_sector | job | skill | area
    query_text: str
    canonical: str | None
    element_id: str | None
    name: str | None
    score: float
    method: str  # vector|fulltext|exact|none


class EntityLinker:
    def __init__(self, driver, database: str | None = None):
        self.driver = driver
        self.db = database or get_salary_graphrag_neo4j_database()

    def link_industry(self, text: str) -> LinkedEntity:
        return self._link(text, "industry", "industryNameFulltext", "industryNameEmbedding", "name")

    def link_sub_sector(self, text: str) -> LinkedEntity:
        return self._link(text, "sub_sector", "subSectorNameFulltext", "subSectorNameEmbedding", "name")

    def link_job(self, text: str) -> LinkedEntity:
        return self._link(text, "job", "jobTitleFulltext", "jobTitleEmbedding", "title")

    def link_area(self, text: str) -> LinkedEntity:
        text = (text or "").strip()
        canonical = normalize_salary_level(text)
        if not canonical:
            return LinkedEntity("area", text, None, None, None, 0.0, "none")
        result = self.driver.execute_query(
            "MATCH (a:Area {name: $name}) RETURN elementId(a) AS eid, a.name AS name",
            parameters_={"name": canonical},
            database_=self.db,
        )
        if result.records:
            r = result.records[0]
            return LinkedEntity("area", text, canonical, r["eid"], r["name"], 1.0, "exact")
        return LinkedEntity("area", text, canonical, None, None, 0.0, "none")

    def link_many(self, texts: list[str], link_fn) -> list[LinkedEntity]:
        return [link_fn(t) for t in (texts or []) if t]

    def resolve_parent_industry(self, sub_sector_element_id: str) -> LinkedEntity | None:
        if not sub_sector_element_id:
            return None
        try:
            result = self.driver.execute_query(
                """
                MATCH (i:Industry)-[:HAS_SUB_SECTOR]->(ss:SubSector)
                WHERE elementId(ss) = $eid
                RETURN elementId(i) AS eid, i.name AS name
                LIMIT 1
                """,
                parameters_={"eid": sub_sector_element_id},
                database_=self.db,
            )
            if result.records:
                r = result.records[0]
                return LinkedEntity(
                    kind="industry",
                    query_text="",
                    canonical=r["name"],
                    element_id=r["eid"],
                    name=r["name"],
                    score=1.0,
                    method="via_sub_sector",
                )
        except Exception as e:
            logger.warning("resolve_parent_industry 失败: %s", e)
        return None

    def link_industries_by_names(self, names: list[str]) -> list[LinkedEntity]:
        names = [n for n in (names or []) if n]
        if not names:
            return []
        try:
            result = self.driver.execute_query(
                """
                UNWIND $names AS name
                MATCH (i:Industry {name: name})
                RETURN elementId(i) AS eid, i.name AS name
                """,
                parameters_={"names": names},
                database_=self.db,
            )
            linked = []
            for r in (result.records or []):
                linked.append(
                    LinkedEntity(
                        kind="industry",
                        query_text=r["name"],
                        canonical=r["name"],
                        element_id=r["eid"],
                        name=r["name"],
                        score=1.0,
                        method="exact",
                    )
                )
            return linked
        except Exception as e:
            logger.warning("link_industries_by_names 失败: %s", e)
            return []

    def link_sub_sectors_by_names(self, names: list[str]) -> list[LinkedEntity]:
        names = [n for n in (names or []) if n]
        if not names:
            return []
        try:
            result = self.driver.execute_query(
                """
                UNWIND $names AS name
                MATCH (ss:SubSector {name: name})
                RETURN elementId(ss) AS eid, ss.name AS name
                """,
                parameters_={"names": names},
                database_=self.db,
            )
            linked = []
            for r in (result.records or []):
                linked.append(
                    LinkedEntity(
                        kind="sub_sector",
                        query_text=r["name"],
                        canonical=r["name"],
                        element_id=r["eid"],
                        name=r["name"],
                        score=1.0,
                        method="exact",
                    )
                )
            return linked
        except Exception as e:
            logger.warning("link_sub_sectors_by_names 失败: %s", e)
            return []

    def _link(
        self,
        text: str,
        kind: str,
        fulltext_index: str,
        vector_index: str,
        name_prop: str,
    ) -> LinkedEntity:
        text = (text or "").strip()
        if not text:
            return LinkedEntity(kind, text, None, None, None, 0.0, "none")

        vec = self._vector_search(vector_index, text, name_prop)
        if vec:
            return LinkedEntity(kind, text, None, vec["eid"], vec["name"], vec["score"], "vector")

        ft = self._fulltext_search(fulltext_index, text, name_prop)
        if ft:
            return LinkedEntity(kind, text, None, ft["eid"], ft["name"], ft["score"], "fulltext")

        return LinkedEntity(kind, text, None, None, None, 0.0, "none")

    def _fulltext_search(self, index_name: str, query: str, name_prop: str) -> dict | None:
        safe = query.replace('"', " ").replace("~", " ").strip()
        if not safe:
            return None
        try:
            result = self.driver.execute_query(
                f"""CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
                    RETURN elementId(node) AS eid, node.{name_prop} AS name, score
                    ORDER BY score DESC LIMIT 1""",
                parameters_={"index": index_name, "q": safe},
                database_=self.db,
            )
            if result.records:
                r = result.records[0]
                return {"eid": r["eid"], "name": r["name"], "score": float(r["score"])}
        except Exception as e:
            logger.warning("fulltext 失败 (%s): %s", index_name, e)
        return None

    def _vector_search(
        self,
        index_name: str,
        text: str,
        name_prop: str,
        top_k: int = 5,
    ) -> dict | None:
        try:
            vector = self._embed_query(text)
            result = self.driver.execute_query(
                f"""CALL db.index.vector.queryNodes($index, $top_k, $vector)
                    YIELD node, score
                    RETURN elementId(node) AS eid, node.{name_prop} AS name, score
                    ORDER BY score DESC LIMIT 1""",
                parameters_={"index": index_name, "vector": vector, "top_k": top_k},
                database_=self.db,
            )
            if result.records:
                r = result.records[0]
                return {"eid": r["eid"], "name": r["name"], "score": float(r["score"])}
        except Exception as e:
            logger.warning("vector 失败 (%s): %s", index_name, e)
        return None

    @staticmethod
    def _embed_query(text: str) -> list[float]:
        embeddings = get_embedding_model()
        return embeddings.embed_query(text)
