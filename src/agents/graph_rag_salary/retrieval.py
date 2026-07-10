# -*- coding: utf-8 -*-
"""薪酬 GraphRAG 检索编排：实体链接 + 模板查询 + 降级。"""

from __future__ import annotations

import json
from typing import Any

from agents.graph_rag_salary.entity_linker import EntityLinker, LinkedEntity
from agents.graph_rag_salary.indexes import ensure_indexes
from agents.graph_rag_salary.intent_router import IntentResult
from agents.graph_rag_salary.neo4j_client import get_salary_neo4j_database
from agents.graph_rag_salary.query_templates import OntologyQueryTemplates
from agents.graph_rag_salary.schema.aliases import (
    is_sub_sector_hint,
    normalize_sub_sector_hint,
)
from utils.log_utils import get_logger

logger = get_logger(__name__)


def _truncate(text: str, limit: int = 200) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def _log_linked(linked: list[LinkedEntity] | None, label: str = "linked") -> str:
    if not linked:
        return f"{label}=[]"
    parts = []
    for x in linked:
        if x is None:
            continue
        name = x.name or "?"
        parts.append(f"{x.query_text}→{name}({x.method})")
    return f"{label}=[{'; '.join(parts)}]"


def _ids(linked: list[LinkedEntity]) -> list[str]:
    return [x.element_id for x in linked if x.element_id]


def _link_debug(linked: list[LinkedEntity]) -> list[dict]:
    return [
        {
            "query": x.query_text,
            "name": x.name,
            "element_id": x.element_id,
            "method": x.method,
            "score": x.score,
        }
        for x in linked
    ]


class SalaryRetriever:
    """对齐参考 KgGraphRag.retrieve*：按意图链接实体、调模板、降级。"""

    def __init__(self, driver, query_text: str, intent: IntentResult):
        self.driver = driver
        self.query_text = query_text
        self.intent = intent
        self.db = get_salary_neo4j_database()
        self.linker = EntityLinker(driver, self.db)
        self.templates = OntologyQueryTemplates(driver, self.db)

    def retrieve(self) -> dict[str, Any]:
        ensure_indexes(self.driver, self.db)
        intent = self.intent.intent if self.intent else "fallback"
        logger.info(
            "retrieve route | branch=%s | db=%s | query=%s",
            intent,
            self.db,
            _truncate(self.query_text),
        )

        if intent == "salary_lookup":
            return self._with_region(self._retrieve_salary_lookup())
        if intent == "compare":
            return self._with_region(self._retrieve_compare())
        if intent == "industry_overview":
            return self._with_region(self._retrieve_overview())
        if intent == "skill_trend":
            return self._with_region(self._retrieve_skill_trend())
        return self._with_region(self._retrieve_fallback())

    def _with_region(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return payload
        region = self.intent.region if self.intent else None
        if region:
            payload["requested_region"] = region
        return payload

    def _resolve_sub_sectors(self, industry_texts: list[str] | None):
        texts = industry_texts or []
        hints = []
        for x in texts:
            if is_sub_sector_hint(x):
                h = normalize_sub_sector_hint(x)
                if h and h not in hints:
                    hints.append(h)
        linked = self.linker.link_many_sub_sectors(hints) if hints else []
        return hints, linked, _ids(linked)

    def _retrieve_salary_lookup(self) -> dict[str, Any]:
        industries = self.linker.link_many_industries(self.intent.industries)
        industry_ids = _ids(industries)
        sub_hints, linked_subs, sub_ids = self._resolve_sub_sectors(self.intent.industries)
        sub_hint = sub_hints[0] if sub_hints else None
        jobs = self.linker.link_many_jobs(
            self.intent.job_titles,
            industry_ids or None,
            sub_sector_hint=sub_hint,
            sub_sector_element_ids=sub_ids or None,
        )
        job_ids = _ids(jobs)

        logger.info(
            "salary_lookup link | %s | %s | %s | sub_hint=%s",
            _log_linked(industries, "industries"),
            _log_linked(linked_subs, "sub_sectors"),
            _log_linked(jobs, "jobs"),
            sub_hint,
        )

        if not job_ids and industry_ids:
            rows = self.templates.industry_overview(
                industry_ids,
                sub_sector=sub_hint,
                sub_sector_ids=sub_ids or None,
            )
            logger.info(
                "salary_lookup degrade | note=岗位未链上，降级 industry_overview | data_rows=%s",
                len(rows),
            )
            return {
                "mode": "salary_lookup→overview_fallback",
                "linked_industries": _link_debug(industries),
                "linked_sub_sectors": _link_debug(linked_subs),
                "linked_jobs": _link_debug(jobs),
                "data": rows,
            }

        rows = self.templates.salary_lookup(
            industry_ids,
            job_ids,
            sub_sector=sub_hint,
            sub_sector_ids=sub_ids or None,
        )
        if not rows:
            logger.info("salary_lookup degrade | note=ontology salary_lookup 无结果，降级 chunk")
            return self._retrieve_fallback(extra={
                "note": "ontology salary_lookup 无结果，降级 chunk",
                "linked_industries": _link_debug(industries),
                "linked_sub_sectors": _link_debug(linked_subs),
                "linked_jobs": _link_debug(jobs),
            })

        logger.info("salary_lookup result | mode=salary_lookup | data_rows=%s", len(rows))
        return {
            "mode": "salary_lookup",
            "linked_industries": _link_debug(industries),
            "linked_sub_sectors": _link_debug(linked_subs),
            "linked_jobs": _link_debug(jobs),
            "data": rows,
        }

    def _retrieve_compare(self) -> dict[str, Any]:
        compare_groups = self.intent.compare_groups or []

        if len(compare_groups) < 2:
            inds = self.intent.industries or []
            jobs = self.intent.job_titles or []
            if len(inds) >= 2:
                compare_groups = [
                    {"industry": inds[0], "job_title": jobs[0] if jobs else ""},
                    {
                        "industry": inds[1],
                        "job_title": jobs[1] if len(jobs) > 1 else (jobs[0] if jobs else ""),
                    },
                ]
            logger.info("compare groups synthesized | count=%s", len(compare_groups))

        template_groups = []
        for g in compare_groups:
            ind_text = g.get("industry") or ""
            job_text = g.get("job_title") or ""
            ind = self.linker.link_industry(ind_text) if ind_text else None
            ind_ids = [ind.element_id] if ind and ind.element_id else []
            sub_sector = normalize_sub_sector_hint(ind_text)
            ss = self.linker.link_sub_sector(sub_sector) if sub_sector else None
            ss_ids = [ss.element_id] if ss and ss.element_id else []
            job = (
                self.linker.link_job(
                    job_text,
                    ind_ids or None,
                    sub_sector_hint=sub_sector,
                    sub_sector_element_ids=ss_ids or None,
                )
                if job_text
                else None
            )
            job_ids = [job.element_id] if job and job.element_id else []
            label = f"{ind_text}-{job_text}".strip("-")
            template_groups.append({
                "label": label,
                "industry_ids": ind_ids,
                "job_ids": job_ids,
                "sub_sector": sub_sector,
                "sub_sector_ids": ss_ids,
                "linked": {
                    "industry": _link_debug([ind]) if ind else [],
                    "sub_sector": _link_debug([ss]) if ss else [],
                    "job": _link_debug([job]) if job else [],
                },
            })
            logger.info(
                "compare group link | label=%s | sub_sector=%s | %s | %s | %s",
                label,
                sub_sector,
                _log_linked([ind] if ind else [], "industry"),
                _log_linked([ss] if ss else [], "sub_sector"),
                _log_linked([job] if job else [], "job"),
            )

        data = self.templates.compare(template_groups)
        group_sizes = [
            {"label": g.get("group_label"), "n": len(g.get("results") or [])}
            for g in data
        ]
        if all(not g.get("results") for g in data):
            logger.info(
                "compare degrade | note=compare 无结构化结果，降级 chunk | groups=%s",
                json.dumps(group_sizes, ensure_ascii=False),
            )
            return self._retrieve_fallback(extra={"note": "compare 无结构化结果，降级 chunk"})

        logger.info(
            "compare result | mode=compare | groups=%s",
            json.dumps(group_sizes, ensure_ascii=False),
        )
        return {
            "mode": "compare",
            "groups": template_groups,
            "data": data,
        }

    def _retrieve_overview(self) -> dict[str, Any]:
        industries = self.linker.link_many_industries(self.intent.industries)
        industry_ids = _ids(industries)
        sub_hints, linked_subs, sub_ids = self._resolve_sub_sectors(self.intent.industries)
        sub_hint = sub_hints[0] if sub_hints else None
        logger.info(
            "overview link | %s | %s",
            _log_linked(industries, "industries"),
            _log_linked(linked_subs, "sub_sectors"),
        )
        if not industry_ids:
            logger.info("overview degrade | note=未链接到行业，降级 chunk")
            return self._retrieve_fallback(extra={
                "note": "未链接到行业，降级 chunk",
                "linked_industries": _link_debug(industries),
                "linked_sub_sectors": _link_debug(linked_subs),
            })
        rows = self.templates.industry_overview(
            industry_ids,
            sub_sector=sub_hint,
            sub_sector_ids=sub_ids or None,
        )
        pos_n = 0
        if rows:
            pos_n = len((rows[0] or {}).get("positions") or [])
        logger.info(
            "overview result | mode=industry_overview | sub_sector_filter=%s | "
            "data_rows=%s | positions=%s",
            sub_hint,
            len(rows),
            pos_n,
        )
        return {
            "mode": "industry_overview",
            "linked_industries": _link_debug(industries),
            "linked_sub_sectors": _link_debug(linked_subs),
            "sub_sector_filter": sub_hint,
            "data": rows,
        }

    def _retrieve_skill_trend(self) -> dict[str, Any]:
        skills = [self.linker.link_skill(s) for s in (self.intent.skills or [])]
        industries = self.linker.link_many_industries(self.intent.industries or [])
        skill_ids = _ids(skills)
        industry_ids = _ids(industries)
        logger.info(
            "skill_trend link | skills_linked=%s/%s | industries_linked=%s/%s",
            len(skill_ids),
            len(skills),
            len(industry_ids),
            len(industries),
        )
        rows = self.templates.skill_trend(
            skill_ids=skill_ids or None,
            industry_ids=industry_ids or None,
            query_text=self.query_text,
        )
        logger.info("skill_trend result | mode=skill_trend | data_rows=%s", len(rows))
        return {
            "mode": "skill_trend",
            "linked_skills": _link_debug(skills),
            "linked_industries": _link_debug(industries),
            "data": rows,
        }

    def _retrieve_fallback(self, extra: dict | None = None) -> dict[str, Any]:
        note = (extra or {}).get("note") if extra else None
        logger.info("fallback chunk start | note=%s | top_k=3", note or "direct_fallback")
        retriever = self.templates.build_chunk_fallback_retriever()
        try:
            result = retriever.search(query_text=self.query_text, top_k=3)
        except TypeError:
            result = retriever.get_search_results(
                query_text=self.query_text,
                top_k=5,
            )

        items = []
        raw_items = getattr(result, "items", None) or getattr(result, "records", None) or []
        for item in raw_items:
            if hasattr(item, "content"):
                items.append({
                    "content": item.content,
                    "metadata": getattr(item, "metadata", None),
                })
            elif isinstance(item, dict):
                items.append(item)
            else:
                items.append({"content": str(item), "metadata": {}})

        logger.info("fallback chunk result | mode=fallback_chunk | items=%s", len(items))
        payload: dict[str, Any] = {
            "mode": "fallback_chunk",
            "data": items,
        }
        if extra:
            payload["extra"] = extra
        return payload


def run_salary_retrieval(
    driver,
    query_text: str,
    intent: IntentResult,
) -> dict[str, Any]:
    """同步入口，供 LangGraph 节点经 asyncio.to_thread 调用。"""
    return SalaryRetriever(driver, query_text, intent).retrieve()
