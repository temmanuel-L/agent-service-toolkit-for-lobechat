# -*- coding: utf-8 -*-
"""推理链定义：意图 = 最长推理链。

两条通用链对应两个业务意图：
- salary_chain（长链）：Industry → SubSector → JobPosition → Area
- industry_supplement（短链）：Industry 属性召回 + JobPosition[category]
"""

from __future__ import annotations


REASONING_CHAINS = {
    "salary_chain": {
        "description": "薪资推理链：Industry → SubSector → JobPosition → Area",
        "nodes": ["Industry", "SubSector", "JobPosition", "Area"],
        "required": ["Industry"],
        "optional": ["SubSector", "JobPosition", "Area"],
        "cypher_template": "salary_chain_recall",
    },
    "industry_supplement": {
        "description": (
            "行业补充信息链：Industry 属性召回"
            "（skills/hot_positions/high_paying_positions/trends/overview）"
            "+ JobPosition[category]"
        ),
        "nodes": ["Industry", "JobPosition"],
        "required": ["Industry"],
        "optional": [],
        "cypher_template": "industry_supplement_recall",
    },
}


BUSINESS_CHAINS = list(REASONING_CHAINS.keys())


def is_business_chain(chain: str | None) -> bool:
    return chain in BUSINESS_CHAINS


def required_layers(chain: str) -> list[str]:
    cfg = REASONING_CHAINS.get(chain) or {}
    return list(cfg.get("required", []))


def optional_layers(chain: str) -> list[str]:
    cfg = REASONING_CHAINS.get(chain) or {}
    return list(cfg.get("optional", []))


def cypher_template_name(chain: str) -> str | None:
    cfg = REASONING_CHAINS.get(chain) or {}
    return cfg.get("cypher_template")
