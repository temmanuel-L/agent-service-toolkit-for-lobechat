# -*- coding: utf-8 -*-
"""枚举规范名清单 + 区域归一化（去除同义别名表，提升通用性）。

设计原则：
- 不再维护硬编码同义别名表（INDUSTRY_ALIASES/JOB_ALIASES 等），改用规范名清单 + 向量检索兜底。
- 枚举型实体（Industry/SubSector/Area）保留规范名清单，供 schema prompt 注入与归一。
- 自由型实体（JobPosition）不枚举，靠 fulltext + 向量检索匹配。
- 复杂文档可通过 load_canonical_names_from_graph 从图里动态读规范名，无需手写别名表。
"""

from __future__ import annotations


# ==================== 枚举规范名清单 ====================
# 一级行业 11 个（与 schema/nodes.py Industry.description 末尾清单一致）
INDUSTRY_CANONICAL_NAMES = [
    "银行与金融服务",
    "工程与制造",
    "财务与会计",
    "医疗与生命科学",
    "人力资源与行政助理",
    "法务",
    "市场营销与电商",
    "销售与零售",
    "采购与供应链",
    "科技",
    "半导体",
]

# 二级行业规范名（薪资报告里出现的；可空，建图时 LLM 自抽，不强制枚举）
SUB_SECTOR_CANONICAL_NAMES = [
    "银行",
    "保险",
    "一级市场",
    "资管",
]

# 薪酬地理口径 4 个枚举（Area 节点全图仅这 4 个）
AREA_CANONICAL_NAMES = ["全国平均", "华东", "华北", "华南"]

# 检索展示排序：全国平均优先，再空，再三大区
AREA_SORT_ORDER = ("全国平均", "", "华东", "华北", "华南")


# ==================== 区域归一化 ====================
def normalize_salary_level(text: str | None) -> str:
    """把用户/抽取的地区口径归一成 Area 规范名；空则返回空串。

    基于 AREA_CANONICAL_NAMES 做精确 + 子串匹配，不再走别名表。
    例：'全国' → '全国平均'，'华东地区' → '华东'，'华南区' → '华南'。
    """
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    # 精确命中
    if t in AREA_CANONICAL_NAMES:
        return t
    # 子串包含：规范名是用户输入的子串，或用户输入是规范名的子串
    # 较长规范名优先，避免 '全国' 抢在 '全国平均' 前误匹配
    candidates = []
    for canon in AREA_CANONICAL_NAMES:
        if canon in t or t in canon:
            candidates.append(canon)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return t


def extract_region_from_text(text: str | None) -> str | None:
    """从问句中抽出首个薪酬地区口径；未提及则 None。

    基于 AREA_CANONICAL_NAMES 子串匹配，不再走别名表。
    """
    if not text:
        return None
    q = str(text).strip()
    candidates = []
    for canon in AREA_CANONICAL_NAMES:
        # 检查规范名本身及其常见简写是否出现在问句中
        probes = [canon]
        if canon == "全国平均":
            probes.append("全国")
        elif canon == "华东":
            probes.append("华东地区")
            probes.append("华东区")
        elif canon == "华北":
            probes.append("华北地区")
            probes.append("华北区")
        elif canon == "华南":
            probes.append("华南地区")
            probes.append("华南区")
        for probe in probes:
            if probe and probe in q:
                candidates.append((probe, canon))
                break
    if not candidates:
        return None
    # 较长 probe 优先，避免 '全国' 抢在 '全国平均' 前误匹配
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0][1]


def area_sort_key(area: str) -> tuple:
    """Area 排序 key：全国平均 → 空 → 华东 → 华北 → 华南 → 其它。"""
    try:
        idx = AREA_SORT_ORDER.index(area)
    except ValueError:
        idx = len(AREA_SORT_ORDER)
    return (idx, area)


# ==================== 从图里动态加载规范名（通用） ====================
def load_canonical_names_from_graph(driver, label: str, prop: str = "name",
                                    database: str | None = None) -> list[str]:
    """从图里读某 label 的 distinct 规范名，供 planner 动态 grounding。

    通用接口：复杂文档（年报等）也能用，无需手写别名表。
    例：load_canonical_names_from_graph(driver, "Industry", "name")
    """
    cypher = f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL RETURN DISTINCT n.{prop} AS name ORDER BY name"
    result = driver.execute_query(cypher, database_=database)
    return [r["name"] for r in (result.records or []) if r.get("name")]


def load_industry_subsector_tree(driver, database: str | None = None) -> dict[str, list[str]]:
    """从图里加载 Industry → [SubSector] 规范名树，供 planner prompt 注入。

    LLM 看到这棵树，能在抽取实体时直接对齐规范名，并补出前置一级行业
    （用户只提"保险"时，LLM 看到树知道保险挂在银行与金融服务下）。

    例返回：{"银行与金融服务": ["银行", "保险", "一级市场", "资管"], ...}
    """
    cypher = """
    MATCH (i:Industry)-[:HAS_SUB_SECTOR]->(ss:SubSector)
    RETURN i.name AS industry, collect(DISTINCT ss.name) AS sub_sectors
    ORDER BY i.name
    """
    result = driver.execute_query(cypher, database_=database)
    tree: dict[str, list[str]] = {}
    for rec in (result.records or []):
        ind = rec.get("industry")
        if not ind:
            continue
        subs = [x for x in (rec.get("sub_sectors") or []) if x]
        tree.setdefault(ind, [])
        for s in subs:
            if s not in tree[ind]:
                tree[ind].append(s)
    # 无 SubSector 的 Industry 也列出（key 对应空 list）
    cypher_solo = "MATCH (i:Industry) WHERE NOT EXISTS { MATCH (i)-[:HAS_SUB_SECTOR]->() } RETURN i.name AS name ORDER BY name"
    solo = driver.execute_query(cypher_solo, database_=database)
    for rec in (solo.records or []):
        name = rec.get("name")
        if name and name not in tree:
            tree[name] = []
    return tree


# ==================== 实体名缓存（避免每次查询都打 Cypher） ====================
_entity_cache: dict[str, list[str]] = {}
_industry_tree_cache: dict[str, list[str]] | None = None


def get_cached_entity_names(driver, label: str, prop: str = "name",
                            database: str | None = None) -> list[str]:
    """从图里加载实体名并缓存。建图后需调用 clear_entity_cache() 刷新。"""
    if label not in _entity_cache:
        _entity_cache[label] = load_canonical_names_from_graph(driver, label, prop, database)
    return _entity_cache[label]


def get_cached_industry_subsector_tree(driver, database: str | None = None) -> dict[str, list[str]]:
    """从图里加载 Industry→SubSector 树并缓存。建图后需调用 clear_entity_cache() 刷新。"""
    global _industry_tree_cache
    if _industry_tree_cache is None:
        _industry_tree_cache = load_industry_subsector_tree(driver, database)
    return _industry_tree_cache


def clear_entity_cache():
    """清空缓存（建图后调用）。"""
    _entity_cache.clear()
    global _industry_tree_cache
    _industry_tree_cache = None


# ==================== 兼容旧名（迁移期保留） ====================
# 旧模块用 SALARY_LEVEL_SORT_ORDER，新模块用 AREA_SORT_ORDER；保留旧名避免外部引用断裂
SALARY_LEVEL_SORT_ORDER = AREA_SORT_ORDER
