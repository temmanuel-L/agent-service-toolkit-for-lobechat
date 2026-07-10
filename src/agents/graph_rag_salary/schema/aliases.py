# -*- coding: utf-8 -*-
"""行业 / 岗位别名词典，供实体链接使用。

key 为图中期望的规范名（或接近规范名），value 为用户可能使用的别名列表。
链接时会双向匹配：用户词命中任一别名或规范名即可。
"""

from __future__ import annotations

# 规范一级行业名 → 别名（不含纯二级行业词，避免「保险」被解析成一级 Industry）
INDUSTRY_ALIASES = {
    "银行与金融服务": [
        "银行", "金融", "金融服务", "银行与金融", "banking",
    ],
    "工程与制造": ["工程", "制造", "制造业", "工程制造"],
    "财务与会计": ["财务", "会计", "财会", "财务会计"],
    "医疗与生命科学": ["医疗", "生命科学", "医药", "医疗健康"],
    "人力资源与行政助理": ["人力资源", "HR", "行政", "行政助理", "人事"],
    "法务": ["法律", "法务合规", "legal"],
    "市场营销与电商": [
        "市场营销", "营销", "电商", "消费品营销", "消费品", "市场", "marketing",
    ],
    "销售与零售": ["销售", "零售", "sales"],
    "采购与供应链": ["采购", "供应链", "供应链管理"],
    "科技": ["科技行业", "互联网", "IT", "人工智能", "AI", "软件"],
    "半导体": ["半导体材料", "芯片", "semiconductor"],
}

# 规范二级行业名 → 别名
SUB_SECTOR_ALIASES = {
    "银行": ["银行业", "商业银行"],
    "保险": ["保险业", "保险行业"],
    "一级市场": ["私募", "PE", "VC", "创投"],
    "资管": ["资产管理", "财富管理"],
}

# 二级行业规范名 → 父一级行业规范名（后处理建 HAS_SUB_SECTOR）
SUB_SECTOR_PARENT_INDUSTRY = {
    "银行": "银行与金融服务",
    "保险": "银行与金融服务",
    "一级市场": "银行与金融服务",
    "资管": "银行与金融服务",
}

# 规范岗位名 → 别名
JOB_ALIASES = {
    "营销负责人": ["市场负责人", "营销总监", "市场总监", "CMO", "营销主管", "市场主管"],
    "财务负责人": ["财务责任人", "财务总监", "CFO", "财务主管", "财务经理"],
    "AI负责人": ["人工智能负责人", "AI总监", "人工智能总监", "AI主管"],
    "研发负责人": ["研发总监", "技术负责人", "研发主管", "CTO"],
    "人力资源负责人": ["HR负责人", "人力资源总监", "CHO", "人事负责人"],
    "销售负责人": ["销售总监", "销售主管", "销售经理"],
    "产品总监": ["产品负责人", "产品经理负责人", "CPO"],
    "首席运营官": ["COO", "运营负责人", "运营总监"],
}

# 技能别名（可选）
SKILL_ALIASES = {
    "人工智能": ["AI", "机器学习", "深度学习"],
    "数字化转型": ["数字化", "digital transformation"],
}

# 薪酬地理口径（SalaryRange.level）规范名 → 别名；非职级
SALARY_LEVEL_ALIASES = {
    "全国平均": ["全国", "nationwide", "全国均值", "全国水平"],
    "华东": ["华东地区", "华东区"],
    "华北": ["华北地区", "华北区"],
    "华南": ["华南地区", "华南区"],
}

# 检索展示排序：全国平均优先，再空 level，再三大区
SALARY_LEVEL_SORT_ORDER = ("全国平均", "", "华东", "华北", "华南")


def build_alias_lookup(alias_map: dict) -> dict:
    """构建 小写别名/规范名 → 规范名 的查找表。"""
    lookup = {}
    for canonical, aliases in alias_map.items():
        lookup[canonical.lower()] = canonical
        for alias in aliases:
            lookup[alias.lower()] = canonical
    return lookup


INDUSTRY_ALIAS_LOOKUP = build_alias_lookup(INDUSTRY_ALIASES)
SUB_SECTOR_ALIAS_LOOKUP = build_alias_lookup(SUB_SECTOR_ALIASES)
JOB_ALIAS_LOOKUP = build_alias_lookup(JOB_ALIASES)
SKILL_ALIAS_LOOKUP = build_alias_lookup(SKILL_ALIASES)
SALARY_LEVEL_ALIAS_LOOKUP = build_alias_lookup(SALARY_LEVEL_ALIASES)


# 这些词应优先识别为 SubSector，而非一级 Industry
SUB_SECTOR_HINTS = {
    "保险", "保险业", "一级市场", "私募", "资管", "财富管理", "银行",
}


def resolve_alias(text: str, lookup: dict) -> str | None:
    """精确别名解析；命中则返回规范名，否则 None。"""
    if not text:
        return None
    key = text.strip().lower()
    if key in lookup:
        return lookup[key]
    # 子串包含：较长别名优先
    candidates = [(k, v) for k, v in lookup.items() if k in key or key in k]
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0][1]


def is_sub_sector_hint(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip()
    return t in SUB_SECTOR_HINTS or any(h in t for h in SUB_SECTOR_HINTS)


def normalize_sub_sector_hint(text: str | None) -> str | None:
    """把「保险行业」等归一成图中更可能出现的子领域词，如「保险」。"""
    if not text or not is_sub_sector_hint(text):
        return None
    # 优先走别名表得到规范名
    canonical = resolve_alias(text, SUB_SECTOR_ALIAS_LOOKUP)
    if canonical:
        return canonical
    t = text.strip()
    hits = [h for h in SUB_SECTOR_HINTS if h == t or h in t]
    if not hits:
        return t
    hits.sort(key=len)
    return hits[0]


def parent_industry_for_sub_sector(sub_sector_name: str | None) -> str | None:
    """返回二级行业对应的父一级行业规范名。"""
    if not sub_sector_name:
        return None
    canonical = resolve_alias(sub_sector_name, SUB_SECTOR_ALIAS_LOOKUP) or sub_sector_name.strip()
    return SUB_SECTOR_PARENT_INDUSTRY.get(canonical)


def normalize_salary_level(text: str | None) -> str:
    """把用户/抽取的地区口径归一成 SalaryRange.level 规范名；空则返回空串。"""
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    canonical = resolve_alias(t, SALARY_LEVEL_ALIAS_LOOKUP)
    if canonical:
        return canonical
    # 精确命中规范名
    if t in SALARY_LEVEL_ALIASES:
        return t
    return t


def extract_region_from_text(text: str | None) -> str | None:
    """从问句中抽出首个薪酬地区口径；未提及则 None。"""
    if not text:
        return None
    q = text.strip()
    # 较长别名优先，避免「全国」抢在「全国平均」前误匹配时仍能归一
    candidates: list[tuple[str, str]] = []
    for canonical, aliases in SALARY_LEVEL_ALIASES.items():
        for alias in [canonical, *aliases]:
            if alias and alias in q:
                candidates.append((alias, canonical))
    if not candidates:
        return None
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0][1]
