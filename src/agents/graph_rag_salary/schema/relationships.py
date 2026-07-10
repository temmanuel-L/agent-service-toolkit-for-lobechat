# ==================== RELATIONSHIP TYPES ====================
# 主链：Report → Industry → SubSector → JobPosition → SalaryRange
# 无二级行业时：Industry → JobPosition（方案 A）
# 已移除 BELONGS_TO，避免与 HAS_POSITION 反向重复。
RELATIONSHIP_TYPES = [
    {
        "label": "HAS_INDUSTRY",
        "description": "报告包含某个一级行业板块。方向：Report → Industry。用于组织行业薪酬数据。"
    },
    {
        "label": "HAS_SUB_SECTOR",
        "description": (
            "一级行业包含二级行业/子领域（如银行、保险、一级市场）。"
            "方向：Industry → SubSector。"
        )
    },
    {
        "label": "HAS_POSITION",
        "description": (
            "行业或二级行业包含具体职位（如热门职位、高薪职位）。"
            "方向：Industry → JobPosition（无明确二级行业时）"
            "或 SubSector → JobPosition（有明确二级行业时）。"
            "有 SubSector 时不要再直连 Industry，避免双挂。"
            "这是职位归属的唯一边类型，不要再创建反向 BELONGS_TO。"
        )
    },
    {
        "label": "HAS_SALARY_RANGE",
        "description": (
            "职位对应薪酬范围（基本薪资 min-max，单位千元人民币）。"
            "方向：JobPosition → SalaryRange。薪酬事实必须经此边关联。"
        )
    },
    {
        "label": "REQUIRES_SKILL",
        "description": "职位或行业要求某项关键/新兴技能。方向：JobPosition → Skill 或 Industry → Skill。"
    },
    {
        "label": "HAS_TREND",
        "description": "报告或行业包含市场趋势/洞察。方向：Report → MarketTrend 或 Industry → MarketTrend。"
    },
]


# ==================== PATTERNS ====================
PATTERNS = [
    ("Report", "HAS_INDUSTRY", "Industry"),
    ("Industry", "HAS_SUB_SECTOR", "SubSector"),
    ("Industry", "HAS_POSITION", "JobPosition"),
    ("SubSector", "HAS_POSITION", "JobPosition"),
    ("JobPosition", "HAS_SALARY_RANGE", "SalaryRange"),
    ("JobPosition", "REQUIRES_SKILL", "Skill"),
    ("Industry", "REQUIRES_SKILL", "Skill"),
    ("Report", "HAS_TREND", "MarketTrend"),
    ("Industry", "HAS_TREND", "MarketTrend"),
]
