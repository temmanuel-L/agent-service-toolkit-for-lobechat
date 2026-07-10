# ==================== NODE TYPES ====================
# Ontology 主链：Report → Industry → SubSector → JobPosition → SalaryRange
# 无二级行业时：Industry → JobPosition（方案 A）
# 字符串外键属性仅作兜底，抽取时优先建边。
NODE_TYPES = [
    # ---------- 报告与行业 ----------
    {
        "label": "Report",
        "description": (
            "Michael Page 2026 中国大陆薪酬报告，包含整体就业市场趋势、11个行业薪酬指南及技能需求。"
            "报告年份 2026，由 Michael Page 发布。通过 HAS_INDUSTRY 关联行业，通过 HAS_TREND 关联整体趋势。"
        ),
        "properties": [
            {"name": "title", "type": "STRING", "required": True},
            {"name": "year", "type": "INTEGER"},
            {"name": "publisher", "type": "STRING"},
            {"name": "region", "type": "STRING"},
        ]
    },
    {
        "label": "Industry",
        "description": (
            "一级行业板块节点，代表报告中的主行业，例如：银行与金融服务、工程与制造、财务与会计、"
            "医疗与生命科学、人力资源与行政助理、法务、市场营销与电商、销售与零售、采购与供应链、"
            "科技、半导体。"
            "可通过 HAS_SUB_SECTOR 关联二级行业（如银行、保险、一级市场）；"
            "无明确二级行业的职位可直接通过 HAS_POSITION 挂到本节点。"
            "可通过 REQUIRES_SKILL / HAS_TREND 关联技能与趋势。overview 用于行业概述与人才需求描述。"
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "overview", "type": "STRING"},
        ]
    },
    {
        "label": "SubSector",
        "description": (
            "二级行业/子领域节点，挂在一级 Industry 之下，例如：银行、保险、一级市场、私募、资管等。"
            "必须通过 Industry-[:HAS_SUB_SECTOR]->SubSector 挂到父行业；"
            "有明确二级行业的职位应通过 SubSector-[:HAS_POSITION]->JobPosition 挂到本节点，"
            "不要再直连 Industry（避免双挂）。"
            "parent_industry 仅为无法建边时的兜底字符串，优先建边。"
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "parent_industry", "type": "STRING"},
        ]
    },

    # ---------- 职位与薪酬 ----------
    {
        "label": "JobPosition",
        "description": (
            "具体职位节点，如客户关系经理、投资经理、产品总监、营销负责人、财务负责人等。"
            "归属规则：有明确二级行业时挂 SubSector-[:HAS_POSITION]->JobPosition；"
            "无二级行业时挂 Industry-[:HAS_POSITION]->JobPosition。"
            "并通过 JobPosition-[:HAS_SALARY_RANGE]->SalaryRange 挂薪酬。"
            "category 区分热门职位/高薪职位等；"
            "sub_sector / industry_name 仅在无法建立归属边时作为兜底字符串，优先建边而非只填字段。"
        ),
        "properties": [
            {"name": "title", "type": "STRING", "required": True},
            {"name": "category", "type": "STRING"},
            {"name": "sub_sector", "type": "STRING"},
            {"name": "industry_name", "type": "STRING"},
        ]
    },
    {
        "label": "SalaryRange",
        "description": (
            "薪酬范围节点，记录基本薪资范围（单位：千元人民币），包含最低值、最高值、地区口径等。"
            "必须通过 JobPosition-[:HAS_SALARY_RANGE]->SalaryRange 挂到对应职位，不要孤立存在。"
            "position_title 仅为冗余展示字段，不能替代 HAS_SALARY_RANGE 边。"
            "level 为薪酬地理口径（非职级）：全国平均 | 华东 | 华北 | 华南；区域三列表应为同一岗位生成多条 SalaryRange。"
        ),
        "properties": [
            {"name": "min_value", "type": "FLOAT"},
            {"name": "max_value", "type": "FLOAT"},
            {"name": "unit", "type": "STRING"},
            {
                "name": "level",
                "type": "STRING",
                "description": "薪酬地理口径：全国平均 | 华东 | 华北 | 华南（非职级）",
            },
            {"name": "position_title", "type": "STRING"},
        ]
    },

    # ---------- 技能与趋势 ----------
    {
        "label": "Skill",
        "description": (
            "技能节点，记录关键技能和新兴技能，如业务拓展能力、AI驱动的自动化、募资能力等。"
            "优先通过 JobPosition-[:REQUIRES_SKILL]->Skill 或 Industry-[:REQUIRES_SKILL]->Skill 建边；"
            "industry 字符串仅作无法建边时的兜底。"
        ),
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "type", "type": "STRING"},
            {"name": "industry", "type": "STRING"},
        ]
    },
    {
        "label": "MarketTrend",
        "description": (
            "市场趋势或洞察节点，记录整体或行业特定趋势，如数字化转型、AI应用、人才战略等。"
            "优先通过 Report-[:HAS_TREND]->MarketTrend 或 Industry-[:HAS_TREND]->MarketTrend 建边；"
            "related_industry 字符串仅作无法建边时的兜底。"
        ),
        "properties": [
            {"name": "trend_name", "type": "STRING", "required": True},
            {"name": "description", "type": "STRING"},
            {"name": "related_industry", "type": "STRING"},
        ]
    },
]
