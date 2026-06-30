"""Text2CypherRetriever 的 few-shot 示例加载。"""

from typing import List


def load_examples() -> List[str]:
    """返回 Text2CypherRetriever 所需的 USER INPUT / QUERY 示例字符串列表。"""
    return [
        # ===================时间类=============================
        # === 季度类 ===
        "USER INPUT: '2023年第二季度持股价值前10的机构投资者' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE r.reportCalendarOrQuarter = '2023-06-30' "
        "RETURN m.managerName AS investor, r.value AS value, r.shares AS shares "
        "ORDER BY r.value DESC LIMIT 10",

        "USER INPUT: '2023年第一季度' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE r.reportCalendarOrQuarter = '2023-03-31' "
        "RETURN m.managerName AS investor, c.companyName AS company, r.value AS value "
        "ORDER BY r.value DESC LIMIT 20",

        # === 精确日期 ===
        "USER INPUT: '2023年6月30日，Vanguard 持有哪些公司' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(m.managerName) CONTAINS toUpper('Vanguard') "
        "  AND r.reportCalendarOrQuarter = '2023-06-30' "
        "RETURN c.companyName AS company, r.shares AS shares, r.value AS value "
        "ORDER BY r.value DESC",

        # === “之前”类时间过滤 ===
        "USER INPUT: '2023年5月份之前持股价值最高的投资者' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE r.reportCalendarOrQuarter < '2023-06-01' "
        "RETURN m.managerName AS investor, r.value AS value, r.reportCalendarOrQuarter AS quarter "
        "ORDER BY r.value DESC LIMIT 10",

        "USER INPUT: '2023年8月份之前' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE r.reportCalendarOrQuarter < '2023-09-01' "
        "RETURN m.managerName AS investor, c.companyName AS company, r.value AS value, r.reportCalendarOrQuarter AS quarter "
        "ORDER BY r.value DESC LIMIT 20",

        # === 聚合 + 时间 ===
        "USER INPUT: '2023年第二季度，Netapp 被多少家机构持有' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(c.companyName) CONTAINS toUpper('Netapp') "
        "  AND r.reportCalendarOrQuarter = '2023-06-30' "
        "RETURN count(DISTINCT m) AS investorCount",
        # ==================名字类===================
        "USER INPUT: '投资者Vanguard持有哪些公司' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(m.managerName) CONTAINS toUpper('Vanguard') "
        "RETURN c.companyName AS company, c.cusip AS cusip, r.shares AS shares, r.value AS value, r.reportCalendarOrQuarter AS quarter "
        "ORDER BY r.value DESC",

        "USER INPUT: 'Vanguard 和 BlackRock 持有哪些公司' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(m.managerName) CONTAINS toUpper('Vanguard') "
        "   OR toUpper(m.managerName) CONTAINS toUpper('BlackRock') "
        "RETURN m.managerName AS investor, c.companyName AS company, r.shares AS shares, r.value AS value, r.reportCalendarOrQuarter AS quarter "
        "ORDER BY r.value DESC",

        "USER INPUT: '2023年6月30日，Vanguard 持有哪些公司' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(m.managerName) CONTAINS toUpper('Vanguard') "
        "  AND r.reportCalendarOrQuarter = '2023-06-30' "
        "RETURN c.companyName AS company, r.shares AS shares, r.value AS value "
        "ORDER BY r.value DESC",

        "USER INPUT: 'Netapp 被多少家不同的机构投资者持有' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(c.companyName) CONTAINS toUpper('Netapp') "
        "RETURN count(DISTINCT m) AS investorCount",

        "USER INPUT: '2020年12月31日，有哪些机构持有 Netapp' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(c.companyName) CONTAINS toUpper('Netapp') "
        "  AND r.reportCalendarOrQuarter = '2020-12-31' "
        "RETURN m.managerName AS investor, r.shares AS shares, r.value AS value "
        "ORDER BY r.value DESC",

        "USER INPUT: '2023年第一季度，BlackRock 对 Netapp 的持股股数和市值分别是多少' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(m.managerName) CONTAINS toUpper('BlackRock') "
        "  AND toUpper(c.companyName) CONTAINS toUpper('Netapp') "
        "  AND r.reportCalendarOrQuarter = '2023-03-31' "
        "RETURN m.managerName AS investor, r.shares AS shares, r.value AS value",

        "USER INPUT: '2023年第二季度，Vanguard 和 BlackRock 谁对 Netapp 的持股价值更高？' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE toUpper(c.companyName) CONTAINS toUpper('Netapp') "
        "  AND r.reportCalendarOrQuarter = '2023-06-30' "
        "  AND (toUpper(m.managerName) CONTAINS toUpper('Vanguard') "
        "       OR toUpper(m.managerName) CONTAINS toUpper('BlackRock')) "
        "RETURN m.managerName AS investor, r.value AS value, r.shares AS shares "
        "ORDER BY r.value DESC",

        "USER INPUT: '2023年第二季度，除了 Vanguard 以外，对 Netapp 持股价值最高的 5 家机构是谁？' "
        "QUERY: MATCH (m:Manager)-[r:OWNS_STOCK_IN]->(c:Company) "
        "WHERE NOT toUpper(m.managerName) CONTAINS toUpper('Vanguard') "
        "  AND toUpper(c.companyName) CONTAINS toUpper('Netapp') "
        "  AND r.reportCalendarOrQuarter = '2023-06-30' "
        "RETURN m.managerName AS investor, r.value AS value, r.shares AS shares "
        "ORDER BY r.value DESC LIMIT 5",

        "USER INPUT: 'Netapp 这份 10-K 的 item7 主要讲了什么内容？' "
        "QUERY: MATCH (c:Company)-[:FILED]->(f:Form)-[:SECTION]->(ch:Chunk) "
        "WHERE toUpper(c.companyName) CONTAINS toUpper('Netapp') AND ch.f10kItem = 'item7' "
        "RETURN ch.text AS content, ch.chunkSeqId AS chunkSeqId ORDER BY ch.chunkSeqId",
    ]
