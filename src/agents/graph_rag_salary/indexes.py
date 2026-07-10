# -*- coding: utf-8 -*-
"""创建 salary GraphRAG 所需的 fulltext / vector 索引（IF NOT EXISTS）。"""

from __future__ import annotations

from agents.graph_rag_salary.neo4j_client import get_salary_neo4j_database
from utils.log_utils import get_logger

logger = get_logger(__name__)

VECTOR_DIM = 1024


def ensure_indexes(driver, database: str | None = None) -> None:
    db = database or get_salary_neo4j_database()
    statements = [
        # Chunk 向量（兜底检索）
        f"""
        CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
        FOR (n:Chunk) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        # 实体向量
        f"""
        CREATE VECTOR INDEX industryNameEmbedding IF NOT EXISTS
        FOR (n:Industry) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        f"""
        CREATE VECTOR INDEX subSectorNameEmbedding IF NOT EXISTS
        FOR (n:SubSector) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        f"""
        CREATE VECTOR INDEX jobTitleEmbedding IF NOT EXISTS
        FOR (n:JobPosition) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        f"""
        CREATE VECTOR INDEX skillNameEmbedding IF NOT EXISTS
        FOR (n:Skill) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        # Fulltext
        """
        CREATE FULLTEXT INDEX industryNameFulltext IF NOT EXISTS
        FOR (n:Industry) ON EACH [n.name]
        """,
        """
        CREATE FULLTEXT INDEX subSectorNameFulltext IF NOT EXISTS
        FOR (n:SubSector) ON EACH [n.name]
        """,
        """
        CREATE FULLTEXT INDEX jobTitleFulltext IF NOT EXISTS
        FOR (n:JobPosition) ON EACH [n.title, n.category, n.sub_sector]
        """,
        """
        CREATE FULLTEXT INDEX skillNameFulltext IF NOT EXISTS
        FOR (n:Skill) ON EACH [n.name]
        """,
        """
        CREATE FULLTEXT INDEX trendNameFulltext IF NOT EXISTS
        FOR (n:MarketTrend) ON EACH [n.trend_name, n.description]
        """,
    ]

    for cypher in statements:
        driver.execute_query(cypher, database_=db)
    logger.info("fulltext + vector 索引已确保存在")
