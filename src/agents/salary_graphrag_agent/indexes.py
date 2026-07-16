# -*- coding: utf-8 -*-
"""创建 salary GraphRAG 所需的 fulltext / vector 索引（v3）。"""

from __future__ import annotations

from agents.salary_graphrag_agent.neo4j_client import get_salary_graphrag_neo4j_database
from utils.log_utils import get_logger

logger = get_logger(__name__)

VECTOR_DIM = 1024


def ensure_indexes(driver, database: str | None = None):
    db = database or get_salary_graphrag_neo4j_database()
    statements = [
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
        CREATE VECTOR INDEX areaNameEmbedding IF NOT EXISTS
        FOR (n:Area) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIM},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
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
        FOR (n:JobPosition) ON EACH [n.title, n.category]
        """,
        """
        CREATE FULLTEXT INDEX areaNameFulltext IF NOT EXISTS
        FOR (n:Area) ON EACH [n.name]
        """,
        """
        CREATE FULLTEXT INDEX chunkTextFulltext IF NOT EXISTS
        FOR (n:Chunk) ON EACH [n.text]
        """,
    ]

    for cypher in statements:
        driver.execute_query(cypher, database_=db)
    logger.info("fulltext + vector 索引已确保存在（v3：无 Skill/MarketTrend 索引）")
