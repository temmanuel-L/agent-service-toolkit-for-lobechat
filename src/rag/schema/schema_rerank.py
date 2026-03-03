"""
RAG 重排（Rerank）阶段的数据模型。

主要用于描述：
- 待重排的候选段落列表；
- Rerank 服务返回的分数及其裁剪、筛选后的结果。
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RerankConfig(BaseModel):
    """Rerank 行为配置。"""

    enabled: bool = Field(
        default=False,
        description="是否启用 Rerank。",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description="Rerank 后保留的候选数量上限。",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        description="可选的最低相关性分数阈值，小于该值的结果将被丢弃。",
    )


class RerankItem(BaseModel):
    """参与重排的单条候选段落。"""

    text: str = Field(description="候选段落文本。")
    score: float = Field(
        default=0.0,
        description="进入 Rerank 之前的原始得分（例如向量/BM25 融合分数）。",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="与段落相关的元数据。",
    )


class RerankResult(BaseModel):
    """Rerank 阶段结果。"""

    items: List[RerankItem] = Field(
        default_factory=list,
        description="按 Rerank 后分数排序的候选列表。",
    )


__all__ = [
    "RerankConfig",
    "RerankItem",
    "RerankResult",
]

