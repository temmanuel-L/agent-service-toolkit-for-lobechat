"""
RAG 检索结果融合（RRF 等）相关工具。

目前仅包含加权 Reciprocal Rank Fusion 实现，后续可在本模块扩展更多融合策略。
"""

from __future__ import annotations

from typing import List

from llama_index.core.schema import NodeWithScore


def reciprocal_rank_fusion(
    vector_results: List[NodeWithScore],
    bm25_results: List[NodeWithScore],
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 0.4,
) -> List[NodeWithScore]:
    """
    Weighted Reciprocal Rank Fusion (W-RRF).

    Score = (1 - w) * RR_vector + w * RR_bm25
    RR = 1 / (k + rank)
    """
    score_map: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    vec_w = 1.0 - bm25_weight
    for rank, nws in enumerate(vector_results):
        nid = nws.node.node_id
        score = vec_w * (1.0 / (rrf_k + rank + 1))
        score_map[nid] = score_map.get(nid, 0.0) + score
        if nid not in node_map:
            node_map[nid] = nws

    for rank, nws in enumerate(bm25_results):
        nid = nws.node.node_id
        score = bm25_weight * (1.0 / (rrf_k + rank + 1))
        score_map[nid] = score_map.get(nid, 0.0) + score
        if nid not in node_map:
            node_map[nid] = nws

    sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
    fused: List[NodeWithScore] = []
    for nid in sorted_ids[:top_k]:
        nws = node_map[nid]
        fused.append(NodeWithScore(node=nws.node, score=score_map[nid]))

    return fused


__all__ = [
    "reciprocal_rank_fusion",
]

