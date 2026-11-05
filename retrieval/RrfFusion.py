from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .vector_store import VectorSearchResult


@dataclass(slots=True)
class _FusedCandidate:
    result: VectorSearchResult
    score: float = 0.0
    route_scores: dict[str, float] = field(default_factory=dict)
    route_ranks: dict[str, int] = field(default_factory=dict)


class RrfFusion:
    """使用 Reciprocal Rank Fusion 融合多路检索结果。"""

    def __init__(self, *, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k 必须为正整数")
        self.k = k

    def fuse(
        self,
        routes: Mapping[str, Sequence[VectorSearchResult]],
        *,
        limit: int,
        weights: Mapping[str, float] | None = None,
    ) -> list[VectorSearchResult]:
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        if not routes:
            return []

        resolved_weights = dict(weights or {})
        if any(weight < 0 for weight in resolved_weights.values()):
            raise ValueError("各检索路权重不能为负数")

        fused: dict[str, _FusedCandidate] = {}
        for route_name, results in routes.items():
            weight = float(resolved_weights.get(route_name, 1.0))
            for rank, result in enumerate(results, start=1):
                entry = fused.setdefault(
                    result.id,
                    _FusedCandidate(result=result),
                )
                entry.score += weight / (self.k + rank)
                entry.route_scores[route_name] = result.score
                entry.route_ranks[route_name] = rank

        ranked = sorted(
            fused.values(),
            key=lambda entry: (-entry.score, entry.result.id),
        )[:limit]
        return [
            VectorSearchResult(
                id=entry.result.id,
                score=round(entry.score, 8),
                payload=dict(entry.result.payload),
                retrieval_mode="hybrid_rrf",
                route_scores=dict(entry.route_scores),
                route_ranks=dict(entry.route_ranks),
            )
            for entry in ranked
        ]


__all__ = ["RrfFusion"]
