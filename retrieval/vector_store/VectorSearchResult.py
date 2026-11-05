from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VectorSearchResult:
    """一次向量检索命中的知识块。"""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    retrieval_mode: str = "dense"
    route_scores: dict[str, float] = field(default_factory=dict)
    route_ranks: dict[str, int] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return str(self.payload.get("content", ""))

    @property
    def chunk_id(self) -> str:
        return str(self.payload.get("chunk_id") or self.id)

    @property
    def document_id(self) -> str:
        return str(self.payload.get("document_id", ""))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "payload": dict(self.payload),
            "retrieval_mode": self.retrieval_mode,
            "route_scores": dict(self.route_scores),
            "route_ranks": dict(self.route_ranks),
        }


__all__ = ["VectorSearchResult"]
