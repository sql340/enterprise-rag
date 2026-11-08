from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GraphSearchHit:
    """图谱检索命中的知识块。"""

    chunk_id: str
    document_id: str
    score: float
    matched_entities: tuple[str, ...] = field(default_factory=tuple)


__all__ = ["GraphSearchHit"]
