from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GraphIndexingResult:
    """一次图谱构建的结果摘要。"""

    indexed_chunks: int
    skipped_chunks: int
    entity_count: int
    relation_count: int
    failures: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexed_chunks": self.indexed_chunks,
            "skipped_chunks": self.skipped_chunks,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "failures": [dict(item) for item in self.failures],
        }


__all__ = ["GraphIndexingResult"]
