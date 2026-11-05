from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """一次入库操作的结果摘要。"""

    indexed_chunks: int
    document_ids: tuple[str, ...]
    sources: tuple[str, ...]
    collection_name: str
    total_chunks: int = 0
    embedded_chunks: int = 0
    skipped_chunks: int = 0
    stale_deleted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexed_chunks": self.indexed_chunks,
            "document_ids": list(self.document_ids),
            "sources": list(self.sources),
            "collection_name": self.collection_name,
            "total_chunks": self.total_chunks,
            "embedded_chunks": self.embedded_chunks,
            "skipped_chunks": self.skipped_chunks,
            "stale_deleted": self.stale_deleted,
        }


__all__ = ["IndexingResult"]
