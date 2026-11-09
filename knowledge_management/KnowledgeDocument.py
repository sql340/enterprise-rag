from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_DOCUMENT_STATUSES = frozenset(
    {"uploaded", "queued", "running", "succeeded", "failed", "archived", "deleted"}
)
QUALITY_STATUSES = frozenset({"unreviewed", "good", "needs_rework", "blocked"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class KnowledgeDocument:
    """知识文档清单，记录来源、构建配置、ACL 和质量状态。"""

    document_id: str
    tenant_id: str
    source: str
    file_name: str
    file_type: str
    storage_path: str
    content_hash: str
    acl: dict[str, Any]
    chunk_strategy: str = "recursive"
    chunk_options: dict[str, Any] = field(default_factory=dict)
    parser_version: str = "v1"
    embedding_fingerprint: str = ""
    build_fingerprint: str = ""
    status: str = "uploaded"
    quality_status: str = "unreviewed"
    quality_note: str = ""
    indexed_document_ids: tuple[str, ...] = field(default_factory=tuple)
    chunk_count: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.document_id = str(self.document_id).strip()
        self.tenant_id = str(self.tenant_id).strip()
        if not self.document_id or not self.tenant_id:
            raise ValueError("document_id 和 tenant_id 不能为空")
        self.status = str(self.status).strip().lower()
        if self.status not in _DOCUMENT_STATUSES:
            raise ValueError(
                "status 必须是 uploaded/queued/running/succeeded/failed/archived/deleted"
            )
        self.quality_status = str(self.quality_status).strip().lower()
        if self.quality_status not in QUALITY_STATUSES:
            raise ValueError("quality_status 必须是 unreviewed/good/needs_rework/blocked")
        self.indexed_document_ids = tuple(self.indexed_document_ids)
        self.acl = dict(self.acl)
        self.chunk_options = dict(self.chunk_options)
        self.chunk_count = int(self.chunk_count)

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self, *, include_storage_path: bool = False) -> dict[str, Any]:
        value = {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "source": self.source,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "acl": dict(self.acl),
            "chunk_strategy": self.chunk_strategy,
            "chunk_options": dict(self.chunk_options),
            "parser_version": self.parser_version,
            "embedding_fingerprint": self.embedding_fingerprint,
            "build_fingerprint": self.build_fingerprint,
            "status": self.status,
            "quality_status": self.quality_status,
            "quality_note": self.quality_note,
            "indexed_document_ids": list(self.indexed_document_ids),
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_storage_path:
            value["storage_path"] = self.storage_path
        return value


__all__ = ["KnowledgeDocument", "QUALITY_STATUSES"]
