from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_JOB_STATUSES = frozenset({"pending", "running", "succeeded", "failed"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class IngestionJob:
    """持久化的知识构建任务。"""

    job_id: str
    tenant_id: str
    document_id: str
    status: str = "pending"
    progress: int = 0
    message: str = ""
    error: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.status = str(self.status).strip().lower()
        if self.status not in _JOB_STATUSES:
            raise ValueError("job.status 必须是 pending/running/succeeded/failed")
        self.progress = max(0, min(100, int(self.progress)))
        self.result = dict(self.result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": dict(self.result),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


__all__ = ["IngestionJob"]
