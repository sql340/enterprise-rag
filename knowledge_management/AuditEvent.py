from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class AuditEvent:
    """知识管理操作审计记录。"""

    tenant_id: str
    actor_id: str
    action: str
    target_type: str
    target_id: str
    detail: dict[str, Any] = field(default_factory=dict)
    event_id: int | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "action": self.action,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "detail": dict(self.detail),
            "created_at": self.created_at,
        }


__all__ = ["AuditEvent"]
