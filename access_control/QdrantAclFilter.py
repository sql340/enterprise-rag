from __future__ import annotations

try:
    from qdrant_client import models
except ImportError as exc:  # pragma: no cover - qdrant-client 是运行依赖
    models = None  # type: ignore[assignment]
    _QDRANT_IMPORT_ERROR: ImportError | None = exc
else:
    _QDRANT_IMPORT_ERROR = None

from .AccessControlError import AccessControlError
from .Classification import Classification
from .Principal import Principal


class QdrantAclFilter:
    """把 Principal 转换为 Qdrant 强制访问过滤条件。"""

    def __init__(self, principal: Principal) -> None:
        if models is None:
            raise AccessControlError(
                "ACL 过滤需要安装 qdrant-client"
            ) from _QDRANT_IMPORT_ERROR
        self.principal = principal

    def build(self) -> "models.Filter":
        must: list[object] = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=self.principal.tenant_id),
            ),
            models.FieldCondition(
                key="classification",
                range=models.Range(lte=int(self.principal.clearance)),
            ),
            models.FieldCondition(
                key="document_status",
                match=models.MatchValue(value="active"),
            ),
        ]

        should: list[object] = []
        if not self.principal.is_admin:
            should = [
                models.FieldCondition(
                    key="classification",
                    match=models.MatchValue(value=int(Classification.PUBLIC)),
                ),
                models.FieldCondition(
                    key="allowed_users",
                    match=models.MatchAny(any=[self.principal.user_id]),
                ),
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(value=self.principal.user_id),
                ),
            ]
            if self.principal.roles:
                should.append(
                    models.FieldCondition(
                        key="allowed_roles",
                        match=models.MatchAny(any=sorted(self.principal.roles)),
                    )
                )
            if self.principal.groups:
                should.append(
                    models.FieldCondition(
                        key="allowed_groups",
                        match=models.MatchAny(any=sorted(self.principal.groups)),
                    )
                )

        return models.Filter(must=must, should=should)


__all__ = ["QdrantAclFilter"]
