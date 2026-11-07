from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .Classification import Classification
from .Principal import Principal


ACL_PAYLOAD_FIELDS = (
    "tenant_id",
    "classification",
    "allowed_roles",
    "allowed_groups",
    "allowed_users",
    "owner_id",
    "document_status",
    "acl_version",
)

_DOCUMENT_STATUSES = frozenset({"active", "archived", "deleted"})


def _normalise_values(values: Iterable[str] | None) -> frozenset[str]:
    if isinstance(values, str):
        values = values.split(",")
    return frozenset(
        str(item).strip().lower()
        for item in (values or ())
        if str(item).strip()
    )


def _normalise_exact_values(values: Iterable[str] | None) -> frozenset[str]:
    if isinstance(values, str):
        values = (values,)
    return frozenset(
        str(item).strip() for item in (values or ()) if str(item).strip()
    )


@dataclass(frozen=True, slots=True)
class DocumentAcl:
    """文档级访问控制信息，知识块继承该权限。"""

    tenant_id: str
    classification: Classification = Classification.INTERNAL
    allowed_roles: frozenset[str] = field(default_factory=frozenset)
    allowed_groups: frozenset[str] = field(default_factory=frozenset)
    allowed_users: frozenset[str] = field(default_factory=frozenset)
    owner_id: str | None = None
    document_status: str = "active"
    acl_version: int = 1

    def __post_init__(self) -> None:
        tenant_id = str(self.tenant_id).strip()
        if not tenant_id:
            raise ValueError("acl.tenant_id 不能为空")
        status = str(self.document_status).strip().lower()
        if status not in _DOCUMENT_STATUSES:
            raise ValueError("document_status 必须是 active、archived 或 deleted")
        if self.acl_version <= 0:
            raise ValueError("acl_version 必须为正整数")

        owner_id = str(self.owner_id).strip() if self.owner_id is not None else ""
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "classification", Classification.parse(self.classification))
        object.__setattr__(self, "allowed_roles", _normalise_values(self.allowed_roles))
        object.__setattr__(self, "allowed_groups", _normalise_values(self.allowed_groups))
        object.__setattr__(
            self, "allowed_users", _normalise_exact_values(self.allowed_users)
        )
        object.__setattr__(self, "owner_id", owner_id or None)
        object.__setattr__(self, "document_status", status)

    def to_payload(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "classification": int(self.classification),
            "allowed_roles": sorted(self.allowed_roles),
            "allowed_groups": sorted(self.allowed_groups),
            "allowed_users": sorted(self.allowed_users),
            "owner_id": self.owner_id,
            "document_status": self.document_status,
            "acl_version": self.acl_version,
        }

    def allows(self, principal: Principal) -> bool:
        """判断主体是否可以读取该文档；默认拒绝。"""

        if not isinstance(principal, Principal):
            raise TypeError("principal 必须是 Principal")
        if self.tenant_id != principal.tenant_id:
            return False
        if self.document_status != "active":
            return False
        if int(self.classification) > int(principal.clearance):
            return False
        if principal.is_admin:
            return True
        if self.classification == Classification.PUBLIC:
            return True
        if self.owner_id == principal.user_id:
            return True
        if principal.user_id in self.allowed_users:
            return True
        return bool(
            principal.roles & self.allowed_roles
            or principal.groups & self.allowed_groups
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DocumentAcl":
        return cls(
            tenant_id=str(payload.get("tenant_id") or ""),
            classification=Classification.parse(payload.get("classification")),
            allowed_roles=payload.get("allowed_roles") or (),
            allowed_groups=payload.get("allowed_groups") or (),
            allowed_users=payload.get("allowed_users") or (),
            owner_id=payload.get("owner_id"),
            document_status=str(payload.get("document_status") or "active"),
            acl_version=int(payload.get("acl_version") or 1),
        )

    @staticmethod
    def payload_signature(payload: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
        if not payload or not any(field in payload for field in ACL_PAYLOAD_FIELDS):
            return None
        try:
            classification = int(Classification.parse(payload.get("classification")))
            return (
                str(payload.get("tenant_id") or ""),
                classification,
                tuple(
                    sorted(
                        str(value).lower()
                        for value in payload.get("allowed_roles") or ()
                    )
                ),
                tuple(
                    sorted(
                        str(value).lower()
                        for value in payload.get("allowed_groups") or ()
                    )
                ),
                tuple(
                    sorted(str(value) for value in payload.get("allowed_users") or ())
                ),
                str(payload.get("owner_id") or ""),
                str(payload.get("document_status") or "").lower(),
                int(payload.get("acl_version") or 0),
            )
        except (TypeError, ValueError):
            return None


__all__ = ["ACL_PAYLOAD_FIELDS", "DocumentAcl"]
