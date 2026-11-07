from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .Classification import Classification


ADMIN_ROLES = frozenset({"platform_admin", "knowledge_admin"})


def _normalise_values(values: Iterable[str] | None) -> frozenset[str]:
    if isinstance(values, str):
        values = values.split(",")
    return frozenset(
        str(item).strip().lower()
        for item in (values or ())
        if str(item).strip()
    )


@dataclass(frozen=True, slots=True)
class Principal:
    """已经通过认证的主体身份。"""

    tenant_id: str
    user_id: str
    roles: frozenset[str] = field(default_factory=frozenset)
    groups: frozenset[str] = field(default_factory=frozenset)
    clearance: Classification = Classification.PUBLIC
    display_name: str | None = None

    def __post_init__(self) -> None:
        tenant_id = str(self.tenant_id).strip()
        user_id = str(self.user_id).strip()
        if not tenant_id:
            raise ValueError("principal.tenant_id 不能为空")
        if not user_id:
            raise ValueError("principal.user_id 不能为空")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "roles", _normalise_values(self.roles))
        object.__setattr__(self, "groups", _normalise_values(self.groups))
        object.__setattr__(self, "clearance", Classification.parse(self.clearance))
        if self.display_name is not None:
            display_name = str(self.display_name).strip()
            object.__setattr__(self, "display_name", display_name or None)

    @property
    def is_admin(self) -> bool:
        return bool(self.roles & ADMIN_ROLES)


__all__ = ["ADMIN_ROLES", "Principal"]
