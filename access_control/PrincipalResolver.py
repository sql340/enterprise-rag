from __future__ import annotations

import hmac
from collections.abc import Mapping

from .AccessControlError import AccessControlError
from .Classification import Classification
from .Principal import Principal


class PrincipalResolver:
    """从可信网关注入的请求头解析登录身份。"""

    TENANT_HEADER = "x-tenant-id"
    USER_HEADER = "x-user-id"
    ROLES_HEADER = "x-user-roles"
    GROUPS_HEADER = "x-user-groups"
    CLEARANCE_HEADER = "x-user-clearance"
    NAME_HEADER = "x-user-name"
    PROXY_TOKEN_HEADER = "x-rag-proxy-token"

    def __init__(self, *, proxy_token: str | None = None) -> None:
        self.proxy_token = proxy_token.strip() if proxy_token else None

    def resolve(self, headers: Mapping[str, str]) -> Principal:
        values = {str(key).lower(): str(value) for key, value in headers.items()}
        if self.proxy_token is not None:
            supplied = values.get(self.PROXY_TOKEN_HEADER, "")
            if not hmac.compare_digest(supplied, self.proxy_token):
                raise AccessControlError("身份代理令牌无效")

        tenant_id = values.get(self.TENANT_HEADER, "").strip()
        user_id = values.get(self.USER_HEADER, "").strip()
        if not tenant_id or not user_id:
            raise AccessControlError(
                "缺少可信身份头 X-Tenant-Id 或 X-User-Id"
            )

        try:
            clearance = Classification.parse(values.get(self.CLEARANCE_HEADER))
        except ValueError as exc:
            raise AccessControlError(str(exc)) from exc

        return Principal(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=_split_header(values.get(self.ROLES_HEADER)),
            groups=_split_header(values.get(self.GROUPS_HEADER)),
            clearance=clearance,
            display_name=values.get(self.NAME_HEADER),
        )


def _split_header(value: str | None) -> frozenset[str]:
    return frozenset(
        item.strip().lower()
        for item in (value or "").split(",")
        if item.strip()
    )


__all__ = ["PrincipalResolver"]
