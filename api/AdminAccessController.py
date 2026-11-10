from __future__ import annotations

from collections.abc import Mapping

from access_control import AccessControlError, Principal, PrincipalResolver


class AdminAccessController:
    """管理端的可信身份解析；没有配置代理令牌时默认拒绝。"""

    def __init__(
        self,
        *,
        proxy_token: str | None,
        resolver: PrincipalResolver | None = None,
    ) -> None:
        self.proxy_token = str(proxy_token).strip() if proxy_token else None
        self.resolver = resolver or PrincipalResolver(proxy_token=self.proxy_token)

    def resolve(self, headers: Mapping[str, str]) -> Principal:
        if not self.proxy_token:
            raise AccessControlError(
                "管理接口要求配置 RAG_ACL_PROXY_TOKEN，拒绝无认证访问"
            )
        return self.resolver.resolve(headers)


__all__ = ["AdminAccessController"]
