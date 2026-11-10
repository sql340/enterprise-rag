from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from access_control import Classification, DocumentAcl, Principal


class AdminAclRequest(BaseModel):
    """文档 ACL 更新请求。"""

    model_config = ConfigDict(extra="forbid")

    classification: str = "internal"
    allowed_roles: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    owner_id: str | None = None
    acl_version: int = Field(default=1, ge=1)

    @field_validator("classification")
    @classmethod
    def _validate_classification(cls, value: str) -> str:
        Classification.parse(value)
        return value

    def to_acl(self, principal: Principal) -> DocumentAcl:
        return DocumentAcl(
            tenant_id=principal.tenant_id,
            classification=self.classification,
            allowed_roles=self.allowed_roles,
            allowed_groups=self.allowed_groups,
            allowed_users=self.allowed_users,
            owner_id=self.owner_id or principal.user_id,
            document_status="active",
            acl_version=self.acl_version,
        )


__all__ = ["AdminAclRequest"]
