from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from access_control import ACL_PAYLOAD_FIELDS


class QueryRequest(BaseModel):
    """在线查询请求。"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "报销需要准备哪些材料？",
                "top_k": 5,
                "filters": {"file_type": "pdf", "chunk_role": "child"},
                "include_sources": True,
            }
        },
    )

    question: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    filters: dict[str, Any] | None = None
    include_sources: bool = True
    retrieval_mode: Literal["dense", "hybrid", "graph", "hybrid_graph"] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)

    @field_validator("question")
    @classmethod
    def _normalise_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value

    @field_validator("filters")
    @classmethod
    def _reject_acl_filters(cls, value: dict[str, Any] | None):
        if not value:
            return value
        reserved = sorted(set(value) & set(ACL_PAYLOAD_FIELDS))
        if reserved:
            raise ValueError(
                "filters 不能包含权限字段，权限由服务端认证上下文决定: "
                + ", ".join(reserved)
            )
        return value


__all__ = ["QueryRequest"]
