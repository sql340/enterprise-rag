from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AdminChunkConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_strategy: str = Field(min_length=1, max_length=64)
    chunk_options: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AdminChunkConfigRequest"]
