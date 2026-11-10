from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AdminQualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_status: Literal["unreviewed", "good", "needs_rework", "blocked"]
    note: str = Field(default="", max_length=2000)


__all__ = ["AdminQualityRequest"]
