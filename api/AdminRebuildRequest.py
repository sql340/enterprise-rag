from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class AdminRebuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quality_status: Literal["unreviewed", "needs_rework"] = "needs_rework"


__all__ = ["AdminRebuildRequest"]
