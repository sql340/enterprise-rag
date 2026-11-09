from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AdminJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force_reindex: bool = False


__all__ = ["AdminJobRequest"]
