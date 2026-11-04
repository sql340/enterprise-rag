from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RagStreamEvent:
    """RAG 流式响应中的单个事件。"""

    event: str
    data: dict[str, Any] = field(default_factory=dict)


__all__ = ["RagStreamEvent"]
