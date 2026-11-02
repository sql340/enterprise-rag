from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class _Span:
    start: int
    end: int


@dataclass(slots=True)
class _Draft:
    span: _Span
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "chunk"
    parent_index: int | None = None


@dataclass(slots=True)
class _Section:
    span: _Span
    level: int
    title: str
    heading_path: tuple[str, ...]
