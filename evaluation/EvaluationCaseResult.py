from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationCaseResult:
    case_id: str
    mode: str
    question: str
    metrics: dict[str, float] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "question": self.question,
            "metrics": dict(self.metrics),
            "retrieved": [dict(item) for item in self.retrieved],
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


__all__ = ["EvaluationCaseResult"]
