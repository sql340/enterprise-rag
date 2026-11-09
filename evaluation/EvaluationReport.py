from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .EvaluationCaseResult import EvaluationCaseResult


@dataclass(slots=True)
class EvaluationReport:
    dataset_name: str
    top_k: int
    modes: tuple[str, ...]
    summary: dict[str, dict[str, float]]
    cases: list[EvaluationCaseResult]
    config: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "dataset_name": self.dataset_name,
            "top_k": self.top_k,
            "modes": list(self.modes),
            "summary": {
                mode: dict(metrics) for mode, metrics in self.summary.items()
            },
            "config": dict(self.config),
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


__all__ = ["EvaluationReport"]
