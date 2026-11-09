from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .EvaluationCase import EvaluationCase


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """固定评测集，可从 JSON 或 JSONL 加载。"""

    name: str
    cases: tuple[EvaluationCase, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("dataset.name 不能为空")
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("dataset.cases 不能为空")
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset 中存在重复 case_id")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        default_name: str = "dataset",
    ) -> "EvaluationDataset":
        if isinstance(payload, list):
            return cls(
                name=default_name,
                cases=tuple(EvaluationCase.from_dict(item) for item in payload),
            )
        if not isinstance(payload, dict):
            raise ValueError("评测集必须是 JSON 对象或数组")
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError("评测集缺少 cases 数组")
        return cls(
            name=str(payload.get("name") or default_name),
            cases=tuple(EvaluationCase.from_dict(item) for item in cases),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, value: str, *, name: str = "dataset") -> "EvaluationDataset":
        return cls.from_dict(json.loads(value), default_name=name)

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationDataset":
        source = Path(path).expanduser()
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".jsonl":
            cases = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    cases.append(EvaluationCase.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"{source}:{line_number} 评测用例不合法: {exc}"
                    ) from exc
            return cls(name=source.stem, cases=tuple(cases))
        return cls.from_json(text, name=source.stem)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "cases": [case.to_dict() for case in self.cases],
        }


__all__ = ["EvaluationDataset"]
