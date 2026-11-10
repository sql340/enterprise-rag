from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """一条固定评测问题及其期望来源。"""

    case_id: str
    question: str
    expected_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    expected_document_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        question = str(self.question).strip()
        if not case_id:
            raise ValueError("case_id 不能为空")
        if not question:
            raise ValueError("question 不能为空")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "question", question)
        object.__setattr__(
            self,
            "expected_chunk_ids",
            _normalise_ids(self.expected_chunk_ids),
        )
        object.__setattr__(
            self,
            "expected_document_ids",
            _normalise_ids(self.expected_document_ids),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.expected_chunk_ids and not self.expected_document_ids:
            raise ValueError("评测用例至少需要一个期望 chunk_id 或 document_id")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationCase":
        return cls(
            case_id=str(payload.get("case_id") or payload.get("id") or ""),
            question=str(payload.get("question") or ""),
            expected_chunk_ids=tuple(payload.get("expected_chunk_ids") or ()),
            expected_document_ids=tuple(payload.get("expected_document_ids") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "expected_document_ids": list(self.expected_document_ids),
            "metadata": dict(self.metadata),
        }


def _normalise_ids(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    return tuple(
        dict.fromkeys(str(item).strip() for item in values or () if str(item).strip())
    )


__all__ = ["EvaluationCase"]
