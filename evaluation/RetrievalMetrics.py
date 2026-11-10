from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .EvaluationCase import EvaluationCase


class RetrievalMetrics:
    """基于期望 chunk/document 的检索指标。"""

    @classmethod
    def calculate(
        cls,
        case: EvaluationCase,
        results: Sequence[Any],
        *,
        k: int,
    ) -> dict[str, float]:
        if k <= 0:
            raise ValueError("k 必须为正整数")
        top = list(results)[:k]
        relevant = cls._relevance(case, top)
        expected_count = len(case.expected_chunk_ids) + len(
            case.expected_document_ids
        )
        found = len(cls._found_keys(case, top))
        recall = min(1.0, found / expected_count) if expected_count else 0.0
        hit_rate = 1.0 if any(relevant) else 0.0
        mrr = 0.0
        for rank, matched in enumerate(relevant, start=1):
            if matched:
                mrr = 1.0 / rank
                break
        dcg = sum(
            (1.0 if matched else 0.0) / math.log2(rank + 1)
            for rank, matched in enumerate(relevant, start=1)
        )
        ideal_hits = min(expected_count, k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg else 0.0
        return {
            "recall_at_k": recall,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
            "hit_rate": hit_rate,
        }

    @staticmethod
    def _relevance(case: EvaluationCase, results: Sequence[Any]) -> list[bool]:
        expected_chunks = set(case.expected_chunk_ids)
        expected_documents = set(case.expected_document_ids)
        values: list[bool] = []
        for result in results:
            chunk_id = result.chunk_id if hasattr(result, "chunk_id") else None
            document_id = (
                result.document_id if hasattr(result, "document_id") else None
            )
            values.append(
                (chunk_id is not None and str(chunk_id) in expected_chunks)
                or (
                    document_id is not None
                    and str(document_id) in expected_documents
                )
            )
        return values

    @staticmethod
    def _found_keys(case: EvaluationCase, results: Sequence[Any]) -> set[tuple[str, str]]:
        expected_chunks = set(case.expected_chunk_ids)
        expected_documents = set(case.expected_document_ids)
        found: set[tuple[str, str]] = set()
        for result in results:
            chunk_id = str(getattr(result, "chunk_id", ""))
            document_id = str(getattr(result, "document_id", ""))
            if chunk_id in expected_chunks:
                found.add(("chunk", chunk_id))
            if document_id in expected_documents:
                found.add(("document", document_id))
        return found


__all__ = ["RetrievalMetrics"]
