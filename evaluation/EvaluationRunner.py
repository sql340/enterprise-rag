from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean
from time import perf_counter
from typing import Any

from access_control import Principal

from .EvaluationCase import EvaluationCase
from .EvaluationCaseResult import EvaluationCaseResult
from .EvaluationDataset import EvaluationDataset
from .EvaluationReport import EvaluationReport
from .RetrievalMetrics import RetrievalMetrics


class EvaluationRunner:
    """对固定评测集执行 dense/hybrid/graph/hybrid_graph 对比。"""

    def __init__(
        self,
        retriever: Any,
        *,
        principal: Principal | None = None,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.principal = principal
        self.filters = dict(filters) if filters else None
        self.score_threshold = score_threshold

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        modes: Sequence[str] = ("dense", "hybrid", "hybrid_graph"),
        top_k: int = 5,
    ) -> EvaluationReport:
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        normalised_modes = tuple(dict.fromkeys(str(mode) for mode in modes))
        if not normalised_modes:
            raise ValueError("modes 不能为空")
        results: list[EvaluationCaseResult] = []
        for mode in normalised_modes:
            for case in dataset.cases:
                results.append(self._run_case(case, mode=mode, top_k=top_k))
        return EvaluationReport(
            dataset_name=dataset.name,
            top_k=top_k,
            modes=normalised_modes,
            summary={mode: self._summarise(results, mode) for mode in normalised_modes},
            cases=results,
            config={
                "dataset_metadata": dict(dataset.metadata),
                "filters": dict(self.filters) if self.filters else None,
                "score_threshold": self.score_threshold,
            },
        )

    def _run_case(
        self,
        case: EvaluationCase,
        *,
        mode: str,
        top_k: int,
    ) -> EvaluationCaseResult:
        started = perf_counter()
        try:
            retrieved = self.retriever.retrieve(
                case.question,
                top_k=top_k,
                score_threshold=self.score_threshold,
                filters=self.filters,
                retrieval_mode=mode,
                principal=self.principal,
            )
        except Exception as exc:
            return EvaluationCaseResult(
                case_id=case.case_id,
                mode=mode,
                question=case.question,
                metrics={
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "ndcg_at_k": 0.0,
                    "hit_rate": 0.0,
                },
                latency_ms=round((perf_counter() - started) * 1000, 2),
                error=str(exc),
            )
        latency_ms = round((perf_counter() - started) * 1000, 2)
        metrics = RetrievalMetrics.calculate(case, retrieved, k=top_k)
        return EvaluationCaseResult(
            case_id=case.case_id,
            mode=mode,
            question=case.question,
            metrics=metrics,
            retrieved=[
                {
                    "rank": rank,
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "score": round(float(item.score), 6),
                    "retrieval_mode": item.retrieval_mode,
                }
                for rank, item in enumerate(retrieved[:top_k], start=1)
            ],
            latency_ms=latency_ms,
        )

    @staticmethod
    def _summarise(
        results: Sequence[EvaluationCaseResult],
        mode: str,
    ) -> dict[str, float]:
        values = [item for item in results if item.mode == mode]
        if not values:
            return {}
        return {
            "cases": float(len(values)),
            "failures": float(sum(1 for item in values if item.error)),
            "recall_at_k": _mean(item.metrics.get("recall_at_k", 0.0) for item in values),
            "mrr": _mean(item.metrics.get("mrr", 0.0) for item in values),
            "ndcg_at_k": _mean(item.metrics.get("ndcg_at_k", 0.0) for item in values),
            "hit_rate": _mean(item.metrics.get("hit_rate", 0.0) for item in values),
            "avg_latency_ms": _mean(item.latency_ms for item in values),
        }


def _mean(values: Any) -> float:
    numbers = list(values)
    return round(mean(numbers), 6) if numbers else 0.0


__all__ = ["EvaluationRunner"]
