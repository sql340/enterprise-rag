from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retrieval.vector_store import VectorSearchResult

from .BaseReranker import BaseReranker
from .RerankerError import RerankerError


class CrossEncoderReranker(BaseReranker):
    """使用 sentence-transformers CrossEncoder 做查询-知识块相关性重排。"""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_candidates: int = 100,
    ) -> None:
        if not str(model_name).strip():
            raise ValueError("rerank model_name 不能为空")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if max_candidates <= 0:
            raise ValueError("max_candidates 必须为正整数")
        self.model_name = str(model_name).strip()
        self.device = device
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self._model: Any = None

    def rerank(
        self,
        question: str,
        results: Sequence[VectorSearchResult],
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        candidates = list(results)[: self.max_candidates]
        if not candidates:
            return []
        pairs = [(question, item.content) for item in candidates]
        try:
            scores = self._get_model().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankerError(f"CrossEncoder 重排失败: {exc}") from exc
        if len(scores) != len(candidates):
            raise RerankerError("CrossEncoder 返回的分数数量与候选数量不一致")

        for result, score in zip(candidates, scores, strict=False):
            result.score = float(score)
            result.route_scores["rerank"] = float(score)
            if "rerank" not in result.retrieval_mode:
                result.retrieval_mode = f"{result.retrieval_mode}+rerank"
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RerankerError(
                "重排需要安装 sentence-transformers"
            ) from exc
        kwargs = {}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model


__all__ = ["CrossEncoderReranker"]
