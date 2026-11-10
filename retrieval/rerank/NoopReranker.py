from __future__ import annotations

from collections.abc import Sequence

from retrieval.vector_store import VectorSearchResult

from .BaseReranker import BaseReranker


class NoopReranker(BaseReranker):
    """不改变顺序的重排器，用于测试或显式关闭重排。"""

    def rerank(
        self,
        question: str,
        results: Sequence[VectorSearchResult],
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        return list(results)[:top_k]


__all__ = ["NoopReranker"]
