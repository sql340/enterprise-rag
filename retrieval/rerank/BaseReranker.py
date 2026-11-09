from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from retrieval.vector_store import VectorSearchResult


class BaseReranker(ABC):
    """对融合后的候选知识块进行二次排序。"""

    @abstractmethod
    def rerank(
        self,
        question: str,
        results: Sequence[VectorSearchResult],
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        """返回按相关性排序后的前 ``top_k`` 个结果。"""


__all__ = ["BaseReranker"]
