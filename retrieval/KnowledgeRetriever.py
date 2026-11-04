from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from data_processing.knowledge_indexing import BaseEmbedding

from .vector_store import BaseVectorStore, VectorSearchResult


class KnowledgeRetriever:
    """把查询文本向量化并召回相关知识块。"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        query_vector = self.embedding.embed_query(question.strip())
        return self.vector_store.search(
            query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )


__all__ = ["KnowledgeRetriever"]
