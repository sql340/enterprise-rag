from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from access_control import AccessControlError, Principal, QdrantAclFilter
from data_processing.knowledge_indexing import BaseEmbedding

from .RrfFusion import RrfFusion
from .vector_store import BaseVectorStore, VectorSearchResult


logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """把查询文本向量化，并支持 dense / hybrid 多路召回。"""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        *,
        retrieval_mode: str = "hybrid",
        candidate_multiplier: int = 4,
        max_candidates: int = 100,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        keyword_weight: float = 0.8,
        acl_enabled: bool = False,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier 必须为正整数")
        if max_candidates <= 0:
            raise ValueError("max_candidates 必须为正整数")
        if dense_weight < 0 or keyword_weight < 0:
            raise ValueError("检索路权重不能为负数")

        self.embedding = embedding
        self.vector_store = vector_store
        self.retrieval_mode = _normalise_mode(retrieval_mode)
        self.candidate_multiplier = candidate_multiplier
        self.max_candidates = max_candidates
        self.rrf_fusion = RrfFusion(k=rrf_k)
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self.acl_enabled = acl_enabled

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, Any] | Any | None = None,
        retrieval_mode: str | None = None,
        principal: Principal | None = None,
    ) -> list[VectorSearchResult]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        mode = (
            self.retrieval_mode
            if retrieval_mode is None
            else _normalise_mode(retrieval_mode)
        )
        effective_filters = self._effective_filters(filters, principal)
        query_vector = self.embedding.embed_query(question.strip())
        if mode == "dense":
            return self._dense_search(
                query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                filters=effective_filters,
                retrieval_mode="dense",
            )

        candidate_limit = min(
            self.max_candidates,
            max(top_k, top_k * self.candidate_multiplier),
        )
        dense_results = self._dense_search(
            query_vector,
            limit=candidate_limit,
            score_threshold=score_threshold,
            filters=effective_filters,
            retrieval_mode="dense",
        )
        if not self.vector_store.supports_text_search:
            return dense_results[:top_k]

        try:
            keyword_results = self.vector_store.text_search(
                question,
                limit=candidate_limit,
                filters=effective_filters,
            )
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("BM25 文本检索不可用，回退到 dense 检索: %s", exc)
            return dense_results[:top_k]

        return self.rrf_fusion.fuse(
            {"dense": dense_results, "keyword": keyword_results},
            limit=top_k,
            weights={"dense": self.dense_weight, "keyword": self.keyword_weight},
        )

    def _dense_search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        score_threshold: float | None,
        filters: Mapping[str, Any] | Any | None,
        retrieval_mode: str,
    ) -> list[VectorSearchResult]:
        results = self.vector_store.search(
            query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
        )
        for rank, result in enumerate(results, start=1):
            result.retrieval_mode = retrieval_mode
            result.route_scores.setdefault("dense", result.score)
            result.route_ranks.setdefault("dense", rank)
        return results

    def _effective_filters(
        self,
        filters: Mapping[str, Any] | Any | None,
        principal: Principal | None,
    ) -> Any:
        access_filter = None
        if self.acl_enabled:
            if principal is None:
                raise AccessControlError("ACL 已启用，但请求缺少认证主体")
            access_filter = QdrantAclFilter(principal).build()
        values = [value for value in (access_filter, filters) if value is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        merge_filters = getattr(self.vector_store, "merge_filters", None)
        if merge_filters is None:
            return tuple(values)
        return merge_filters(*values)


def _normalise_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    if value in {"dense", "vector"}:
        return "dense"
    if value in {"hybrid", "hybrid_rrf", "multi_route"}:
        return "hybrid"
    raise ValueError("retrieval_mode 必须是 dense 或 hybrid")


__all__ = ["KnowledgeRetriever"]
