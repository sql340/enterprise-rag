from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from access_control import AccessControlError, Principal, QdrantAclFilter
from data_processing.knowledge_indexing import BaseEmbedding
from graph_rag import GraphRetriever, GraphSearchHit
from observability import RagMetrics

from .RrfFusion import RrfFusion
from .rerank import BaseReranker
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
        graph_retriever: GraphRetriever | None = None,
        graph_weight: float = 0.8,
        acl_enabled: bool = False,
        reranker: BaseReranker | None = None,
        rerank_candidates: int = 0,
        expand_parent_context: bool = False,
        metrics: RagMetrics | None = None,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier 必须为正整数")
        if max_candidates <= 0:
            raise ValueError("max_candidates 必须为正整数")
        if dense_weight < 0 or keyword_weight < 0 or graph_weight < 0:
            raise ValueError("检索路权重不能为负数")
        if rerank_candidates < 0:
            raise ValueError("rerank_candidates 不能为负数")

        self.embedding = embedding
        self.vector_store = vector_store
        self.retrieval_mode = _normalise_mode(retrieval_mode)
        self.candidate_multiplier = candidate_multiplier
        self.max_candidates = max_candidates
        self.rrf_fusion = RrfFusion(k=rrf_k)
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self.graph_retriever = graph_retriever
        self.graph_weight = graph_weight
        self.acl_enabled = acl_enabled
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.expand_parent_context = expand_parent_context
        self.metrics = metrics

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
        started_at = perf_counter()
        outcome = "success"
        result_count = 0
        try:
            effective_filters = self._effective_filters(filters, principal)
            if mode == "graph":
                graph_limit = self._candidate_limit(top_k)
                graph_started_at = perf_counter()
                graph_results = self._graph_search(
                    question,
                    filters=effective_filters,
                    principal=principal,
                    limit=graph_limit,
                )
                self._observe_route(
                    "graph",
                    "success" if graph_results else "empty",
                    graph_started_at,
                )
                results = self._finalize(
                    question,
                    graph_results,
                    top_k=top_k,
                    filters=effective_filters,
                )
            else:
                query_vector = self._embed_query(question.strip())
                if mode == "dense":
                    dense_results = self._dense_search(
                        query_vector,
                        limit=self._candidate_limit(top_k),
                        score_threshold=score_threshold,
                        filters=effective_filters,
                        retrieval_mode="dense",
                    )
                    results = self._finalize(
                        question,
                        dense_results,
                        top_k=top_k,
                        filters=effective_filters,
                    )
                else:
                    candidate_limit = min(
                        self.max_candidates,
                        max(top_k, top_k * self.candidate_multiplier),
                    )
                    if self.reranker is not None:
                        candidate_limit = min(
                            self.max_candidates,
                            max(candidate_limit, self._candidate_limit(top_k)),
                        )
                    dense_results = self._dense_search(
                        query_vector,
                        limit=candidate_limit,
                        score_threshold=score_threshold,
                        filters=effective_filters,
                        retrieval_mode="dense",
                    )
                    routes = {"dense": dense_results}

                    if self.vector_store.supports_text_search:
                        keyword_started_at = perf_counter()
                        try:
                            routes["keyword"] = self.vector_store.text_search(
                                question,
                                limit=candidate_limit,
                                filters=effective_filters,
                            )
                        except (NotImplementedError, RuntimeError) as exc:
                            self._observe_route(
                                "keyword", "fallback", keyword_started_at
                            )
                            self._observe_route_error("keyword", exc)
                            logger.warning(
                                "BM25 文本检索不可用，继续使用其它检索路: %s", exc
                            )
                        else:
                            self._observe_route(
                                "keyword", "success", keyword_started_at
                            )

                    if mode == "hybrid_graph":
                        graph_started_at = perf_counter()
                        graph_results = self._graph_search(
                            question,
                            filters=effective_filters,
                            principal=principal,
                            limit=candidate_limit,
                        )
                        self._observe_route(
                            "graph",
                            "success" if graph_results else "empty",
                            graph_started_at,
                        )
                        if graph_results:
                            routes["graph"] = graph_results

                    if len(routes) == 1:
                        fused = dense_results
                    else:
                        fusion_started_at = perf_counter()
                        fused = self.rrf_fusion.fuse(
                            routes,
                            limit=(
                                candidate_limit
                                if self.reranker is not None
                                else top_k
                            ),
                            weights={
                                "dense": self.dense_weight,
                                "keyword": self.keyword_weight,
                                "graph": self.graph_weight,
                            },
                        )
                        self._observe_route("fusion", "success", fusion_started_at)
                    results = self._finalize(
                        question,
                        fused,
                        top_k=top_k,
                        filters=effective_filters,
                    )

            result_count = len(results)
            if not results:
                outcome = "empty"
            return results
        except Exception:
            outcome = "error"
            raise
        finally:
            if self.metrics is not None:
                self.metrics.observe_retrieval(
                    mode=mode,
                    outcome=outcome,
                    duration_seconds=perf_counter() - started_at,
                    result_count=result_count,
                )

    def _embed_query(self, question: str) -> list[float]:
        started_at = perf_counter()
        try:
            vector = self.embedding.embed_query(question)
        except Exception as exc:
            self._observe_embedding("error", started_at)
            self._observe_route("embedding", "error", started_at)
            self._observe_route_error("embedding", exc)
            raise
        self._observe_embedding("success", started_at)
        self._observe_route("embedding", "success", started_at)
        return vector

    def _observe_embedding(self, outcome: str, started_at: float) -> None:
        if self.metrics is None:
            return
        self.metrics.observe_embedding(
            model=str(getattr(self.embedding, "model_name", "unknown")),
            operation="query",
            outcome=outcome,
            duration_seconds=perf_counter() - started_at,
            text_count=1,
        )

    def _candidate_limit(self, top_k: int) -> int:
        if self.reranker is None:
            return top_k
        requested = self.rerank_candidates or top_k * self.candidate_multiplier
        return min(self.max_candidates, max(top_k, requested))

    def _finalize(
        self,
        question: str,
        results: Sequence[VectorSearchResult],
        *,
        top_k: int,
        filters: Mapping[str, Any] | Any | None,
    ) -> list[VectorSearchResult]:
        values = list(results)
        if self.reranker is not None:
            rerank_started_at = perf_counter()
            try:
                values = self.reranker.rerank(question, values, top_k=top_k)
            except Exception as exc:
                self._observe_route("rerank", "fallback", rerank_started_at)
                self._observe_route_error("rerank", exc)
                logger.warning("重排失败，退回融合排序: %s", exc)
                values = values[:top_k]
            else:
                self._observe_route("rerank", "success", rerank_started_at)
        else:
            values = values[:top_k]
        if self.expand_parent_context:
            return self._expand_parent_context(values, filters=filters)
        return values

    def _expand_parent_context(
        self,
        results: Sequence[VectorSearchResult],
        *,
        filters: Mapping[str, Any] | Any | None,
    ) -> list[VectorSearchResult]:
        values = list(results)
        parent_ids = list(
            dict.fromkeys(
                str(item.payload.get("parent_id"))
                for item in values
                if item.payload.get("parent_id")
                and str(item.payload.get("parent_id")) != item.chunk_id
            )
        )
        if not parent_ids:
            return values
        getter = getattr(self.vector_store, "get_chunks_by_ids", None)
        if getter is None:
            return values
        parent_started_at = perf_counter()
        try:
            parents = getter(parent_ids, filters=filters)
        except Exception as exc:
            self._observe_route("parent_expand", "fallback", parent_started_at)
            self._observe_route_error("parent_expand", exc)
            logger.warning("父块上下文扩展失败，保留原始知识块: %s", exc)
            return values
        self._observe_route("parent_expand", "success", parent_started_at)
        parents_by_id = {item.chunk_id: item for item in parents}
        for item in values:
            parent_id = str(item.payload.get("parent_id") or "")
            parent = parents_by_id.get(parent_id)
            if parent is None:
                continue
            item.payload["context_content"] = parent.content
            item.payload["parent_chunk_id"] = parent.chunk_id
            item.payload["parent_expanded"] = True
        return values

    def _dense_search(
        self,
        query_vector: list[float],
        *,
        limit: int,
        score_threshold: float | None,
        filters: Mapping[str, Any] | Any | None,
        retrieval_mode: str,
    ) -> list[VectorSearchResult]:
        started_at = perf_counter()
        try:
            results = self.vector_store.search(
                query_vector,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters,
            )
        except Exception as exc:
            self._observe_route("dense", "error", started_at)
            self._observe_route_error("dense", exc)
            raise
        self._observe_route("dense", "success", started_at)
        for rank, result in enumerate(results, start=1):
            result.retrieval_mode = retrieval_mode
            result.route_scores.setdefault("dense", result.score)
            result.route_ranks.setdefault("dense", rank)
        return results

    def _graph_search(
        self,
        question: str,
        *,
        filters: Mapping[str, Any] | Any | None,
        principal: Principal | None,
        limit: int,
    ) -> list[VectorSearchResult]:
        if self.graph_retriever is None or principal is None:
            return []

        ceiling = max(limit, self.max_candidates)
        fetch_limit = min(
            ceiling,
            max(limit, limit * self.candidate_multiplier),
        )
        while True:
            try:
                hits = self.graph_retriever.retrieve(
                    question,
                    tenant_id=principal.tenant_id,
                    limit=fetch_limit,
                )
            except Exception as exc:
                self._observe_route_error("graph", exc)
                logger.warning("图谱检索不可用，跳过 graph 路: %s", exc)
                return []
            if not hits:
                return []

            results = self._hydrate_graph_hits(hits, filters=filters)
            if (
                len(results) >= limit
                or fetch_limit >= ceiling
                or len(hits) < fetch_limit
            ):
                return results[:limit]

            next_limit = min(
                ceiling,
                max(
                    fetch_limit + limit,
                    fetch_limit * self.candidate_multiplier,
                ),
            )
            if next_limit <= fetch_limit:
                return results[:limit]
            fetch_limit = next_limit

    def _hydrate_graph_hits(
        self,
        hits: Sequence[GraphSearchHit],
        *,
        filters: Mapping[str, Any] | Any | None,
    ) -> list[VectorSearchResult]:
        chunk_ids = list(dict.fromkeys(hit.chunk_id for hit in hits))
        try:
            chunks = self.vector_store.get_chunks_by_ids(
                chunk_ids,
                filters=filters,
            )
        except Exception as exc:
            self._observe_route_error("graph", exc)
            logger.warning("图谱候选正文读取失败，跳过 graph 路: %s", exc)
            return []
        chunks_by_id = {result.chunk_id: result for result in chunks}

        results: list[VectorSearchResult] = []
        for hit in hits:
            result = chunks_by_id.get(hit.chunk_id)
            if result is None:
                continue
            result.score = hit.score
            result.retrieval_mode = "graph"
            result.route_scores["graph"] = hit.score
            results.append(result)
        for rank, result in enumerate(results, start=1):
            result.route_ranks["graph"] = rank
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

    def _observe_route(self, route: str, outcome: str, started_at: float) -> None:
        if self.metrics is not None:
            self.metrics.observe_retrieval_route(
                route=route,
                outcome=outcome,
                duration_seconds=perf_counter() - started_at,
            )

    def _observe_route_error(self, route: str, exc: Exception) -> None:
        if self.metrics is not None:
            self.metrics.observe_retrieval_route_error(
                route=route,
                error_type=type(exc).__name__,
            )

    def close(self) -> None:
        try:
            self.vector_store.close()
        finally:
            if self.graph_retriever is not None:
                self.graph_retriever.close()


def _normalise_mode(mode: str) -> str:
    value = str(mode).strip().lower().replace("-", "_")
    if value in {"dense", "vector"}:
        return "dense"
    if value in {"hybrid", "hybrid_rrf", "multi_route"}:
        return "hybrid"
    if value in {"graph", "graph_only"}:
        return "graph"
    if value in {"hybrid_graph", "graph_hybrid"}:
        return "hybrid_graph"
    raise ValueError("retrieval_mode 必须是 dense、hybrid、graph 或 hybrid_graph")


__all__ = ["KnowledgeRetriever"]
