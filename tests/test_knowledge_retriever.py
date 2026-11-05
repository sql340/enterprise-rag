from __future__ import annotations

import unittest

from data_processing.knowledge_indexing import BaseEmbedding
from retrieval import KnowledgeRetriever
from retrieval.vector_store import VectorSearchResult


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    supports_text_search = True

    def __init__(self) -> None:
        self.search_limits: list[int] = []

    def search(self, query_vector, *, limit=5, score_threshold=None, filters=None):
        self.search_limits.append(limit)
        return [
            VectorSearchResult("a", 0.9, {"content": "a"}),
            VectorSearchResult("b", 0.8, {"content": "b"}),
        ][:limit]

    def text_search(self, query_text, *, limit=5, filters=None):
        return [
            VectorSearchResult("b", 7.0, {"content": "b"}),
            VectorSearchResult("c", 6.0, {"content": "c"}),
        ]


class BrokenTextSearchStore(FakeVectorStore):
    def text_search(self, query_text, *, limit=5, filters=None):
        raise RuntimeError("text index unavailable")


class KnowledgeRetrieverTest(unittest.TestCase):
    def test_hybrid_mode_fuses_dense_and_keyword_routes(self) -> None:
        store = FakeVectorStore()
        retriever = KnowledgeRetriever(
            FakeEmbedding(),
            store,
            retrieval_mode="hybrid",
            candidate_multiplier=3,
        )

        results = retriever.retrieve("query", top_k=2)

        self.assertEqual(store.search_limits, [6])
        self.assertEqual([item.id for item in results], ["b", "a"])
        self.assertTrue(all(item.retrieval_mode == "hybrid_rrf" for item in results))

    def test_dense_mode_keeps_single_route(self) -> None:
        retriever = KnowledgeRetriever(
            FakeEmbedding(),
            FakeVectorStore(),
            retrieval_mode="dense",
        )

        results = retriever.retrieve("query", top_k=1)

        self.assertEqual([item.id for item in results], ["a"])
        self.assertEqual(results[0].retrieval_mode, "dense")

    def test_hybrid_falls_back_to_dense_when_text_route_fails(self) -> None:
        retriever = KnowledgeRetriever(
            FakeEmbedding(),
            BrokenTextSearchStore(),
            retrieval_mode="hybrid",
        )

        results = retriever.retrieve("query", top_k=1)

        self.assertEqual([item.id for item in results], ["a"])
        self.assertEqual(results[0].retrieval_mode, "dense")


if __name__ == "__main__":
    unittest.main()
