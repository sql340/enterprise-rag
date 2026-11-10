from __future__ import annotations

import unittest

from data_processing.knowledge_indexing import BaseEmbedding
from retrieval import KnowledgeRetriever
from retrieval.rerank import BaseReranker
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


class FakeStore:
    supports_text_search = False

    def __init__(self, results, parents=None) -> None:
        self.results = results
        self.parents = parents or []
        self.search_limits: list[int] = []
        self.parent_requests = []

    def search(self, query_vector, *, limit=5, score_threshold=None, filters=None):
        self.search_limits.append(limit)
        return list(self.results)[:limit]

    def get_chunks_by_ids(self, chunk_ids, *, filters=None):
        self.parent_requests.append((list(chunk_ids), filters))
        return list(self.parents)


class ReverseReranker(BaseReranker):
    def __init__(self) -> None:
        self.candidates = 0

    def rerank(self, question, results, *, top_k):
        self.candidates = len(results)
        values = list(results)
        for index, item in enumerate(values):
            item.score = float(index)
            item.retrieval_mode = f"{item.retrieval_mode}+rerank"
        return list(reversed(values))[:top_k]


class RerankAndContextTest(unittest.TestCase):
    def test_reranker_receives_candidates_and_controls_top_k(self) -> None:
        store = FakeStore(
            [
                VectorSearchResult(f"c{index}", 1.0, {"content": f"text {index}"})
                for index in range(5)
            ]
        )
        reranker = ReverseReranker()
        retriever = KnowledgeRetriever(
            FakeEmbedding(),
            store,
            retrieval_mode="dense",
            reranker=reranker,
            rerank_candidates=3,
        )

        results = retriever.retrieve("query", top_k=2)

        self.assertEqual(store.search_limits, [3])
        self.assertEqual(reranker.candidates, 3)
        self.assertEqual([item.id for item in results], ["c2", "c1"])
        self.assertTrue(all("rerank" in item.retrieval_mode for item in results))

    def test_parent_context_is_expanded_after_retrieval(self) -> None:
        child = VectorSearchResult(
            "child",
            0.9,
            {
                "chunk_id": "child",
                "parent_id": "parent",
                "content": "短子块",
            },
        )
        store = FakeStore(
            [child],
            parents=[
                VectorSearchResult(
                    "parent",
                    0.0,
                    {"chunk_id": "parent", "content": "完整父块上下文"},
                )
            ],
        )
        retriever = KnowledgeRetriever(
            FakeEmbedding(),
            store,
            retrieval_mode="dense",
            expand_parent_context=True,
        )

        results = retriever.retrieve("query", top_k=1)

        self.assertEqual(store.parent_requests[0][0], ["parent"])
        self.assertEqual(results[0].payload["context_content"], "完整父块上下文")
        self.assertEqual(results[0].context_content, "完整父块上下文")


if __name__ == "__main__":
    unittest.main()
