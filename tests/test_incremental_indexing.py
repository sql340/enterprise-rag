from __future__ import annotations

import unittest

from qdrant_client import QdrantClient

from data_processing.chunker import Chunker
from data_processing.data_reader import Document
from data_processing.knowledge_indexing import BaseEmbedding, KnowledgeIndexer
from retrieval.vector_store import QdrantVectorStore


class CountingEmbedding(BaseEmbedding):
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []
        self._model_name = "fake-embedding-v1"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return 2

    def set_model_name(self, value: str) -> None:
        self._model_name = value

    def _embed_texts(self, texts, *, query):
        self.embedded_texts.extend(texts)
        return [[1.0, float(len(text))] for text in texts]


class IncrementalIndexingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantVectorStore(
            "incremental_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
        )
        self.embedding = CountingEmbedding()
        self.indexer = KnowledgeIndexer(self.embedding, self.store)

    def _chunks(self, text: str, *, chunk_size: int = 80):
        chunker = Chunker(
            "recursive",
            chunk_size=chunk_size,
            chunk_overlap=10,
        )
        return chunker.split(
            Document(
                content=text,
                metadata={"source": "/knowledge/policy.md", "file_type": "markdown"},
            )
        )

    def test_unchanged_chunks_are_not_embedded_twice(self) -> None:
        chunks = self._chunks("第一段知识。\n\n第二段知识。\n\n第三段知识。")

        first = self.indexer.index_chunks(chunks)
        self.assertEqual(first.embedded_chunks, len(chunks))
        self.assertEqual(first.skipped_chunks, 0)
        self.assertEqual(self.store.count(), len(chunks))

        self.embedding.embedded_texts.clear()
        second = self.indexer.index_chunks(chunks)
        self.assertEqual(second.embedded_chunks, 0)
        self.assertEqual(second.skipped_chunks, len(chunks))
        self.assertEqual(self.embedding.embedded_texts, [])
        self.assertEqual(self.store.count(), len(chunks))

    def test_config_change_rebuilds_and_removes_stale_chunks(self) -> None:
        original = self._chunks("第一段知识。\n\n第二段知识。\n\n第三段知识。")
        self.indexer.index_chunks(original)

        rebuilt = self._chunks(
            "第一段知识。\n\n第二段知识。\n\n第三段知识。",
            chunk_size=20,
        )
        result = self.indexer.index_chunks(rebuilt)

        self.assertEqual(result.embedded_chunks, len(rebuilt))
        self.assertGreater(result.stale_deleted, 0)
        self.assertEqual(self.store.count(), len(rebuilt))

    def test_force_reindex_embeds_unchanged_chunks(self) -> None:
        chunks = self._chunks("第一段知识。\n\n第二段知识。")
        self.indexer.index_chunks(chunks)

        self.embedding.embedded_texts.clear()
        result = self.indexer.index_chunks(chunks, force_reindex=True)

        self.assertEqual(result.embedded_chunks, len(chunks))
        self.assertEqual(len(self.embedding.embedded_texts), len(chunks))

    def test_embedding_model_change_reembeds_chunks(self) -> None:
        chunks = self._chunks("第一段知识。\n\n第二段知识。")
        self.indexer.index_chunks(chunks)

        self.embedding.set_model_name("fake-embedding-v2")
        self.embedding.embedded_texts.clear()
        result = self.indexer.index_chunks(chunks)

        self.assertEqual(result.embedded_chunks, len(chunks))
        self.assertEqual(len(self.embedding.embedded_texts), len(chunks))


if __name__ == "__main__":
    unittest.main()
