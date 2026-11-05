from __future__ import annotations

import unittest

from qdrant_client import QdrantClient, models

from data_processing.chunker import Chunk
from retrieval.vector_store import QdrantVectorStore


def _chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        content=content,
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        char_start=0,
        char_end=len(content),
        strategy="recursive",
        metadata={"source": "/knowledge/en.txt", "file_type": "text"},
        content_hash=f"hash-{chunk_id}",
    )


class QdrantTextSearchTest(unittest.TestCase):
    def test_text_search_returns_bm25_hits(self) -> None:
        store = QdrantVectorStore(
            "text_search_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
        )
        store.ensure_collection(2)
        store.client.create_payload_index(
            collection_name=store.collection_name,
            field_name="content",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                scoring=True,
            ),
        )
        chunks = [
            _chunk("a", "reimbursement invoice approval"),
            _chunk("b", "travel lodging standard"),
        ]
        store.upsert_chunks(
            chunks,
            [[1.0, 0.0], [0.0, 1.0]],
            embedding_model="fake",
        )

        results = store.text_search("reimbursement invoice", limit=5)

        self.assertEqual([item.chunk_id for item in results], ["a"])
        self.assertGreater(results[0].score, 0)
        self.assertEqual(results[0].retrieval_mode, "keyword")


if __name__ == "__main__":
    unittest.main()
