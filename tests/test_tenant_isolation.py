from __future__ import annotations

import unittest

from qdrant_client import QdrantClient

from access_control import DocumentAcl
from data_processing.chunker import Chunk, TenantChunkScope
from data_processing.knowledge_indexing import BaseEmbedding, KnowledgeIndexer
from retrieval.vector_store import QdrantVectorStore


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-tenant-embedding"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


def chunk(tenant_id: str = "tenant-a") -> Chunk:
    return Chunk(
        content="同一个文件内容",
        chunk_id="same-chunk-id",
        document_id="same-document-id",
        chunk_index=0,
        char_start=0,
        char_end=8,
        strategy="recursive",
        content_hash="same-content-hash",
        metadata={
            "source": "upload://same.md",
            "tenant_id": tenant_id,
            "file_type": "markdown",
        },
    )


class TenantChunkScopeTest(unittest.TestCase):
    def test_same_chunk_in_two_tenants_gets_different_ids(self) -> None:
        first = TenantChunkScope.apply([chunk("tenant-a")], "tenant-a")[0]
        second = TenantChunkScope.apply([chunk("tenant-b")], "tenant-b")[0]

        self.assertNotEqual(first.chunk_id, second.chunk_id)
        self.assertNotEqual(first.document_id, second.document_id)
        self.assertEqual(first.metadata["tenant_id"], "tenant-a")
        self.assertTrue(first.metadata["tenant_scoped"])

    def test_existing_scope_cannot_be_reused_for_another_tenant(self) -> None:
        scoped = TenantChunkScope.apply([chunk("tenant-a")], "tenant-a")
        with self.assertRaises(ValueError):
            TenantChunkScope.apply(scoped, "tenant-b")


class TenantVectorIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantVectorStore(
            "tenant_isolation_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
        )
        self.indexer = KnowledgeIndexer(
            FakeEmbedding(),
            self.store,
            sync_scope="source",
        )

    def tearDown(self) -> None:
        self.store.close()

    def test_index_requires_explicit_or_acl_tenant(self) -> None:
        raw = chunk("")
        raw.metadata.pop("tenant_id")
        with self.assertRaises(ValueError):
            self.indexer.index_chunks([raw])

    def test_acl_tenant_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.indexer.index_chunks(
                [chunk("tenant-b")],
                tenant_id="tenant-a",
                acl=DocumentAcl(tenant_id="tenant-a"),
            )

    def test_two_tenants_with_identical_source_do_not_overwrite(self) -> None:
        self.indexer.index_chunks(
            [chunk("tenant-a")],
            tenant_id="tenant-a",
            acl=DocumentAcl(tenant_id="tenant-a", owner_id="u-a"),
        )
        self.indexer.index_chunks(
            [chunk("tenant-b")],
            tenant_id="tenant-b",
            acl=DocumentAcl(tenant_id="tenant-b", owner_id="u-b"),
        )

        records_a = self.store.list_chunk_records(
            tenant_id="tenant-a",
            scope_field="source",
            scope_value="upload://same.md",
        )
        records_b = self.store.list_chunk_records(
            tenant_id="tenant-b",
            scope_field="source",
            scope_value="upload://same.md",
        )
        self.assertEqual(len(records_a), 1)
        self.assertEqual(len(records_b), 1)
        self.assertNotEqual(next(iter(records_a)), next(iter(records_b)))

    def test_reindexing_one_tenant_does_not_prune_another(self) -> None:
        self.indexer.index_chunks(
            [chunk("tenant-a")],
            tenant_id="tenant-a",
            acl=DocumentAcl(tenant_id="tenant-a"),
        )
        self.indexer.index_chunks(
            [chunk("tenant-b")],
            tenant_id="tenant-b",
            acl=DocumentAcl(tenant_id="tenant-b"),
        )
        changed = chunk("tenant-a")
        changed.content = "更新后的内容"
        changed.content_hash = "changed-hash"
        self.indexer.index_chunks(
            [changed],
            tenant_id="tenant-a",
            acl=DocumentAcl(tenant_id="tenant-a"),
        )

        self.assertEqual(
            len(
                self.store.list_chunk_records(
                    tenant_id="tenant-a",
                    scope_field="source",
                    scope_value="upload://same.md",
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                self.store.list_chunk_records(
                    tenant_id="tenant-b",
                    scope_field="source",
                    scope_value="upload://same.md",
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
