from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient

from access_control import Classification, DocumentAcl
from data_processing.chunker import Chunk
from data_processing.knowledge_indexing import BaseEmbedding, KnowledgeIndexer
from graph_rag import (
    BaseGraphExtractor,
    GraphEntity,
    GraphExtraction,
    GraphIndexer,
    SqliteGraphStore,
)
from knowledge_management import (
    KnowledgeGovernanceStore,
    KnowledgeIngestionService,
)
from retrieval.vector_store import QdrantVectorStore


class FakeEmbedding(BaseEmbedding):
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-governance-embedding"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        if not query:
            self.embedded_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeGraphExtractor(BaseGraphExtractor):
    @property
    def extractor_version(self) -> str:
        return "fake-graph-v1"

    def extract(self, chunk: Chunk) -> GraphExtraction:
        return GraphExtraction(
            entities=(GraphEntity("公司", "organization"),),
            relations=(),
        )


class KnowledgeManagementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.vector_store = QdrantVectorStore(
            "knowledge_management_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
        )
        self.addCleanup(self.vector_store.close)
        self.graph_store = SqliteGraphStore(":memory:")
        self.addCleanup(self.graph_store.close)
        self.governance = KnowledgeGovernanceStore(":memory:")
        self.addCleanup(self.governance.close)
        self.embedding = FakeEmbedding()
        self.service = KnowledgeIngestionService(
            governance_store=self.governance,
            data_dir=Path(self.temp_dir.name) / "uploads",
            indexer=KnowledgeIndexer(self.embedding, self.vector_store),
            graph_indexer=GraphIndexer(self.graph_store, FakeGraphExtractor()),
        )
        self.acl = DocumentAcl(
            tenant_id="tenant-a",
            classification=Classification.INTERNAL,
            owner_id="admin-1",
        )
        self.content = (
            "# 差旅政策\n\n"
            "员工出差前需要在系统中提交申请，并附上预计交通与住宿费用。\n\n"
            "差旅结束后十个工作日内提交发票和审批单，财务审核后完成报销。\n"
        ).encode("utf-8")

    def _upload(self, tenant_id: str = "tenant-a") -> object:
        acl = DocumentAcl(
            tenant_id=tenant_id,
            classification=Classification.INTERNAL,
            owner_id="admin-1",
        )
        return self.service.upload_document(
            tenant_id=tenant_id,
            actor_id="admin-1",
            file_name="policy.md",
            content=self.content,
            acl=acl,
            chunk_strategy="recursive",
            chunk_options={"chunk_size": 80, "chunk_overlap": 10},
        )

    def test_upload_preview_build_and_audit(self) -> None:
        document = self._upload()
        self.assertTrue(Path(document.storage_path).is_file())

        preview = self.service.preview_document(
            tenant_id="tenant-a",
            document_id=document.document_id,
        )
        self.assertGreater(preview["chunk_count"], 0)
        self.assertIn("quality", preview)

        job = self.service.create_job(
            tenant_id="tenant-a",
            document_id=document.document_id,
            actor_id="admin-1",
        )
        completed = self.service.run_job(job.job_id)
        refreshed = self.service.get_document(
            tenant_id="tenant-a",
            document_id=document.document_id,
        )

        self.assertEqual(completed.status, "succeeded")
        self.assertFalse(completed.result["skipped"])
        self.assertIsNotNone(completed.result["graph"])
        self.assertEqual(refreshed.status, "succeeded")
        self.assertGreater(refreshed.chunk_count, 0)
        self.assertTrue(refreshed.build_fingerprint)
        self.assertTrue(self.embedding.embedded_texts)
        actions = {
            event.action
            for event in self.service.list_audit_events(tenant_id="tenant-a")
        }
        self.assertIn("document.upload", actions)
        self.assertIn("ingestion_job.succeed", actions)

    def test_upload_path_is_sanitised_for_hostile_tenant_id(self) -> None:
        tenant_id = "../../tenant a"
        document = self.service.upload_document(
            tenant_id=tenant_id,
            actor_id="admin-1",
            file_name="../../escape.md",
            content=self.content,
            acl=DocumentAcl(tenant_id=tenant_id, owner_id="admin-1"),
        )
        stored = Path(document.storage_path)
        self.assertTrue(stored.is_file())
        self.assertNotIn("..", stored.parts)
        self.assertEqual(document.file_name, "escape.md")

    def test_good_unchanged_document_is_skipped(self) -> None:
        document = self._upload()
        job = self.service.create_job(
            tenant_id="tenant-a",
            document_id=document.document_id,
            actor_id="admin-1",
        )
        self.service.run_job(job.job_id)
        self.service.set_quality_status(
            tenant_id="tenant-a",
            document_id=document.document_id,
            quality_status="good",
            note="人工复核通过",
            actor_id="reviewer",
        )
        embedded_before = len(self.embedding.embedded_texts)

        skipped = self.service.create_job(
            tenant_id="tenant-a",
            document_id=document.document_id,
            actor_id="admin-1",
        )

        self.assertEqual(skipped.status, "succeeded")
        self.assertEqual(skipped.message, "skipped_unchanged")
        self.assertTrue(skipped.result["skipped"])
        self.assertEqual(len(self.embedding.embedded_texts), embedded_before)

    def test_acl_update_is_synced_to_vectors(self) -> None:
        document = self._upload()
        job = self.service.create_job(
            tenant_id="tenant-a",
            document_id=document.document_id,
            actor_id="admin-1",
        )
        self.service.run_job(job.job_id)
        new_acl = DocumentAcl(
            tenant_id="tenant-a",
            classification=Classification.CONFIDENTIAL,
            allowed_roles=frozenset({"finance"}),
            owner_id="admin-1",
            acl_version=2,
        )

        self.service.update_acl(
            tenant_id="tenant-a",
            document_id=document.document_id,
            acl=new_acl,
            actor_id="admin-1",
        )

        records = self.vector_store.list_chunk_records(
            tenant_id="tenant-a",
            scope_field="source",
            scope_value=document.source,
        )
        self.assertTrue(records)
        for payload in records.values():
            self.assertEqual(payload["classification"], int(Classification.CONFIDENTIAL))
            self.assertEqual(payload["allowed_roles"], ["finance"])

    def test_cross_tenant_document_is_not_visible(self) -> None:
        document = self._upload("tenant-a")
        with self.assertRaises(ValueError):
            self.service.get_document(
                tenant_id="tenant-b",
                document_id=document.document_id,
            )
        self.assertEqual(
            self.service.list_documents(tenant_id="tenant-b"),
            [],
        )

    def test_chunk_config_change_invalidates_skip(self) -> None:
        document = self._upload()
        job = self.service.create_job(
            tenant_id="tenant-a",
            document_id=document.document_id,
            actor_id="admin-1",
        )
        self.service.run_job(job.job_id)
        self.service.set_quality_status(
            tenant_id="tenant-a",
            document_id=document.document_id,
            quality_status="good",
            note="ok",
            actor_id="reviewer",
        )
        self.service.update_chunk_config(
            tenant_id="tenant-a",
            document_id=document.document_id,
            chunk_strategy="fixed",
            chunk_options={"chunk_size": 60, "chunk_overlap": 5},
            actor_id="admin-1",
        )

        changed = self.service.create_job(
            tenant_id="tenant-a",
            document_id=document.document_id,
            actor_id="admin-1",
        )

        self.assertEqual(changed.status, "pending")
        self.assertFalse(changed.result.get("skipped", False))

    def test_delete_vectors_only_affects_the_tenant(self) -> None:
        first = self._upload("tenant-a")
        second = self._upload("tenant-b")
        for document, tenant in ((first, "tenant-a"), (second, "tenant-b")):
            job = self.service.create_job(
                tenant_id=tenant,
                document_id=document.document_id,
                actor_id="admin-1",
            )
            self.service.run_job(job.job_id)

        self.service.archive_document(
            tenant_id="tenant-a",
            document_id=first.document_id,
            actor_id="admin-1",
            delete_vectors=True,
        )

        self.assertEqual(
            self.vector_store.list_chunk_records(
                tenant_id="tenant-a",
                scope_field="source",
                scope_value=first.source,
            ),
            {},
        )
        self.assertTrue(
            self.vector_store.list_chunk_records(
                tenant_id="tenant-b",
                scope_field="source",
                scope_value=second.source,
            )
        )


if __name__ == "__main__":
    unittest.main()
