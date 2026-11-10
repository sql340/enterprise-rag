from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from qdrant_client import models

from access_control import Classification, DocumentAcl
from api.app import _build_vector_store, create_app
from configs import RagSettings
from data_processing.knowledge_indexing import BaseEmbedding, IndexingResult
from knowledge_management import (
    IngestionJob,
    IngestionJobWorker,
    KnowledgeDocument,
    KnowledgeGovernanceStore,
    KnowledgeIngestionService,
)
from retrieval.vector_store import QdrantVectorStore


class FakeQdrantClient:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.dimension = 2
        self.indexes: dict[str, object] = {}
        self.fail_text_index = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def get_collection(self, collection_name: str) -> object:
        vectors = SimpleNamespace(size=self.dimension)
        params = SimpleNamespace(vectors=vectors)
        config = SimpleNamespace(params=params)
        return SimpleNamespace(
            payload_schema=dict(self.indexes),
            config=config,
        )

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: object,
        on_disk_payload: bool,
    ) -> None:
        self.exists = True
        self.dimension = int(getattr(vectors_config, "size"))

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: object,
        wait: bool,
    ) -> None:
        if field_name == "content" and self.fail_text_index:
            raise RuntimeError("text index failed")
        self.indexes[field_name] = field_schema


class QdrantIndexBootstrapTest(unittest.TestCase):
    def test_existing_collection_gets_acl_and_text_indexes(self) -> None:
        client = FakeQdrantClient(exists=True)
        store = QdrantVectorStore(
            "bootstrap_test",
            client=client,
            create_payload_indexes=True,
            enable_text_index=True,
        )

        status = store.ensure_search_indexes()

        self.assertTrue(status["ready"])
        self.assertTrue(status["collection_exists"])
        self.assertIn("tenant_id", client.indexes)
        self.assertIn("classification", client.indexes)
        self.assertIn("content", client.indexes)
        self.assertEqual(status["missing_payload_indexes"], [])

    def test_failed_text_index_is_retried_on_next_bootstrap(self) -> None:
        client = FakeQdrantClient(exists=True)
        client.fail_text_index = True
        store = QdrantVectorStore(
            "bootstrap_test",
            client=client,
            create_payload_indexes=True,
            enable_text_index=True,
        )

        first = store.ensure_search_indexes()
        client.fail_text_index = False
        second = store.ensure_search_indexes()

        self.assertFalse(first["ready"])
        self.assertFalse(first["text_index"])
        self.assertTrue(second["ready"])
        self.assertTrue(second["text_index"])

    def test_missing_collection_is_reported_as_not_ready(self) -> None:
        store = QdrantVectorStore(
            "missing_collection",
            client=FakeQdrantClient(exists=False),
            create_payload_indexes=True,
        )

        status = store.ensure_search_indexes()

        self.assertFalse(status["ready"])
        self.assertFalse(status["collection_exists"])

    def test_production_vector_store_creates_payload_indexes(self) -> None:
        store = _build_vector_store(
            RagSettings(
                qdrant_url=":memory:",
                qdrant_collection="production_defaults",
            )
        )
        self.addCleanup(store.close)

        self.assertTrue(store.create_payload_indexes)


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-dimension-embedding"

    @property
    def dimension(self) -> int:
        return 4

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    collection_name = "readiness_test"

    def __init__(self) -> None:
        self.collection_exists = False
        self.ready = False
        self.dimension: int | None = None

    def search_index_status(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "collection_exists": self.collection_exists,
            "missing_payload_indexes": [] if self.ready else ["tenant_id"],
            "text_index": self.ready,
        }

    def ensure_collection(self, dimension: int) -> None:
        self.collection_exists = True
        self.ready = True
        self.dimension = dimension

    def ensure_search_indexes(self) -> dict[str, object]:
        return self.search_index_status()


class FakeRuntime:
    def __init__(self, vector_store: FakeVectorStore) -> None:
        self.retriever = SimpleNamespace(
            embedding=FakeEmbedding(),
            vector_store=vector_store,
        )

    async def aclose(self) -> None:
        return None


class ReadinessApiTest(unittest.TestCase):
    def _settings(self, *, bootstrap: bool = True) -> RagSettings:
        return RagSettings(
            rag_graph_enabled=False,
            rag_ingestion_worker_enabled=False,
            qdrant_collection="readiness_test",
            qdrant_enable_text_index=False,
            rag_bootstrap_search_indexes=bootstrap,
        )

    def test_startup_creates_missing_collection_then_reports_ready(self) -> None:
        vector_store = FakeVectorStore()
        app = create_app(
            service=FakeRuntime(vector_store),
            settings=self._settings(),
            job_worker_enabled=False,
        )

        with TestClient(app) as client:
            health = client.get("/health")
            ready = client.get("/ready")

        self.assertEqual(vector_store.dimension, 4)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.status_code, 200, ready.text)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertTrue(ready.json()["search_indexes"]["ready"])

    def test_missing_collection_fails_readiness_when_bootstrap_disabled(self) -> None:
        vector_store = FakeVectorStore()
        app = create_app(
            service=FakeRuntime(vector_store),
            settings=self._settings(bootstrap=False),
            job_worker_enabled=False,
        )

        with TestClient(app) as client:
            health = client.get("/health")
            ready = client.get("/ready")

        self.assertIsNone(vector_store.dimension)
        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.status_code, 503, ready.text)
        self.assertEqual(ready.json()["status"], "not_ready")


def _document(document_id: str = "doc-1") -> KnowledgeDocument:
    acl = DocumentAcl(
        tenant_id="tenant-a",
        classification=Classification.INTERNAL,
        owner_id="admin-1",
    )
    return KnowledgeDocument(
        document_id=document_id,
        tenant_id="tenant-a",
        source=f"upload://tenant-a/{document_id}/policy.md",
        file_name="policy.md",
        file_type="markdown",
        storage_path=f"/tmp/{document_id}.md",
        content_hash=f"hash-{document_id}",
        acl=acl.to_payload(),
        status="queued",
    )


def _job(document: KnowledgeDocument, job_id: str = "job-1") -> IngestionJob:
    return IngestionJob(
        job_id=job_id,
        tenant_id=document.tenant_id,
        document_id=document.document_id,
    )


class StubIngestion:
    def __init__(self, store: KnowledgeGovernanceStore) -> None:
        self.governance_store = store
        self.executed: list[tuple[str, str, int]] = []

    def run_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> IngestionJob:
        job = self.governance_store.get_job_internal(job_id)
        assert job is not None
        self.executed.append((job_id, worker_id, job.attempt_count))
        job.status = "succeeded"
        job.progress = 100
        job.message = "completed"
        self.governance_store.save_job(job)
        return job


class StealingIndexer:
    def __init__(
        self,
        store: KnowledgeGovernanceStore,
        job_id: str,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.embedding = FakeEmbedding()
        self.vector_store = SimpleNamespace(collection_name="lease_loss_test")

    def index_chunks(self, chunks, **kwargs) -> IndexingResult:
        expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        self.store._connection.execute(
            "UPDATE knowledge_jobs SET lease_expires_at = ? WHERE job_id = ?",
            (expired, self.job_id),
        )
        self.store._connection.commit()
        self.store.claim_next_job(
            worker_id="worker-2",
            lease_seconds=60,
            max_attempts=3,
        )
        document_ids = tuple(sorted({chunk.document_id for chunk in chunks}))
        return IndexingResult(
            indexed_chunks=len(chunks),
            document_ids=document_ids,
            sources=(),
            collection_name=self.vector_store.collection_name,
            total_chunks=len(chunks),
            embedded_chunks=len(chunks),
        )


class IngestionLeaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "governance.sqlite3"

    def _store(self) -> KnowledgeGovernanceStore:
        store = KnowledgeGovernanceStore(self.db_path)
        self.addCleanup(store.close)
        return store

    @staticmethod
    def _expire_lease(store: KnowledgeGovernanceStore, job_id: str) -> None:
        expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        store._connection.execute(
            "UPDATE knowledge_jobs SET lease_expires_at = ? WHERE job_id = ?",
            (expired, job_id),
        )
        store._connection.commit()

    def test_claim_is_atomic_and_expired_lease_is_reclaimed(self) -> None:
        first = self._store()
        second = self._store()
        document = _document()
        job = _job(document)
        first.save_document(document)
        first.create_job(job)

        claimed = first.claim_next_job(
            worker_id="worker-1",
            lease_seconds=60,
            max_attempts=3,
        )
        blocked = second.claim_next_job(
            worker_id="worker-2",
            lease_seconds=60,
            max_attempts=3,
        )
        wrong_heartbeat = first.heartbeat_job(
            job_id=job.job_id,
            worker_id="worker-2",
            lease_seconds=60,
        )
        right_heartbeat = first.heartbeat_job(
            job_id=job.job_id,
            worker_id="worker-1",
            lease_seconds=60,
        )
        self._expire_lease(first, job.job_id)
        stale_heartbeat = first.heartbeat_job(
            job_id=job.job_id,
            worker_id="worker-1",
            lease_seconds=60,
        )
        reclaimed = second.claim_next_job(
            worker_id="worker-2",
            lease_seconds=60,
            max_attempts=3,
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.attempt_count, 1)
        self.assertIsNone(blocked)
        self.assertFalse(wrong_heartbeat)
        self.assertTrue(right_heartbeat)
        self.assertFalse(stale_heartbeat)
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.worker_id, "worker-2")
        self.assertEqual(reclaimed.attempt_count, 2)

    def test_expired_lease_at_max_attempts_becomes_failed(self) -> None:
        store = self._store()
        document = _document()
        job = _job(document)
        store.save_document(document)
        store.create_job(job)
        expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        store._connection.execute(
            """
            UPDATE knowledge_jobs
            SET status = 'running', worker_id = 'dead-worker',
                lease_expires_at = ?, attempt_count = 3
            WHERE job_id = ?
            """,
            (expired, job.job_id),
        )
        store._connection.commit()

        claimed = store.claim_next_job(
            worker_id="worker-2",
            lease_seconds=60,
            max_attempts=3,
        )
        failed_job = store.get_job_internal(job.job_id)
        failed_document = store.get_document(
            tenant_id=document.tenant_id,
            document_id=document.document_id,
        )

        self.assertIsNone(claimed)
        self.assertIsNotNone(failed_job)
        self.assertEqual(failed_job.status, "failed")
        self.assertIn("最大重试次数", failed_job.error)
        self.assertEqual(failed_document.status, "failed")

    def test_legacy_running_job_without_lease_is_recovered(self) -> None:
        store = self._store()
        document = _document()
        job = _job(document)
        store.save_document(document)
        store.create_job(job)
        store._connection.execute(
            """
            UPDATE knowledge_jobs
            SET status = 'running', worker_id = 'legacy-background-task',
                lease_expires_at = NULL
            WHERE job_id = ?
            """,
            (job.job_id,),
        )
        store._connection.commit()

        reclaimed = store.claim_next_job(
            worker_id="durable-worker",
            lease_seconds=60,
            max_attempts=3,
        )

        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.worker_id, "durable-worker")
        self.assertEqual(reclaimed.attempt_count, 1)

    def test_worker_recovers_expired_job_after_process_restart(self) -> None:
        old_store = KnowledgeGovernanceStore(self.db_path)
        document = _document()
        job = _job(document)
        old_store.save_document(document)
        old_store.create_job(job)
        claimed = old_store.claim_next_job(
            worker_id="crashed-worker",
            lease_seconds=60,
            max_attempts=3,
        )
        self.assertIsNotNone(claimed)
        self._expire_lease(old_store, job.job_id)
        old_store.close()

        new_store = self._store()
        ingestion = StubIngestion(new_store)
        worker = IngestionJobWorker(
            ingestion,
            worker_id="restarted-worker",
            poll_seconds=0.01,
            lease_seconds=60,
            max_attempts=3,
        )
        worker.start()
        try:
            deadline = time.monotonic() + 2.0
            recovered = None
            while time.monotonic() < deadline:
                recovered = new_store.get_job_internal(job.job_id)
                if recovered is not None and recovered.status == "succeeded":
                    break
                time.sleep(0.01)
        finally:
            worker.stop()

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, "succeeded")
        self.assertEqual(recovered.attempt_count, 2)
        self.assertEqual(len(ingestion.executed), 1)
        self.assertEqual(ingestion.executed[0][1], "restarted-worker")

    def test_stale_worker_does_not_overwrite_a_new_lease_owner(self) -> None:
        store = self._store()
        document = _document()
        job = _job(document)
        store.save_document(document)
        store.create_job(job)
        claimed = store.claim_next_job(
            worker_id="worker-1",
            lease_seconds=60,
            max_attempts=3,
        )
        self.assertIsNotNone(claimed)
        upload_dir = Path(self.temp_dir.name) / "uploads"
        source = upload_dir / "policy.md"
        upload_dir.mkdir(parents=True, exist_ok=True)
        source.write_text("# Policy\n\nEmployees submit travel requests.", encoding="utf-8")
        document.storage_path = str(source)
        store.save_document(document)
        service = KnowledgeIngestionService(
            governance_store=store,
            data_dir=upload_dir,
            indexer=StealingIndexer(store, job.job_id),
        )

        service.run_job(
            job.job_id,
            worker_id="worker-1",
            lease_seconds=60,
        )
        current = store.get_job_internal(job.job_id)

        self.assertIsNotNone(current)
        self.assertEqual(current.status, "running")
        self.assertEqual(current.worker_id, "worker-2")
        self.assertEqual(current.attempt_count, 2)


class GovernanceSchemaMigrationTest(unittest.TestCase):
    def test_existing_job_table_gets_lease_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "governance.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE knowledge_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    error TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.close()

            store = KnowledgeGovernanceStore(db_path)
            try:
                columns = {
                    str(row["name"])
                    for row in store._connection.execute(
                        "PRAGMA table_info(knowledge_jobs)"
                    ).fetchall()
                }
            finally:
                store.close()

        self.assertTrue(
            {
                "attempt_count",
                "worker_id",
                "lease_expires_at",
                "heartbeat_at",
            }.issubset(columns)
        )


if __name__ == "__main__":
    unittest.main()
