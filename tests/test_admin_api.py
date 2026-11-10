from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from api.app import create_app
from configs import RagSettings
from data_processing.knowledge_indexing import BaseEmbedding, KnowledgeIndexer
from knowledge_management import (
    KnowledgeAdminService,
    KnowledgeGovernanceStore,
    KnowledgeIngestionService,
)
from retrieval import RagStreamEvent
from retrieval.vector_store import QdrantVectorStore


class FakeQueryService:
    async def stream(self, question: str, **kwargs):
        yield RagStreamEvent("delta", {"content": "ok"})

    async def aclose(self) -> None:
        return None


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-admin-embedding"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


class AdminApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.vector_store = QdrantVectorStore(
            "admin_api_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
        )
        self.addCleanup(self.vector_store.close)
        self.governance = KnowledgeGovernanceStore(":memory:")
        self.addCleanup(self.governance.close)
        self.ingestion = KnowledgeIngestionService(
            governance_store=self.governance,
            data_dir=Path(self.temp_dir.name) / "uploads",
            indexer=KnowledgeIndexer(FakeEmbedding(), self.vector_store),
            graph_indexer=None,
        )
        self.admin_service = KnowledgeAdminService(self.ingestion)
        self.settings = RagSettings(
            rag_acl_enabled=True,
            rag_acl_proxy_token="gateway-secret",
            rag_graph_enabled=False,
            rag_admin_max_file_bytes=1024 * 1024,
        )

    def _headers(self, **overrides) -> dict[str, str]:
        values = {
            "X-Rag-Proxy-Token": "gateway-secret",
            "X-Tenant-Id": "tenant-a",
            "X-User-Id": "admin-1",
            "X-User-Roles": "knowledge_admin",
            "X-User-Clearance": "confidential",
        }
        values.update(overrides)
        return values

    def _client(self, settings: RagSettings | None = None) -> TestClient:
        app = create_app(
            service=FakeQueryService(),
            admin_service=self.admin_service,
            settings=settings or self.settings,
        )
        return TestClient(app)

    def _upload(self, client: TestClient, **headers):
        return client.post(
            "/v1/admin/documents/upload",
            headers=self._headers(**headers),
            files={
                "file": (
                    "policy.md",
                    "# 差旅政策\n\n员工出差需要提交申请，并在十个工作日内完成报销。".encode(),
                    "text/markdown",
                )
            },
            data={
                "chunk_strategy": "recursive",
                "chunk_options": '{"chunk_size": 80, "chunk_overlap": 10}',
                "classification": "internal",
                "allowed_roles": "finance, hr",
            },
        )

    def test_admin_identity_roles_and_tenant_are_enforced(self) -> None:
        with self._client() as client:
            missing = client.get("/v1/admin/documents")
            viewer = client.get(
                "/v1/admin/documents",
                headers=self._headers(**{"X-User-Roles": "viewer"}),
            )
            me = client.get("/v1/admin/me", headers=self._headers())

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(viewer.status_code, 403)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["tenant_id"], "tenant-a")

    def test_admin_requires_proxy_token_configuration(self) -> None:
        settings = RagSettings(
            rag_acl_enabled=True,
            rag_acl_proxy_token=None,
            rag_graph_enabled=False,
        )
        with self._client(settings) as client:
            response = client.get("/v1/admin/documents", headers=self._headers())
        self.assertEqual(response.status_code, 503)

    def test_upload_preview_build_list_and_audit(self) -> None:
        with self._client() as client:
            uploaded = self._upload(client)
            self.assertEqual(uploaded.status_code, 201, uploaded.text)
            document = uploaded.json()

            preview = client.post(
                f"/v1/admin/documents/{document['document_id']}/preview",
                headers=self._headers(),
            )
            created_job = client.post(
                f"/v1/admin/documents/{document['document_id']}/jobs",
                headers=self._headers(),
                json={"force_reindex": False},
            )
            self.assertEqual(created_job.status_code, 202, created_job.text)
            job_id = created_job.json()["job_id"]
            job = client.get(f"/v1/admin/jobs/{job_id}", headers=self._headers())
            documents = client.get("/v1/admin/documents", headers=self._headers())
            audit = client.get("/v1/admin/audit", headers=self._headers())

        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertGreater(preview.json()["chunk_count"], 0)
        self.assertEqual(job.status_code, 200, job.text)
        self.assertTrue(job.json()["status"] in {"running", "succeeded"})
        self.assertEqual(documents.json()["count"], 1)
        actions = {item["action"] for item in audit.json()["items"]}
        self.assertIn("document.upload", actions)

    def test_cross_tenant_and_acl_updates(self) -> None:
        with self._client() as client:
            document = self._upload(client).json()
            other_tenant = client.get(
                "/v1/admin/documents",
                headers=self._headers(**{"X-Tenant-Id": "tenant-b"}),
            )
            forbidden = client.get(
                f"/v1/admin/documents/{document['document_id']}",
                headers=self._headers(**{"X-User-Clearance": "public"}),
            )
            updated = client.patch(
                f"/v1/admin/documents/{document['document_id']}/acl",
                headers=self._headers(),
                json={
                    "classification": "confidential",
                    "allowed_roles": ["finance"],
                    "allowed_groups": [],
                    "allowed_users": [],
                    "acl_version": 1,
                },
            )

        self.assertEqual(other_tenant.status_code, 200)
        self.assertEqual(other_tenant.json()["count"], 0)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["acl"]["classification"], 2)

    def test_admin_console_is_served(self) -> None:
        with self._client() as client:
            page = client.get("/admin")
            script = client.get("/ui/app.js")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Atlas Knowledge", page.text)
        self.assertEqual(script.status_code, 200)

    def test_readiness_reports_running_worker(self) -> None:
        with self._client() as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["ingestion_worker"]["enabled"])
        self.assertTrue(body["ingestion_worker"]["running"])


if __name__ == "__main__":
    unittest.main()
