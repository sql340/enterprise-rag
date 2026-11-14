from __future__ import annotations

import asyncio
import json
import logging
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.app import create_app
from configs import RagSettings
from data_processing.knowledge_indexing import BaseEmbedding
from knowledge_management import IngestionJobWorker
from observability import (
    DependencyHealthChecker,
    JsonLogFormatter,
    RagMetrics,
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from retrieval import KnowledgeRetriever, PromptBuilder, RagQueryService, RagStreamEvent
from retrieval.vector_store import VectorSearchResult


class FakeQueryService:
    async def stream(self, question: str, **kwargs):
        yield RagStreamEvent("delta", {"content": "ok"})
        yield RagStreamEvent("done", {"finish_reason": "stop", "elapsed_ms": 1.0})

    async def aclose(self) -> None:
        return None


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-observability-embedding"

    @property
    def dimension(self) -> int:
        return 2

    @property
    def is_loaded(self) -> bool:
        return True

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


class UnloadedEmbedding(FakeEmbedding):
    @property
    def is_loaded(self) -> bool:
        return False


class FakeVectorStore:
    supports_text_search = False
    collection_name = "observability_test"

    def search(self, query_vector, *, limit=5, score_threshold=None, filters=None):
        return [
            VectorSearchResult(
                "chunk-1",
                0.9,
                {"content": "policy", "source": "policy.md"},
            )
        ][:limit]

    def search_index_status(self) -> dict[str, object]:
        return {
            "ready": True,
            "collection_exists": True,
            "missing_payload_indexes": [],
            "text_index": True,
        }

    def close(self) -> None:
        return None


class FakeLLM:
    model_name = "fake-observability-llm"

    async def stream_chat(self, messages, *, temperature, max_tokens):
        yield "hello"

    async def aclose(self) -> None:
        return None


class ObservabilityApiTest(unittest.TestCase):
    def _settings(self, **overrides) -> RagSettings:
        values = {
            "rag_acl_enabled": False,
            "rag_graph_enabled": False,
            "rag_ingestion_worker_enabled": False,
            "rag_access_log_enabled": False,
            "rag_log_format": "text",
            "rag_dependency_health_external_enabled": False,
            "qdrant_collection": "observability_test",
            "qdrant_enable_text_index": False,
        }
        values.update(overrides)
        return RagSettings(**values)

    def _service(self, embedding: BaseEmbedding | None = None) -> RagQueryService:
        metrics = RagMetrics()
        retriever = KnowledgeRetriever(
            embedding or FakeEmbedding(),
            FakeVectorStore(),
            retrieval_mode="dense",
            metrics=metrics,
        )
        return RagQueryService(
            retriever,
            FakeLLM(),
            PromptBuilder(),
            metrics=metrics,
        )

    def test_request_id_and_http_metrics(self) -> None:
        app = create_app(service=FakeQueryService(), settings=self._settings())
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

        with TestClient(app) as client:
            response = client.get(
                "/health",
                headers={
                    "X-Request-ID": "request-123",
                    "traceparent": f"00-{trace_id}-00f067aa0ba902b7-01",
                },
            )
            metrics = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "request-123")
        self.assertEqual(response.headers["x-trace-id"], trace_id)
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("rag_http_requests_total", metrics.text)
        self.assertIn('route="/health"', metrics.text)

    def test_query_metrics_are_exported(self) -> None:
        app = create_app(service=FakeQueryService(), settings=self._settings())

        with TestClient(app) as client:
            response = client.post(
                "/v1/query",
                json={"question": "policy"},
                headers={"X-Request-ID": "query-123"},
            )
            metrics = client.get("/metrics")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["x-request-id"], "query-123")
        self.assertIn("event: delta", response.text)
        self.assertIn("rag_query_requests_total", metrics.text)
        self.assertIn('outcome="success"', metrics.text)
        self.assertIn("rag_query_first_token_seconds", metrics.text)

    def test_static_assets_use_stable_route_label(self) -> None:
        app = create_app(service=FakeQueryService(), settings=self._settings())

        with TestClient(app) as client:
            response = client.get("/ui/app.js")
            metrics = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn('route="/ui/*"', metrics.text)

    def test_metrics_endpoint_can_be_disabled(self) -> None:
        app = create_app(
            service=FakeQueryService(),
            settings=self._settings(rag_metrics_enabled=False),
        )

        with TestClient(app) as client:
            response = client.get("/metrics")

        self.assertEqual(response.status_code, 404)

    def test_retrieval_and_llm_metrics_are_recorded(self) -> None:
        service = self._service()
        metrics = service.metrics
        self.assertIsNotNone(metrics)

        async def consume() -> None:
            async for _ in service.stream("policy"):
                pass

        asyncio.run(consume())
        rendered = metrics.render().decode("utf-8")

        self.assertIn("rag_retrieval_requests_total", rendered)
        self.assertIn('mode="dense"', rendered)
        self.assertIn("rag_embedding_duration_seconds", rendered)
        self.assertIn("rag_llm_ttft_seconds", rendered)
        self.assertIn("rag_llm_stream_duration_seconds", rendered)

    def test_unloaded_embedding_degrades_readiness(self) -> None:
        app = create_app(
            service=self._service(UnloadedEmbedding()),
            settings=self._settings(),
        )

        with TestClient(app) as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "degraded")
        self.assertFalse(body["dependencies"]["embedding"]["ready"])

    def test_required_dependency_failure_fails_readiness(self) -> None:
        app = create_app(
            service=self._service(UnloadedEmbedding()),
            settings=self._settings(
                rag_dependency_health_fail_readiness=True,
            ),
        )

        with TestClient(app) as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["status"], "not_ready")


class JsonLogFormatterTest(unittest.TestCase):
    def test_formatter_adds_request_context_and_allowlisted_fields(self) -> None:
        token = bind_request_context(
            RequestContext(request_id="req-1", trace_id="trace-1")
        )
        try:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="hello %s",
                args=("world",),
                exc_info=None,
            )
            record.event = "test.event"
            record.duration_ms = 12.5
            record.secret = "must-not-be-logged"
            payload = json.loads(JsonLogFormatter().format(record))
        finally:
            reset_request_context(token)

        self.assertEqual(payload["message"], "hello world")
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["trace_id"], "trace-1")
        self.assertEqual(payload["event"], "test.event")
        self.assertEqual(payload["duration_ms"], 12.5)
        self.assertNotIn("secret", payload)


class WorkerMetricsTest(unittest.TestCase):
    def test_queue_depth_is_exported(self) -> None:
        metrics = RagMetrics()

        class Store:
            def count_jobs_by_status(self) -> dict[str, int]:
                return {"pending": 2, "running": 1, "succeeded": 5}

        class Ingestion:
            governance_store = Store()

        worker = IngestionJobWorker(Ingestion(), metrics=metrics)
        worker._refresh_queue_depth()
        rendered = metrics.render().decode("utf-8")

        self.assertIn('rag_ingestion_queue_depth{status="pending"} 2.0', rendered)
        self.assertIn('rag_ingestion_queue_depth{status="running"} 1.0', rendered)
        self.assertIn('rag_ingestion_queue_depth{status="failed"} 0.0', rendered)


class DependencyHealthCheckerTest(unittest.TestCase):
    def test_external_check_is_cached_and_reported(self) -> None:
        calls = 0

        def http_checker(url, headers, timeout):
            nonlocal calls
            calls += 1
            return True, None

        settings = RagSettings(
            rag_graph_enabled=False,
            rag_dependency_health_llm_url="http://llm.internal/health",
        )
        service = SimpleNamespace(
            llm=SimpleNamespace(
                model_name="test-llm",
                base_url="http://llm.internal/v1",
                api_key=None,
            )
        )
        checker = DependencyHealthChecker(
            service,
            settings,
            http_checker=http_checker,
        )

        first = checker.check()
        second = checker.check()

        self.assertTrue(first["llm"]["ready"])
        self.assertEqual(first, second)
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
