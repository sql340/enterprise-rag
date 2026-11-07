from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from access_control import Classification
from api.app import create_app
from configs import RagSettings
from retrieval import RagStreamEvent


class FakeQueryService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def stream(self, question: str, **kwargs):
        self.calls.append({"question": question, **kwargs})
        yield RagStreamEvent("delta", {"content": "ok"})
        yield RagStreamEvent("done", {"finish_reason": "stop", "elapsed_ms": 1.0})

    async def aclose(self) -> None:
        return None


class ApiAclTest(unittest.TestCase):
    def _settings(self, **overrides) -> RagSettings:
        values = {
            "rag_acl_enabled": True,
            "rag_acl_proxy_token": "gateway-secret",
        }
        values.update(overrides)
        return RagSettings(**values)

    def _headers(self, **overrides) -> dict[str, str]:
        values = {
            "X-Rag-Proxy-Token": "gateway-secret",
            "X-Tenant-Id": "tenant-a",
            "X-User-Id": "user-1",
            "X-User-Clearance": "internal",
        }
        values.update(overrides)
        return values

    def test_query_rejects_missing_or_invalid_identity(self) -> None:
        service = FakeQueryService()
        app = create_app(service=service, settings=self._settings())

        with TestClient(app) as client:
            missing = client.post("/v1/query", json={"question": "policy"})
            wrong_token = client.post(
                "/v1/query",
                json={"question": "policy"},
                headers=self._headers(**{"X-Rag-Proxy-Token": "forged"}),
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong_token.status_code, 401)
        self.assertEqual(service.calls, [])

    def test_query_passes_resolved_principal_to_service(self) -> None:
        service = FakeQueryService()
        app = create_app(service=service, settings=self._settings())

        with TestClient(app) as client:
            response = client.post(
                "/v1/query",
                json={"question": "policy", "top_k": 3},
                headers=self._headers(**{"X-User-Roles": "finance"}),
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: delta", response.text)
        self.assertEqual(len(service.calls), 1)
        principal = service.calls[0]["principal"]
        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.user_id, "user-1")
        self.assertEqual(principal.roles, frozenset({"finance"}))
        self.assertEqual(principal.clearance, Classification.INTERNAL)

    def test_query_fails_closed_when_proxy_token_is_not_configured(self) -> None:
        service = FakeQueryService()
        app = create_app(
            service=service,
            settings=self._settings(rag_acl_proxy_token=None),
        )

        with TestClient(app) as client:
            response = client.post(
                "/v1/query",
                json={"question": "policy"},
                headers=self._headers(),
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(service.calls, [])

    def test_request_body_cannot_override_acl_filters(self) -> None:
        service = FakeQueryService()
        app = create_app(service=service, settings=self._settings())

        with TestClient(app) as client:
            response = client.post(
                "/v1/query",
                json={
                    "question": "policy",
                    "filters": {"tenant_id": "tenant-b"},
                },
                headers=self._headers(),
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(service.calls, [])

    def test_acl_can_be_disabled_for_legacy_deployments(self) -> None:
        service = FakeQueryService()
        app = create_app(
            service=service,
            settings=self._settings(rag_acl_enabled=False),
        )

        with TestClient(app) as client:
            response = client.post("/v1/query", json={"question": "policy"})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(service.calls[0]["principal"])


if __name__ == "__main__":
    unittest.main()
