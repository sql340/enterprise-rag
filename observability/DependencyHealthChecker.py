from __future__ import annotations

import copy
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from time import monotonic, perf_counter
from typing import Any

from configs import RagSettings

from .RagMetrics import RagMetrics


_HttpCheck = Callable[[str, dict[str, str], float], tuple[bool, str | None]]


class DependencyHealthChecker:
    """检查 Embedding / LLM 依赖，并缓存结果避免探针风暴。"""

    def __init__(
        self,
        service: Any,
        settings: RagSettings,
        *,
        metrics: RagMetrics | None = None,
        http_checker: _HttpCheck | None = None,
    ) -> None:
        self.service = service
        self.settings = settings
        self.metrics = metrics
        self.http_checker = http_checker or _default_http_check
        self._lock = threading.Lock()
        self._cached: dict[str, dict[str, Any]] | None = None
        self._checked_at = 0.0

    def check(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        now = monotonic()
        with self._lock:
            if (
                not force
                and self._cached is not None
                and now - self._checked_at
                < self.settings.rag_dependency_health_cache_seconds
            ):
                return copy.deepcopy(self._cached)
            result = self._check_all()
            self._cached = result
            self._checked_at = now
        self._record_metrics(result)
        return copy.deepcopy(result)

    def _check_all(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        embedding = getattr(getattr(self.service, "retriever", None), "embedding", None)
        if embedding is not None:
            result["embedding"] = self._check_embedding(embedding)
        llm = getattr(self.service, "llm", None)
        if llm is not None:
            result["llm"] = self._check_llm(llm)
        return result

    def _check_embedding(self, embedding: Any) -> dict[str, Any]:
        model = str(getattr(embedding, "model_name", "unknown"))
        if hasattr(embedding, "is_loaded"):
            loaded = bool(getattr(embedding, "is_loaded", False))
            return {
                "type": "local",
                "model": model,
                "ready": loaded,
                "loaded": loaded,
                "detail": "model_loaded" if loaded else "model_not_loaded",
                "latency_ms": 0.0,
            }

        if (
            self.settings.rag_dependency_health_external_enabled
            and getattr(embedding, "base_url", None)
        ):
            url = (
                self.settings.rag_dependency_health_embedding_url
                or _models_url(str(embedding.base_url))
            )
            return self._check_http(
                dependency="embedding",
                dependency_type="api",
                url=url,
                api_key=getattr(embedding, "api_key", None),
                model=model,
            )

        return {
            "type": "custom",
            "model": model,
            "ready": True,
            "skipped": True,
            "detail": "health_check_not_supported",
            "latency_ms": 0.0,
        }

    def _check_llm(self, llm: Any) -> dict[str, Any]:
        model = str(getattr(llm, "model_name", "unknown"))
        if not (
            self.settings.rag_dependency_health_external_enabled
            and getattr(llm, "base_url", None)
        ):
            return {
                "type": "custom",
                "model": model,
                "ready": True,
                "skipped": True,
                "detail": "health_check_not_supported",
                "latency_ms": 0.0,
            }
        url = (
            self.settings.rag_dependency_health_llm_url
            or _models_url(str(llm.base_url))
        )
        return self._check_http(
            dependency="llm",
            dependency_type="api",
            url=url,
            api_key=getattr(llm, "api_key", None),
            model=model,
        )

    def _check_http(
        self,
        *,
        dependency: str,
        dependency_type: str,
        url: str,
        api_key: str | None,
        model: str,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        started_at = perf_counter()
        try:
            ready, detail = self.http_checker(
                url,
                headers,
                self.settings.rag_dependency_health_timeout_seconds,
            )
        except Exception as exc:
            ready = False
            detail = type(exc).__name__
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return {
            "type": dependency_type,
            "model": model,
            "url": url,
            "ready": ready,
            "detail": detail,
            "latency_ms": latency_ms,
        }

    def _record_metrics(self, result: dict[str, dict[str, Any]]) -> None:
        if self.metrics is None:
            return
        for name, value in result.items():
            self.metrics.set_dependency_status(
                dependency=name,
                ready=bool(value.get("ready")),
                latency_seconds=float(value.get("latency_ms", 0.0)) / 1000.0,
            )


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}/models"


def _default_http_check(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> tuple[bool, str | None]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            if 200 <= status < 300:
                return True, None
            return False, f"http_{status}"
    except urllib.error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except urllib.error.URLError as exc:
        return False, f"connection_error:{exc.reason}"
    except TimeoutError:
        return False, "timeout"
    except Exception as exc:
        return False, type(exc).__name__


__all__ = ["DependencyHealthChecker"]
