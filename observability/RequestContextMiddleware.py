from __future__ import annotations

import logging
import re
import uuid
from time import perf_counter

from starlette.types import ASGIApp, Receive, Scope, Send

from .RagMetrics import RagMetrics
from .RequestContext import RequestContext, bind_request_context, reset_request_context


logger = logging.getLogger("enterprise_rag.access")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestContextMiddleware:
    """为请求生成关联 ID，并记录 HTTP 访问日志和指标。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: RagMetrics,
        access_log_enabled: bool = True,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.access_log_enabled = access_log_enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = _headers(scope)
        request_id = _request_id(headers.get("x-request-id"))
        trace_id = _trace_id(headers.get("traceparent"))
        context = RequestContext(request_id=request_id, trace_id=trace_id)
        token = bind_request_context(context)
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        skip_self_metrics = path == "/metrics"
        started_at = perf_counter()
        status_code = 500

        async def send_with_context(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = message.setdefault("headers", [])
                _append_header(response_headers, b"x-request-id", request_id)
                _append_header(response_headers, b"x-trace-id", trace_id)
            await send(message)

        if not skip_self_metrics:
            self.metrics.increment_http_in_flight(method)
        try:
            await self.app(scope, receive, send_with_context)
        except Exception as exc:
            status_code = 500
            logger.exception(
                "HTTP 请求处理异常",
                extra={
                    "event": "http.error",
                    "method": method,
                    "status": status_code,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            duration_seconds = perf_counter() - started_at
            route = _route_label(scope, skip_self_metrics)
            if not skip_self_metrics:
                self.metrics.decrement_http_in_flight(method)
                self.metrics.observe_http_request(
                    method=method,
                    route=route,
                    status_code=status_code,
                    duration_seconds=duration_seconds,
                )
                if status_code in {401, 403}:
                    self.metrics.observe_acl_denial(f"http_{status_code}")
            if self.access_log_enabled and not skip_self_metrics:
                logger.log(
                    logging.ERROR if status_code >= 500 else logging.INFO,
                    "HTTP 请求完成",
                    extra={
                        "event": "http.request",
                        "method": method,
                        "route": route,
                        "status": status_code,
                        "duration_ms": round(duration_seconds * 1000, 2),
                    },
                )
            reset_request_context(token)


def _headers(scope: Scope) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in scope.get("headers", []):
        try:
            values[key.decode("latin-1").lower()] = value.decode("latin-1")
        except UnicodeDecodeError:
            continue
    return values


def _request_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if _REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _trace_id(traceparent: str | None) -> str:
    candidate = (traceparent or "").strip().split("-")
    if len(candidate) >= 4:
        trace_id = candidate[1].lower()
        if len(trace_id) == 32 and trace_id != "0" * 32:
            try:
                int(trace_id, 16)
            except ValueError:
                pass
            else:
                return trace_id
    return uuid.uuid4().hex


def _append_header(headers: list[tuple[bytes, bytes]], key: bytes, value: str) -> None:
    if any(existing.lower() == key for existing, _ in headers):
        return
    headers.append((key, value.encode("latin-1", errors="replace")))


def _route_label(scope: Scope, skip_self_metrics: bool) -> str:
    if skip_self_metrics:
        return "/metrics"
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    root_path = str(scope.get("root_path") or "")
    if root_path == "/ui" or root_path.startswith("/ui/"):
        return "/ui/*"
    return "unmatched"


__all__ = ["RequestContextMiddleware"]
