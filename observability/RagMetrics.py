from __future__ import annotations

import time
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


_HTTP_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)
_RETRIEVAL_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
_LLM_TTFT_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)
_LLM_STREAM_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)
_INGESTION_BUCKETS = (
    0.1,
    0.5,
    1.0,
    5.0,
    15.0,
    30.0,
    60.0,
    300.0,
    900.0,
    1800.0,
    3600.0,
)


class RagMetrics:
    """应用内 Prometheus 指标注册表，每个应用实例独立。"""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.registry = CollectorRegistry(auto_describe=True)
        if not enabled:
            return

        registry = self.registry
        self.http_requests = Counter(
            "rag_http_requests_total",
            "HTTP 请求数",
            ("method", "route", "status"),
            registry=registry,
        )
        self.http_duration = Histogram(
            "rag_http_request_duration_seconds",
            "HTTP 请求耗时",
            ("method", "route"),
            buckets=_HTTP_BUCKETS,
            registry=registry,
        )
        self.http_in_flight = Gauge(
            "rag_http_requests_in_flight",
            "当前处理中的 HTTP 请求数",
            ("method",),
            registry=registry,
        )
        self.query_requests = Counter(
            "rag_query_requests_total",
            "RAG 查询数",
            ("outcome",),
            registry=registry,
        )
        self.query_duration = Histogram(
            "rag_query_duration_seconds",
            "RAG 查询耗时",
            ("outcome",),
            buckets=_LLM_STREAM_BUCKETS,
            registry=registry,
        )
        self.query_first_token = Histogram(
            "rag_query_first_token_seconds",
            "从查询开始到首个回答增量的耗时",
            buckets=_LLM_TTFT_BUCKETS,
            registry=registry,
        )
        self.retrieval_requests = Counter(
            "rag_retrieval_requests_total",
            "知识召回请求数",
            ("mode", "outcome"),
            registry=registry,
        )
        self.retrieval_duration = Histogram(
            "rag_retrieval_duration_seconds",
            "知识召回耗时",
            ("mode", "outcome"),
            buckets=_RETRIEVAL_BUCKETS,
            registry=registry,
        )
        self.retrieval_results = Histogram(
            "rag_retrieval_results_count",
            "知识召回返回结果数",
            ("mode",),
            buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
            registry=registry,
        )
        self.retrieval_empty = Counter(
            "rag_retrieval_empty_total",
            "知识召回为空次数",
            ("mode",),
            registry=registry,
        )
        self.retrieval_route_duration = Histogram(
            "rag_retrieval_route_duration_seconds",
            "单条召回链路耗时",
            ("route", "outcome"),
            buckets=_RETRIEVAL_BUCKETS,
            registry=registry,
        )
        self.retrieval_route_errors = Counter(
            "rag_retrieval_route_errors_total",
            "单条召回链路错误数",
            ("route", "error_type"),
            registry=registry,
        )
        self.embedding_requests = Counter(
            "rag_embedding_requests_total",
            "Embedding 请求数",
            ("model", "operation", "outcome"),
            registry=registry,
        )
        self.embedding_duration = Histogram(
            "rag_embedding_duration_seconds",
            "Embedding 耗时",
            ("model", "operation", "outcome"),
            buckets=_RETRIEVAL_BUCKETS,
            registry=registry,
        )
        self.embedding_texts = Counter(
            "rag_embedding_texts_total",
            "Embedding 处理的文本数",
            ("model", "operation"),
            registry=registry,
        )
        self.llm_requests = Counter(
            "rag_llm_requests_total",
            "LLM 流式请求数",
            ("model", "outcome"),
            registry=registry,
        )
        self.llm_stream_duration = Histogram(
            "rag_llm_stream_duration_seconds",
            "LLM 流式生成耗时",
            ("model", "outcome"),
            buckets=_LLM_STREAM_BUCKETS,
            registry=registry,
        )
        self.llm_ttft = Histogram(
            "rag_llm_ttft_seconds",
            "LLM 首 token 时间",
            ("model", "outcome"),
            buckets=_LLM_TTFT_BUCKETS,
            registry=registry,
        )
        self.ingestion_jobs = Counter(
            "rag_ingestion_jobs_total",
            "知识构建任务完成数",
            ("status",),
            registry=registry,
        )
        self.ingestion_duration = Histogram(
            "rag_ingestion_job_duration_seconds",
            "知识构建任务耗时",
            ("status",),
            buckets=_INGESTION_BUCKETS,
            registry=registry,
        )
        self.ingestion_queue_depth = Gauge(
            "rag_ingestion_queue_depth",
            "知识构建任务队列深度",
            ("status",),
            registry=registry,
        )
        for status in ("pending", "running", "succeeded", "failed"):
            self.ingestion_queue_depth.labels(status).set(0)
        self.worker_up = Gauge(
            "rag_worker_up",
            "知识构建 worker 是否运行",
            registry=registry,
        )
        self.worker_up.set(0)
        self.dependency_up = Gauge(
            "rag_dependency_up",
            "外部依赖是否可用",
            ("dependency",),
            registry=registry,
        )
        self.dependency_duration = Histogram(
            "rag_dependency_check_duration_seconds",
            "外部依赖健康检查耗时",
            ("dependency",),
            buckets=_HTTP_BUCKETS,
            registry=registry,
        )
        self.dependency_last_success = Gauge(
            "rag_dependency_last_success_timestamp_seconds",
            "外部依赖最近一次成功时间",
            ("dependency",),
            registry=registry,
        )
        self.acl_denials = Counter(
            "rag_acl_denials_total",
            "权限拒绝次数",
            ("reason",),
            registry=registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def observe_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        method_value = _label(method.upper())
        route_value = _label(route)
        self.http_requests.labels(method_value, route_value, str(status_code)).inc()
        self.http_duration.labels(method_value, route_value).observe(
            max(0.0, duration_seconds)
        )

    def increment_http_in_flight(self, method: str) -> None:
        if self.enabled:
            self.http_in_flight.labels(_label(method.upper())).inc()

    def decrement_http_in_flight(self, method: str) -> None:
        if self.enabled:
            self.http_in_flight.labels(_label(method.upper())).dec()

    def observe_query(
        self,
        *,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        outcome_value = _label(outcome)
        self.query_requests.labels(outcome_value).inc()
        self.query_duration.labels(outcome_value).observe(
            max(0.0, duration_seconds)
        )

    def observe_query_first_token(self, duration_seconds: float) -> None:
        if self.enabled:
            self.query_first_token.observe(max(0.0, duration_seconds))

    def observe_retrieval(
        self,
        *,
        mode: str,
        outcome: str,
        duration_seconds: float,
        result_count: int,
    ) -> None:
        if not self.enabled:
            return
        mode_value = _label(mode)
        outcome_value = _label(outcome)
        self.retrieval_requests.labels(mode_value, outcome_value).inc()
        self.retrieval_duration.labels(mode_value, outcome_value).observe(
            max(0.0, duration_seconds)
        )
        self.retrieval_results.labels(mode_value).observe(max(0, result_count))
        if result_count == 0:
            self.retrieval_empty.labels(mode_value).inc()

    def observe_retrieval_route(
        self,
        *,
        route: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        if self.enabled:
            self.retrieval_route_duration.labels(
                _label(route), _label(outcome)
            ).observe(max(0.0, duration_seconds))

    def observe_retrieval_route_error(self, *, route: str, error_type: str) -> None:
        if self.enabled:
            self.retrieval_route_errors.labels(
                _label(route), _label(error_type)
            ).inc()

    def observe_embedding(
        self,
        *,
        model: str,
        operation: str,
        outcome: str,
        duration_seconds: float,
        text_count: int = 0,
    ) -> None:
        if not self.enabled:
            return
        model_value = _label(model)
        operation_value = _label(operation)
        outcome_value = _label(outcome)
        self.embedding_requests.labels(
            model_value, operation_value, outcome_value
        ).inc()
        self.embedding_duration.labels(
            model_value, operation_value, outcome_value
        ).observe(max(0.0, duration_seconds))
        if text_count > 0:
            self.embedding_texts.labels(model_value, operation_value).inc(text_count)

    def observe_llm_stream(
        self,
        *,
        model: str,
        outcome: str,
        duration_seconds: float,
        ttft_seconds: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        model_value = _label(model)
        outcome_value = _label(outcome)
        self.llm_requests.labels(model_value, outcome_value).inc()
        self.llm_stream_duration.labels(model_value, outcome_value).observe(
            max(0.0, duration_seconds)
        )
        if ttft_seconds is not None:
            self.llm_ttft.labels(model_value, outcome_value).observe(
                max(0.0, ttft_seconds)
            )

    def observe_ingestion_job(self, *, status: str, duration_seconds: float) -> None:
        if not self.enabled:
            return
        status_value = _label(status)
        self.ingestion_jobs.labels(status_value).inc()
        self.ingestion_duration.labels(status_value).observe(
            max(0.0, duration_seconds)
        )

    def set_ingestion_queue_depth(self, *, status: str, count: int) -> None:
        if self.enabled:
            self.ingestion_queue_depth.labels(_label(status)).set(max(0, count))

    def set_worker_running(self, running: bool) -> None:
        if self.enabled:
            self.worker_up.set(1 if running else 0)

    def set_dependency_status(
        self,
        *,
        dependency: str,
        ready: bool,
        latency_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        name = _label(dependency)
        self.dependency_up.labels(name).set(1 if ready else 0)
        self.dependency_duration.labels(name).observe(max(0.0, latency_seconds))
        if ready:
            self.dependency_last_success.labels(name).set(time.time())

    def observe_acl_denial(self, reason: str) -> None:
        if self.enabled:
            self.acl_denials.labels(_label(reason)).inc()


def _label(value: Any) -> str:
    text = str(value).strip()
    return text[:200] if text else "unknown"


__all__ = ["RagMetrics"]
