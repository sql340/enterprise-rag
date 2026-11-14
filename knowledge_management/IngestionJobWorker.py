from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from time import perf_counter

from observability import RagMetrics

from .KnowledgeIngestionService import KnowledgeIngestionService


logger = logging.getLogger(__name__)


class IngestionJobWorker:
    """轮询 SQLite 任务表，通过租约在重启和多实例之间恢复构建任务。"""

    def __init__(
        self,
        ingestion: KnowledgeIngestionService,
        *,
        worker_id: str | None = None,
        poll_seconds: float = 1.0,
        lease_seconds: int = 1800,
        max_attempts: int = 3,
        metrics: RagMetrics | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds 必须为正数")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须为正整数")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须为正整数")
        self.ingestion = ingestion
        self.worker_id = worker_id or self._default_worker_id()
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.metrics = metrics
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    @staticmethod
    def _default_worker_id() -> str:
        return (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        with self._start_lock:
            if self.running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"rag-ingestion-worker-{self.worker_id}",
                daemon=True,
            )
            self._thread.start()

    def notify(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join()
        self._thread = None

    def _run(self) -> None:
        if self.metrics is not None:
            self.metrics.set_worker_running(True)
        logger.info(
            "知识构建 worker 已启动: %s",
            self.worker_id,
            extra={"event": "ingestion.worker.start", "worker_id": self.worker_id},
        )
        self._refresh_queue_depth()
        try:
            while not self._stop_event.is_set():
                try:
                    job = self.ingestion.governance_store.claim_next_job(
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                        max_attempts=self.max_attempts,
                    )
                except Exception:
                    logger.exception("领取知识构建任务失败")
                    job = None

                if job is None:
                    self._refresh_queue_depth()
                    self._wake_event.wait(self.poll_seconds)
                    self._wake_event.clear()
                    continue

                logger.info(
                    "领取知识构建任务: job=%s document=%s attempt=%s",
                    job.job_id,
                    job.document_id,
                    job.attempt_count,
                    extra={
                        "event": "ingestion.job.claim",
                        "job_id": job.job_id,
                        "document_id": job.document_id,
                        "attempt": job.attempt_count,
                        "worker_id": self.worker_id,
                    },
                )
                started_at = perf_counter()
                try:
                    completed = self.ingestion.run_job(
                        job.job_id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                except Exception:
                    self._observe_job("failed", started_at)
                    logger.exception("知识构建任务执行失败: %s", job.job_id)
                else:
                    status = str(getattr(completed, "status", "unknown"))
                    if status == "running":
                        status = "lease_lost"
                    self._observe_job(status, started_at)
                finally:
                    self._refresh_queue_depth()
        finally:
            if self.metrics is not None:
                self.metrics.set_worker_running(False)
            logger.info(
                "知识构建 worker 已停止: %s",
                self.worker_id,
                extra={
                    "event": "ingestion.worker.stop",
                    "worker_id": self.worker_id,
                },
            )

    def _observe_job(self, status: str, started_at: float) -> None:
        if self.metrics is not None:
            self.metrics.observe_ingestion_job(
                status=status,
                duration_seconds=perf_counter() - started_at,
            )

    def _refresh_queue_depth(self) -> None:
        if self.metrics is None or not self.metrics.enabled:
            return
        store = getattr(self.ingestion, "governance_store", None)
        counter = getattr(store, "count_jobs_by_status", None)
        if not callable(counter):
            return
        try:
            counts = counter()
        except Exception:
            return
        for status in ("pending", "running", "succeeded", "failed"):
            self.metrics.set_ingestion_queue_depth(
                status=status,
                count=int(counts.get(status, 0)),
            )


__all__ = ["IngestionJobWorker"]
