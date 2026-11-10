from __future__ import annotations

import logging
import os
import socket
import threading
import uuid

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
        logger.info("知识构建 worker 已启动: %s", self.worker_id)
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
                    self._wake_event.wait(self.poll_seconds)
                    self._wake_event.clear()
                    continue

                logger.info(
                    "领取知识构建任务: job=%s document=%s attempt=%s",
                    job.job_id,
                    job.document_id,
                    job.attempt_count,
                )
                try:
                    self.ingestion.run_job(
                        job.job_id,
                        worker_id=self.worker_id,
                        lease_seconds=self.lease_seconds,
                    )
                except Exception:
                    logger.exception("知识构建任务执行失败: %s", job.job_id)
        finally:
            logger.info("知识构建 worker 已停止: %s", self.worker_id)


__all__ = ["IngestionJobWorker"]
