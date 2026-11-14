from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .AuditEvent import AuditEvent
from .IngestionJob import IngestionJob
from .KnowledgeDocument import KnowledgeDocument


class KnowledgeGovernanceStore:
    """SQLite 知识治理存储：文档清单、任务、ACL 与审计。"""

    def __init__(self, path: str | Path = ".cache/knowledge_governance.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    acl_json TEXT NOT NULL,
                    chunk_strategy TEXT NOT NULL,
                    chunk_options_json TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    embedding_fingerprint TEXT NOT NULL,
                    build_fingerprint TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    quality_note TEXT NOT NULL,
                    indexed_document_ids_json TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (tenant_id, source)
                );

                CREATE TABLE IF NOT EXISTS knowledge_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    error TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT NOT NULL DEFAULT '',
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES knowledge_documents(document_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS knowledge_audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_tenant
                    ON knowledge_documents(tenant_id, status, quality_status);
                CREATE INDEX IF NOT EXISTS idx_knowledge_jobs_tenant
                    ON knowledge_jobs(tenant_id, document_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_knowledge_jobs_status
                    ON knowledge_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_knowledge_audit_tenant
                    ON knowledge_audit_events(tenant_id, created_at);
                """
            )
            self._ensure_column(
                "knowledge_documents",
                "build_fingerprint",
                "TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                "knowledge_jobs", "attempt_count", "INTEGER NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                "knowledge_jobs", "worker_id", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column("knowledge_jobs", "lease_expires_at", "TEXT")
            self._ensure_column("knowledge_jobs", "heartbeat_at", "TEXT")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def save_document(self, document: KnowledgeDocument) -> None:
        document.touch()
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT tenant_id FROM knowledge_documents WHERE document_id = ?",
                (document.document_id,),
            ).fetchone()
            if existing is not None and existing["tenant_id"] != document.tenant_id:
                raise ValueError("document_id 已属于其它租户")
            self._connection.execute(
                """
                INSERT INTO knowledge_documents (
                    document_id, tenant_id, source, file_name, file_type,
                    storage_path, content_hash, acl_json, chunk_strategy,
                    chunk_options_json, parser_version, embedding_fingerprint,
                    build_fingerprint, status, quality_status, quality_note,
                    indexed_document_ids_json, chunk_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source = excluded.source,
                    file_name = excluded.file_name,
                    file_type = excluded.file_type,
                    storage_path = excluded.storage_path,
                    content_hash = excluded.content_hash,
                    acl_json = excluded.acl_json,
                    chunk_strategy = excluded.chunk_strategy,
                    chunk_options_json = excluded.chunk_options_json,
                    parser_version = excluded.parser_version,
                    embedding_fingerprint = excluded.embedding_fingerprint,
                    build_fingerprint = excluded.build_fingerprint,
                    status = excluded.status,
                    quality_status = excluded.quality_status,
                    quality_note = excluded.quality_note,
                    indexed_document_ids_json = excluded.indexed_document_ids_json,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                (
                    document.document_id,
                    document.tenant_id,
                    document.source,
                    document.file_name,
                    document.file_type,
                    document.storage_path,
                    document.content_hash,
                    _json(document.acl),
                    document.chunk_strategy,
                    _json(document.chunk_options),
                    document.parser_version,
                    document.embedding_fingerprint,
                    document.build_fingerprint,
                    document.status,
                    document.quality_status,
                    document.quality_note,
                    _json(document.indexed_document_ids),
                    document.chunk_count,
                    document.created_at,
                    document.updated_at,
                ),
            )

    def get_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> KnowledgeDocument | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM knowledge_documents
                WHERE tenant_id = ? AND document_id = ?
                """,
                (tenant_id, document_id),
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(
        self,
        *,
        tenant_id: str,
        status: str | None = None,
        quality_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeDocument]:
        if limit <= 0 or limit > 500:
            raise ValueError("limit 必须在 1 到 500 之间")
        if offset < 0:
            raise ValueError("offset 不能为负数")
        conditions = ["tenant_id = ?"]
        values: list[Any] = [tenant_id]
        if status:
            conditions.append("status = ?")
            values.append(status)
        if quality_status:
            conditions.append("quality_status = ?")
            values.append(quality_status)
        values.extend([limit, offset])
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM knowledge_documents
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC, document_id
                LIMIT ? OFFSET ?
                """,
                values,
            ).fetchall()
        return [_row_to_document(row) for row in rows]

    def create_job(self, job: IngestionJob) -> None:
        document = self.get_document(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
        )
        if document is None:
            raise ValueError("任务关联的文档不存在或不属于当前租户")
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO knowledge_jobs (
                    job_id, tenant_id, document_id, status, progress,
                    message, error, result_json, attempt_count, worker_id,
                    lease_expires_at, heartbeat_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.tenant_id,
                    job.document_id,
                    job.status,
                    job.progress,
                    job.message,
                    job.error,
                    _json(job.result),
                    job.attempt_count,
                    job.worker_id,
                    job.lease_expires_at,
                    job.heartbeat_at,
                    job.created_at,
                    job.updated_at,
                ),
            )

    def save_job(self, job: IngestionJob) -> None:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT tenant_id FROM knowledge_jobs WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("任务不存在")
            if row["tenant_id"] != job.tenant_id:
                raise ValueError("任务租户不一致")
            self._connection.execute(
                """
                UPDATE knowledge_jobs SET
                    status = ?, progress = ?, message = ?, error = ?,
                    result_json = ?, attempt_count = ?, worker_id = ?,
                    lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE job_id = ? AND tenant_id = ?
                """,
                (
                    job.status,
                    job.progress,
                    job.message,
                    job.error,
                    _json(job.result),
                    job.attempt_count,
                    job.worker_id,
                    job.lease_expires_at,
                    job.heartbeat_at,
                    job.updated_at,
                    job.job_id,
                    job.tenant_id,
                ),
            )

    def claim_next_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> IngestionJob | None:
        """原子领取一个待执行任务或租约已过期的任务。"""

        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("worker_id 不能为空")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须为正整数")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须为正整数")

        now = datetime.now(timezone.utc)
        now_value = now.isoformat()
        lease_value = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                exhausted = connection.execute(
                    """
                    SELECT job_id, document_id FROM knowledge_jobs
                    WHERE status = 'running'
                      AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                      AND attempt_count >= ?
                    """,
                    (now_value, max_attempts),
                ).fetchall()
                for row in exhausted:
                    connection.execute(
                        """
                        UPDATE knowledge_documents
                        SET status = 'failed', updated_at = ?
                        WHERE document_id = ?
                        """,
                        (now_value, row["document_id"]),
                    )
                    connection.execute(
                        """
                        UPDATE knowledge_jobs
                        SET status = 'failed', progress = 100,
                            message = 'failed', error = ?,
                            lease_expires_at = NULL, updated_at = ?
                        WHERE job_id = ?
                        """,
                        (
                            "任务租约过期且已达到最大重试次数",
                            now_value,
                            row["job_id"],
                        ),
                    )

                row = connection.execute(
                    """
                    SELECT * FROM knowledge_jobs
                    WHERE status = 'pending'
                       OR (
                           status = 'running'
                           AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                       )
                    ORDER BY created_at, job_id
                    LIMIT 1
                    """,
                    (now_value,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None

                connection.execute(
                    """
                    UPDATE knowledge_jobs
                    SET status = 'running', progress = 5, message = 'claimed',
                        error = '', worker_id = ?, lease_expires_at = ?,
                        heartbeat_at = ?, attempt_count = attempt_count + 1,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        worker,
                        lease_value,
                        now_value,
                        now_value,
                        row["job_id"],
                    ),
                )
                claimed = connection.execute(
                    "SELECT * FROM knowledge_jobs WHERE job_id = ?",
                    (row["job_id"],),
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return _row_to_job(claimed) if claimed else None

    def heartbeat_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> bool:
        """续租；任务已被其它 worker 接管时返回 False。"""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds 必须为正整数")
        now = datetime.now(timezone.utc)
        now_value = now.isoformat()
        lease_value = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE knowledge_jobs
                SET lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE job_id = ? AND worker_id = ? AND status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                """,
                (lease_value, now_value, now_value, job_id, worker_id, now_value),
            )
        return cursor.rowcount == 1

    def get_job(
        self,
        *,
        tenant_id: str,
        job_id: str,
    ) -> IngestionJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM knowledge_jobs WHERE tenant_id = ? AND job_id = ?",
                (tenant_id, job_id),
            ).fetchone()
        return _row_to_job(row) if row else None

    def get_job_internal(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM knowledge_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(
        self,
        *,
        tenant_id: str,
        document_id: str | None = None,
        limit: int = 100,
    ) -> list[IngestionJob]:
        if limit <= 0 or limit > 500:
            raise ValueError("limit 必须在 1 到 500 之间")
        conditions = ["tenant_id = ?"]
        values: list[Any] = [tenant_id]
        if document_id:
            conditions.append("document_id = ?")
            values.append(document_id)
        values.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM knowledge_jobs
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC, job_id
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def count_jobs_by_status(self) -> dict[str, int]:
        """返回全局任务数量，用于 worker 队列指标。"""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM knowledge_jobs
                GROUP BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def record_audit(self, event: AuditEvent) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO knowledge_audit_events (
                    tenant_id, actor_id, action, target_type, target_id,
                    detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.tenant_id,
                    event.actor_id,
                    event.action,
                    event.target_type,
                    event.target_id,
                    _json(event.detail),
                    event.created_at,
                ),
            )
        return int(cursor.lastrowid)

    def list_audit_events(
        self,
        *,
        tenant_id: str,
        limit: int = 200,
    ) -> list[AuditEvent]:
        if limit <= 0 or limit > 1000:
            raise ValueError("limit 必须在 1 到 1000 之间")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM knowledge_audit_events
                WHERE tenant_id = ?
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (tenant_id, limit),
            ).fetchall()
        return [_row_to_audit(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None  # type: ignore[assignment]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(value: str) -> Any:
    return json.loads(value) if value else None


def _row_to_document(row: sqlite3.Row) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=str(row["document_id"]),
        tenant_id=str(row["tenant_id"]),
        source=str(row["source"]),
        file_name=str(row["file_name"]),
        file_type=str(row["file_type"]),
        storage_path=str(row["storage_path"]),
        content_hash=str(row["content_hash"]),
        acl=dict(_load_json(row["acl_json"]) or {}),
        chunk_strategy=str(row["chunk_strategy"]),
        chunk_options=dict(_load_json(row["chunk_options_json"]) or {}),
        parser_version=str(row["parser_version"]),
        embedding_fingerprint=str(row["embedding_fingerprint"]),
        build_fingerprint=str(row["build_fingerprint"]),
        status=str(row["status"]),
        quality_status=str(row["quality_status"]),
        quality_note=str(row["quality_note"]),
        indexed_document_ids=tuple(_load_json(row["indexed_document_ids_json"]) or ()),
        chunk_count=int(row["chunk_count"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_job(row: sqlite3.Row) -> IngestionJob:
    return IngestionJob(
        job_id=str(row["job_id"]),
        tenant_id=str(row["tenant_id"]),
        document_id=str(row["document_id"]),
        status=str(row["status"]),
        progress=int(row["progress"]),
        message=str(row["message"]),
        error=str(row["error"]),
        result=dict(_load_json(row["result_json"]) or {}),
        attempt_count=int(row["attempt_count"]),
        worker_id=str(row["worker_id"]),
        lease_expires_at=(
            str(row["lease_expires_at"]) if row["lease_expires_at"] else None
        ),
        heartbeat_at=str(row["heartbeat_at"]) if row["heartbeat_at"] else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_audit(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        event_id=int(row["event_id"]),
        tenant_id=str(row["tenant_id"]),
        actor_id=str(row["actor_id"]),
        action=str(row["action"]),
        target_type=str(row["target_type"]),
        target_id=str(row["target_id"]),
        detail=dict(_load_json(row["detail_json"]) or {}),
        created_at=str(row["created_at"]),
    )


__all__ = ["KnowledgeGovernanceStore"]
