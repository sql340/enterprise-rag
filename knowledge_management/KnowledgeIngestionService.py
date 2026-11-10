from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from access_control import DocumentAcl
from data_processing.chunker import Chunker, TenantChunkScope
from data_processing.data_reader import DataReader, Document
from data_processing.knowledge_indexing import KnowledgeIndexer
from graph_rag import GraphIndexer

from .AuditEvent import AuditEvent
from .ChunkQualityAnalyzer import ChunkQualityAnalyzer
from .IngestionJob import IngestionJob
from .IngestionJobLeaseError import IngestionJobLeaseError
from .KnowledgeBuildFingerprint import KnowledgeBuildFingerprint
from .KnowledgeDocument import KnowledgeDocument
from .KnowledgeGovernanceStore import KnowledgeGovernanceStore


logger = logging.getLogger(__name__)


class KnowledgeIngestionService:
    """知识上传、解析、分块、向量化、图谱构建和治理操作。"""

    def __init__(
        self,
        *,
        governance_store: KnowledgeGovernanceStore,
        data_dir: str | Path,
        indexer: KnowledgeIndexer,
        graph_indexer: GraphIndexer | None = None,
        reader: DataReader | None = None,
        parser_version: str = "v1",
        embedding_fingerprint: str = "",
        max_file_bytes: int = 50 * 1024 * 1024,
        quality_analyzer: ChunkQualityAnalyzer | None = None,
    ) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes 必须为正整数")
        self.governance_store = governance_store
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.indexer = indexer
        self.graph_indexer = graph_indexer
        self.reader = reader or DataReader()
        self.parser_version = parser_version
        self.embedding_fingerprint = (
            embedding_fingerprint.strip() or _embedding_fingerprint(indexer)
        )
        self.max_file_bytes = max_file_bytes
        self.quality_analyzer = quality_analyzer or ChunkQualityAnalyzer()

    def upload_document(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        file_name: str,
        content: bytes,
        acl: DocumentAcl,
        chunk_strategy: str = "recursive",
        chunk_options: dict[str, Any] | None = None,
    ) -> KnowledgeDocument:
        tenant = str(tenant_id).strip()
        if acl.tenant_id != tenant:
            raise ValueError("上传租户与 ACL tenant_id 不一致")
        if not content:
            raise ValueError("上传文件不能为空")
        if len(content) > self.max_file_bytes:
            raise ValueError("上传文件超过大小限制")
        safe_name = Path(str(file_name)).name.strip()
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("文件名无效")
        suffix = Path(safe_name).suffix.lower()
        if not suffix or not self.reader.supports(suffix):
            raise ValueError(f"不支持的文件类型: {suffix or '无扩展名'}")
        self._validate_chunk_config(chunk_strategy, chunk_options or {})

        document_id = uuid.uuid4().hex
        tenant_dir = self.data_dir / _safe_segment(tenant)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        storage_path = tenant_dir / f"{document_id}{suffix}"
        temporary = storage_path.with_suffix(storage_path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(storage_path)

        document = KnowledgeDocument(
            document_id=document_id,
            tenant_id=tenant,
            source=f"upload://{tenant}/{document_id}/{safe_name}",
            file_name=safe_name,
            file_type=suffix.lstrip("."),
            storage_path=str(storage_path),
            content_hash=hashlib.sha256(content).hexdigest(),
            acl=acl.to_payload(),
            chunk_strategy=chunk_strategy,
            chunk_options=dict(chunk_options or {}),
            parser_version=self.parser_version,
            embedding_fingerprint=self.embedding_fingerprint,
        )
        self.governance_store.save_document(document)
        self._audit(
            tenant,
            actor_id,
            "document.upload",
            "document",
            document_id,
            {"file_name": safe_name, "size_bytes": len(content)},
        )
        return document

    def preview_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        document = self._require_document(tenant_id, document_id)
        documents = self._read_documents(document)
        chunks = self._chunk_documents(document, documents)
        chunks, quality_report = self.quality_analyzer.analyze(chunks)
        lengths = [len(chunk.content) for chunk in chunks]
        samples = [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "length": len(chunk.content),
                "content": chunk.content[:500],
                "metadata": {
                    key: chunk.metadata.get(key)
                    for key in (
                        "source",
                        "page_number",
                        "slide_number",
                        "sheet_name",
                        "heading_path",
                        "chunk_role",
                    )
                    if chunk.metadata.get(key) is not None
                },
            }
            for chunk in chunks[:10]
        ]
        return {
            "document_id": document.document_id,
            "parsed_documents": len(documents),
            "chunk_count": len(chunks),
            "min_chunk_length": min(lengths) if lengths else 0,
            "max_chunk_length": max(lengths) if lengths else 0,
            "samples": samples,
            "quality": quality_report,
        }

    def create_job(
        self,
        *,
        tenant_id: str,
        document_id: str,
        actor_id: str,
        force_reindex: bool = False,
        quality_status: Sequence[str] | None = None,
    ) -> IngestionJob:
        document = self._require_document(tenant_id, document_id)
        if document.status in {"archived", "deleted"}:
            raise ValueError("归档或删除的文档不能重建")
        if document.quality_status == "blocked" and not force_reindex:
            raise ValueError("质量状态为 blocked 的文档必须先解除阻断")
        if quality_status is not None and document.quality_status not in set(
            quality_status
        ):
            raise ValueError("文档质量状态不满足重建筛选条件")
        active = [
            job
            for job in self.governance_store.list_jobs(
                tenant_id=tenant_id,
                document_id=document_id,
            )
            if job.status in {"pending", "running"}
        ]
        if active:
            raise ValueError("该文档已有待执行或运行中的任务")
        build_fingerprint = self._build_fingerprint(document)
        if (
            not force_reindex
            and document.status == "succeeded"
            and document.quality_status == "good"
            and KnowledgeBuildFingerprint.matches(
                document,
                embedding_fingerprint=self.embedding_fingerprint,
                graph_extractor_version=self._graph_extractor_version(),
            )
        ):
            job = IngestionJob(
                job_id=uuid.uuid4().hex,
                tenant_id=document.tenant_id,
                document_id=document.document_id,
                status="succeeded",
                progress=100,
                message="skipped_unchanged",
                result={
                    "skipped": True,
                    "reason": "unchanged_good_document",
                    "build_fingerprint": build_fingerprint,
                    "force_reindex": False,
                },
            )
            self.governance_store.create_job(job)
            self._audit(
                document.tenant_id,
                actor_id,
                "ingestion_job.skip",
                "job",
                job.job_id,
                {
                    "document_id": document.document_id,
                    "reason": "unchanged_good_document",
                },
            )
            return job
        job = IngestionJob(
            job_id=uuid.uuid4().hex,
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            result={
                "force_reindex": bool(force_reindex),
                "build_fingerprint": build_fingerprint,
                "previous_build_fingerprint": document.build_fingerprint,
            },
        )
        self.governance_store.create_job(job)
        document.status = "queued"
        self.governance_store.save_document(document)
        self._audit(
            document.tenant_id,
            actor_id,
            "ingestion_job.create",
            "job",
            job.job_id,
            {"document_id": document.document_id, "force_reindex": force_reindex},
        )
        return job

    def retry_job(
        self,
        *,
        tenant_id: str,
        job_id: str,
        actor_id: str,
    ) -> IngestionJob:
        previous = self.governance_store.get_job(tenant_id=tenant_id, job_id=job_id)
        if previous is None:
            raise ValueError("任务不存在或不属于当前租户")
        if previous.status != "failed":
            raise ValueError("只有失败任务可以重试")
        return self.create_job(
            tenant_id=tenant_id,
            document_id=previous.document_id,
            actor_id=actor_id,
            force_reindex=True,
        )

    def run_job(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 1800,
    ) -> IngestionJob:
        job = self.governance_store.get_job_internal(job_id)
        if job is None:
            raise ValueError("任务不存在")
        if job.status == "succeeded":
            return job
        if job.status == "running" and (
            not worker_id or job.worker_id != worker_id
        ):
            return job
        if job.status not in {"pending", "running", "failed"}:
            return job

        document: KnowledgeDocument | None = None
        try:
            document = self._require_document(job.tenant_id, job.document_id)
            self._heartbeat(job, worker_id, lease_seconds)
            job.status = "running"
            job.progress = 5
            job.message = "started"
            job.updated_at = _now()
            self.governance_store.save_job(job)
            document.status = "running"
            self.governance_store.save_document(document)

            force_reindex = bool(job.result.get("force_reindex"))
            previous_build_fingerprint = str(document.build_fingerprint or "")
            build_fingerprint = self._build_fingerprint(document)
            documents = self._read_documents(document)
            self._heartbeat(job, worker_id, lease_seconds)
            job.progress = 25
            job.message = "parsed"
            self.governance_store.save_job(job)
            chunks = self._chunk_documents(document, documents)
            chunks, quality_report = self.quality_analyzer.analyze(chunks)
            self._heartbeat(job, worker_id, lease_seconds)
            job.progress = 45
            job.message = "chunked"
            job.result = {**job.result, "quality_report": quality_report}
            self.governance_store.save_job(job)

            acl = DocumentAcl.from_payload(document.acl)
            acl_payload = acl.to_payload()
            scoped_chunks = TenantChunkScope.apply(
                [
                    replace(
                        chunk,
                        metadata={**chunk.metadata, **acl_payload},
                    )
                    for chunk in chunks
                ],
                document.tenant_id,
            )
            vector_result = self.indexer.index_chunks(
                scoped_chunks,
                acl=acl,
                tenant_id=document.tenant_id,
                force_reindex=force_reindex,
            )
            self._heartbeat(job, worker_id, lease_seconds)
            job.progress = 75
            job.message = "vectorized"
            self.governance_store.save_job(job)

            graph_result: dict[str, Any] | None = None
            if self.graph_indexer is not None:
                indexed = self.graph_indexer.index_chunks(
                    scoped_chunks,
                    acl=acl,
                    force_reindex=force_reindex,
                )
                self._sync_graph_documents(
                    document=document,
                    chunks=scoped_chunks,
                    new_document_ids=vector_result.document_ids,
                )
                graph_result = indexed.to_dict()
            self._heartbeat(job, worker_id, lease_seconds)
            job.progress = 95
            job.message = "indexed"
            self.governance_store.save_job(job)

            document.status = "succeeded"
            document.indexed_document_ids = vector_result.document_ids
            document.chunk_count = len(scoped_chunks)
            document.embedding_fingerprint = self.embedding_fingerprint
            document.build_fingerprint = build_fingerprint
            content_changed = previous_build_fingerprint != build_fingerprint
            if quality_report.get("needs_rework"):
                document.quality_status = "needs_rework"
                document.quality_note = (
                    "分块质量检查发现低质量块占比过高，请调整解析或分块配置后重建"
                )
            elif content_changed:
                document.quality_status = "unreviewed"
                document.quality_note = "文档内容或构建配置已变更，请复核质量"
            self.governance_store.save_document(document)
            self._heartbeat(job, worker_id, lease_seconds)
            job.status = "succeeded"
            job.progress = 100
            job.message = "completed"
            job.error = ""
            job.lease_expires_at = None
            job.heartbeat_at = _now()
            job.result = {
                "vector": vector_result.to_dict(),
                "graph": graph_result,
                "force_reindex": force_reindex,
                "build_fingerprint": build_fingerprint,
                "quality_report": quality_report,
                "skipped": False,
            }
            job.updated_at = _now()
            self.governance_store.save_job(job)
            self._audit(
                job.tenant_id,
                "system",
                "ingestion_job.succeed",
                "job",
                job.job_id,
                {"document_id": document.document_id, "chunks": len(scoped_chunks)},
            )
            return job
        except IngestionJobLeaseError:
            logger.warning("任务租约已失效，停止写入: %s", job.job_id)
            return job
        except Exception as exc:
            if worker_id is not None:
                try:
                    self._heartbeat(job, worker_id, lease_seconds)
                except IngestionJobLeaseError:
                    logger.warning("任务租约已失效，停止写入失败状态: %s", job.job_id)
                    return job
            if document is not None:
                document.status = "failed"
                self.governance_store.save_document(document)
            job.status = "failed"
            job.progress = 100
            job.message = "failed"
            job.error = str(exc)
            job.lease_expires_at = None
            job.heartbeat_at = _now()
            job.updated_at = _now()
            self.governance_store.save_job(job)
            self._audit(
                job.tenant_id,
                "system",
                "ingestion_job.fail",
                "job",
                job.job_id,
                {"document_id": job.document_id, "error": str(exc)},
            )
            raise

    def _heartbeat(
        self,
        job: IngestionJob,
        worker_id: str | None,
        lease_seconds: int,
    ) -> None:
        if not worker_id:
            return
        if not self.governance_store.heartbeat_job(
            job_id=job.job_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        ):
            raise IngestionJobLeaseError(f"任务 {job.job_id} 的租约已失效")
        now = datetime.now(timezone.utc)
        job.heartbeat_at = now.isoformat()
        job.lease_expires_at = (
            now + timedelta(seconds=lease_seconds)
        ).isoformat()

    def update_acl(
        self,
        *,
        tenant_id: str,
        document_id: str,
        acl: DocumentAcl,
        actor_id: str,
    ) -> KnowledgeDocument:
        document = self._require_document(tenant_id, document_id)
        if acl.tenant_id != document.tenant_id:
            raise ValueError("ACL tenant_id 与文档租户不一致")
        document.acl = acl.to_payload()
        for indexed_document_id in document.indexed_document_ids:
            self.indexer.update_acl_by_document(indexed_document_id, acl)
        self.governance_store.save_document(document)
        self._audit(
            tenant_id,
            actor_id,
            "document.acl.update",
            "document",
            document_id,
            {"acl_version": acl.acl_version},
        )
        return document

    def set_quality_status(
        self,
        *,
        tenant_id: str,
        document_id: str,
        quality_status: str,
        note: str,
        actor_id: str,
    ) -> KnowledgeDocument:
        document = self._require_document(tenant_id, document_id)
        resolved = str(quality_status).strip().lower()
        if resolved not in {"unreviewed", "good", "needs_rework", "blocked"}:
            raise ValueError(
                "quality_status 必须是 unreviewed/good/needs_rework/blocked"
            )
        document.quality_status = resolved
        document.quality_note = str(note).strip()
        self.governance_store.save_document(document)
        self._audit(
            tenant_id,
            actor_id,
            "document.quality.update",
            "document",
            document_id,
            {
                "quality_status": document.quality_status,
                "quality_note": document.quality_note,
            },
        )
        return document

    def update_chunk_config(
        self,
        *,
        tenant_id: str,
        document_id: str,
        chunk_strategy: str,
        chunk_options: dict[str, Any],
        actor_id: str,
    ) -> KnowledgeDocument:
        document = self._require_document(tenant_id, document_id)
        if document.status in {"archived", "deleted"}:
            raise ValueError("归档或删除的文档不能修改分块配置")
        self._validate_chunk_config(chunk_strategy, chunk_options)
        document.chunk_strategy = str(chunk_strategy).strip()
        document.chunk_options = dict(chunk_options or {})
        self.governance_store.save_document(document)
        self._audit(
            tenant_id,
            actor_id,
            "document.chunk_config.update",
            "document",
            document_id,
            {
                "chunk_strategy": document.chunk_strategy,
                "chunk_options": document.chunk_options,
            },
        )
        return document

    def rebuild_documents_by_quality(
        self,
        *,
        tenant_id: str,
        quality_status: str,
        actor_id: str,
        document_ids: Sequence[str] | None = None,
    ) -> tuple[list[IngestionJob], list[dict[str, str]]]:
        resolved = str(quality_status).strip().lower()
        if resolved not in {"unreviewed", "good", "needs_rework", "blocked"}:
            raise ValueError("quality_status 不合法")
        jobs: list[IngestionJob] = []
        skipped: list[dict[str, str]] = []
        allowed = set(document_ids) if document_ids is not None else None
        for document in self.list_documents(tenant_id=tenant_id, limit=500):
            if allowed is not None and document.document_id not in allowed:
                continue
            if document.quality_status != resolved:
                continue
            try:
                jobs.append(
                    self.create_job(
                        tenant_id=tenant_id,
                        document_id=document.document_id,
                        actor_id=actor_id,
                        force_reindex=True,
                    )
                )
            except ValueError as exc:
                skipped.append(
                    {"document_id": document.document_id, "reason": str(exc)}
                )
        return jobs, skipped

    def archive_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        actor_id: str,
        delete_vectors: bool = False,
    ) -> KnowledgeDocument:
        document = self._require_document(tenant_id, document_id)
        acl = DocumentAcl.from_payload(document.acl)
        archived = DocumentAcl(
            tenant_id=acl.tenant_id,
            classification=acl.classification,
            allowed_roles=acl.allowed_roles,
            allowed_groups=acl.allowed_groups,
            allowed_users=acl.allowed_users,
            owner_id=acl.owner_id,
            document_status="deleted" if delete_vectors else "archived",
            acl_version=acl.acl_version + 1,
        )
        document.acl = archived.to_payload()
        for indexed_document_id in document.indexed_document_ids:
            self.indexer.update_acl_by_document(indexed_document_id, archived)
        document.status = "deleted" if delete_vectors else "archived"
        self.governance_store.save_document(document)
        if delete_vectors:
            for indexed_document_id in document.indexed_document_ids:
                self.indexer.vector_store.delete_by_document(
                    tenant_id=tenant_id,
                    document_id=indexed_document_id,
                )
                if self.graph_indexer is not None:
                    self.graph_indexer.delete_by_document(
                        tenant_id=tenant_id,
                        document_id=indexed_document_id,
                    )
        self._audit(
            tenant_id,
            actor_id,
            "document.delete" if delete_vectors else "document.archive",
            "document",
            document_id,
            {"deleted_vectors": delete_vectors},
        )
        return document

    def get_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> KnowledgeDocument:
        return self._require_document(tenant_id, document_id)

    def list_documents(
        self,
        *,
        tenant_id: str,
        status: str | None = None,
        quality_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeDocument]:
        return self.governance_store.list_documents(
            tenant_id=tenant_id,
            status=status,
            quality_status=quality_status,
            limit=limit,
            offset=offset,
        )

    def list_jobs(
        self,
        *,
        tenant_id: str,
        document_id: str | None = None,
        limit: int = 100,
    ) -> list[IngestionJob]:
        return self.governance_store.list_jobs(
            tenant_id=tenant_id,
            document_id=document_id,
            limit=limit,
        )

    def get_job(self, *, tenant_id: str, job_id: str) -> IngestionJob:
        job = self.governance_store.get_job(tenant_id=tenant_id, job_id=job_id)
        if job is None:
            raise ValueError("任务不存在或不属于当前租户")
        return job

    def list_audit_events(self, *, tenant_id: str, limit: int = 200):
        return self.governance_store.list_audit_events(
            tenant_id=tenant_id,
            limit=limit,
        )

    def _require_document(self, tenant_id: str, document_id: str) -> KnowledgeDocument:
        document = self.governance_store.get_document(
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if document is None:
            raise ValueError("文档不存在或不属于当前租户")
        return document

    def _read_documents(self, document: KnowledgeDocument) -> list[Document]:
        values = self.reader.read(document.storage_path)
        if not values:
            raise ValueError("文件解析结果为空")
        final: list[Document] = []
        for value in values:
            metadata = {
                **value.metadata,
                "source": document.source,
                "tenant_id": document.tenant_id,
                "file_name": document.file_name,
                "governance_document_id": document.document_id,
            }
            final.append(Document(content=value.content, metadata=metadata))
        return final

    def _chunk_documents(
        self,
        document: KnowledgeDocument,
        documents: list[Document],
    ):
        return Chunker(
            document.chunk_strategy,
            **document.chunk_options,
        ).split(documents)

    @staticmethod
    def _validate_chunk_config(
        chunk_strategy: str,
        chunk_options: dict[str, Any],
    ) -> None:
        Chunker(str(chunk_strategy), **dict(chunk_options or {}))

    def _build_fingerprint(self, document: KnowledgeDocument) -> str:
        return KnowledgeBuildFingerprint.build(
            document,
            embedding_fingerprint=self.embedding_fingerprint,
            graph_extractor_version=self._graph_extractor_version(),
        )

    def _graph_extractor_version(self) -> str | None:
        if self.graph_indexer is None:
            return None
        extractor = getattr(self.graph_indexer, "extractor", None)
        version = getattr(extractor, "extractor_version", None)
        return str(version) if version else None

    def _sync_graph_documents(
        self,
        *,
        document: KnowledgeDocument,
        chunks,
        new_document_ids: tuple[str, ...],
    ) -> None:
        if self.graph_indexer is None:
            return
        new_ids = set(new_document_ids)
        for old_id in document.indexed_document_ids:
            if old_id not in new_ids:
                self.graph_indexer.delete_by_document(
                    tenant_id=document.tenant_id,
                    document_id=old_id,
                )
        grouped: dict[str, list[str]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.document_id, []).append(chunk.chunk_id)
        for document_id, chunk_ids in grouped.items():
            self.graph_indexer.prune_stale_chunks(
                tenant_id=document.tenant_id,
                document_id=document_id,
                expected_chunk_ids=chunk_ids,
            )

    def _audit(
        self,
        tenant_id: str,
        actor_id: str,
        action: str,
        target_type: str,
        target_id: str,
        detail: dict[str, Any],
    ) -> None:
        self.governance_store.record_audit(
            AuditEvent(
                tenant_id=tenant_id,
                actor_id=actor_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                detail=detail,
            )
        )


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "tenant"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_fingerprint(indexer: KnowledgeIndexer) -> str:
    embedding = getattr(indexer, "embedding", None)
    model_name = str(getattr(embedding, "model_name", "") or "unknown")
    collection = str(getattr(indexer.vector_store, "collection_name", "") or "")
    return f"{type(embedding).__name__}:{model_name}:{collection}"


__all__ = ["KnowledgeIngestionService"]
