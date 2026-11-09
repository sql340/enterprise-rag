from __future__ import annotations

from dataclasses import replace
from typing import Any

from access_control import AccessControlError, Classification, DocumentAcl, Principal

from .AuditEvent import AuditEvent
from .IngestionJob import IngestionJob
from .KnowledgeDocument import KnowledgeDocument
from .KnowledgeIngestionService import KnowledgeIngestionService


class KnowledgeAdminService:
    """知识管理 API 的授权边界与业务适配层。"""

    def __init__(self, ingestion: KnowledgeIngestionService) -> None:
        self.ingestion = ingestion

    def require_admin(self, principal: Principal) -> None:
        if not principal.is_admin:
            raise AccessControlError("需要 knowledge_admin 或 platform_admin 角色")

    def upload_document(
        self,
        *,
        principal: Principal,
        file_name: str,
        content: bytes,
        acl: DocumentAcl,
        chunk_strategy: str,
        chunk_options: dict[str, Any],
    ) -> KnowledgeDocument:
        self.require_admin(principal)
        self._ensure_acl_within_clearance(principal, acl)
        if acl.tenant_id != principal.tenant_id:
            raise AccessControlError("不能为其它租户上传知识")
        return self.ingestion.upload_document(
            tenant_id=principal.tenant_id,
            actor_id=principal.user_id,
            file_name=file_name,
            content=content,
            acl=acl,
            chunk_strategy=chunk_strategy,
            chunk_options=chunk_options,
        )

    def list_documents(
        self,
        *,
        principal: Principal,
        status: str | None = None,
        quality_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeDocument]:
        self.require_admin(principal)
        visible: list[KnowledgeDocument] = []
        matched = 0
        cursor = 0
        page_size = min(500, max(limit, 1))
        while len(visible) < limit:
            page = self.ingestion.list_documents(
                tenant_id=principal.tenant_id,
                status=status,
                quality_status=quality_status,
                limit=page_size,
                offset=cursor,
            )
            if not page:
                break
            for item in page:
                if not self._can_view(principal, item):
                    continue
                if matched < offset:
                    matched += 1
                    continue
                visible.append(item)
                matched += 1
                if len(visible) >= limit:
                    break
            cursor += len(page)
            if len(page) < page_size:
                break
        return visible

    def get_document(
        self,
        *,
        principal: Principal,
        document_id: str,
    ) -> KnowledgeDocument:
        self.require_admin(principal)
        document = self.ingestion.get_document(
            tenant_id=principal.tenant_id,
            document_id=document_id,
        )
        self._ensure_can_view(principal, document)
        return document

    def preview_document(
        self,
        *,
        principal: Principal,
        document_id: str,
    ) -> dict[str, Any]:
        self.get_document(principal=principal, document_id=document_id)
        return self.ingestion.preview_document(
            tenant_id=principal.tenant_id,
            document_id=document_id,
        )

    def create_job(
        self,
        *,
        principal: Principal,
        document_id: str,
        force_reindex: bool = False,
    ) -> IngestionJob:
        self.get_document(principal=principal, document_id=document_id)
        return self.ingestion.create_job(
            tenant_id=principal.tenant_id,
            document_id=document_id,
            actor_id=principal.user_id,
            force_reindex=force_reindex,
        )

    def get_job(self, *, principal: Principal, job_id: str) -> IngestionJob:
        self.require_admin(principal)
        job = self.ingestion.get_job(tenant_id=principal.tenant_id, job_id=job_id)
        document = self.ingestion.get_document(
            tenant_id=principal.tenant_id,
            document_id=job.document_id,
        )
        self._ensure_can_view(principal, document)
        return job

    def list_jobs(
        self,
        *,
        principal: Principal,
        document_id: str | None = None,
        limit: int = 100,
    ) -> list[IngestionJob]:
        self.require_admin(principal)
        accessible = self._visible_document_ids(principal)
        jobs = self.ingestion.list_jobs(
            tenant_id=principal.tenant_id,
            document_id=document_id,
            limit=limit,
        )
        return [job for job in jobs if job.document_id in accessible]

    def retry_job(self, *, principal: Principal, job_id: str) -> IngestionJob:
        self.get_job(principal=principal, job_id=job_id)
        return self.ingestion.retry_job(
            tenant_id=principal.tenant_id,
            job_id=job_id,
            actor_id=principal.user_id,
        )

    def update_acl(
        self,
        *,
        principal: Principal,
        document_id: str,
        acl: DocumentAcl,
    ) -> KnowledgeDocument:
        current = self.get_document(principal=principal, document_id=document_id)
        if current.status in {"archived", "deleted"}:
            raise ValueError("归档或删除的文档不能修改 ACL")
        self._ensure_acl_within_clearance(principal, acl)
        if acl.tenant_id != current.tenant_id:
            raise AccessControlError("ACL tenant_id 与文档租户不一致")
        current_acl = DocumentAcl.from_payload(current.acl)
        if acl.acl_version <= current_acl.acl_version:
            acl = replace(acl, acl_version=current_acl.acl_version + 1)
        return self.ingestion.update_acl(
            tenant_id=principal.tenant_id,
            document_id=document_id,
            acl=acl,
            actor_id=principal.user_id,
        )

    def update_quality(
        self,
        *,
        principal: Principal,
        document_id: str,
        quality_status: str,
        note: str,
    ) -> KnowledgeDocument:
        self.get_document(principal=principal, document_id=document_id)
        return self.ingestion.set_quality_status(
            tenant_id=principal.tenant_id,
            document_id=document_id,
            quality_status=quality_status,
            note=note,
            actor_id=principal.user_id,
        )

    def update_chunk_config(
        self,
        *,
        principal: Principal,
        document_id: str,
        chunk_strategy: str,
        chunk_options: dict[str, Any],
    ) -> KnowledgeDocument:
        self.get_document(principal=principal, document_id=document_id)
        return self.ingestion.update_chunk_config(
            tenant_id=principal.tenant_id,
            document_id=document_id,
            chunk_strategy=chunk_strategy,
            chunk_options=chunk_options,
            actor_id=principal.user_id,
        )

    def rebuild_by_quality(
        self,
        *,
        principal: Principal,
        quality_status: str,
    ) -> tuple[list[IngestionJob], list[dict[str, str]]]:
        visible_ids = list(self._visible_document_ids(principal))
        return self.ingestion.rebuild_documents_by_quality(
            tenant_id=principal.tenant_id,
            quality_status=quality_status,
            actor_id=principal.user_id,
            document_ids=visible_ids,
        )

    def archive_document(
        self,
        *,
        principal: Principal,
        document_id: str,
        delete_vectors: bool = False,
    ) -> KnowledgeDocument:
        self.get_document(principal=principal, document_id=document_id)
        return self.ingestion.archive_document(
            tenant_id=principal.tenant_id,
            document_id=document_id,
            actor_id=principal.user_id,
            delete_vectors=delete_vectors,
        )

    def list_audit_events(
        self,
        *,
        principal: Principal,
        limit: int = 200,
    ) -> list[AuditEvent]:
        self.require_admin(principal)
        visible_documents = self._visible_document_ids(principal)
        visible_jobs: dict[str, str] = {
            job.job_id: job.document_id
            for job in self.list_jobs(principal=principal, limit=500)
        }
        events = self.ingestion.list_audit_events(
            tenant_id=principal.tenant_id,
            limit=min(limit * 4, 1000),
        )
        filtered: list[AuditEvent] = []
        for event in events:
            if event.target_type == "document" and event.target_id in visible_documents:
                filtered.append(event)
            elif (
                event.target_type == "job"
                and visible_jobs.get(event.target_id) in visible_documents
            ):
                filtered.append(event)
            if len(filtered) >= limit:
                break
        return filtered

    @staticmethod
    def supported_file_types() -> dict[str, list[str]]:
        from data_processing.data_reader import DataReader

        reader = DataReader()
        return {
            "extensions": list(reader.supported_extensions),
            "file_types": list(reader.supported_file_types),
            "chunk_strategies": [
                "fixed",
                "recursive",
                "sentence_window",
                "paragraph",
                "markdown_heading",
                "parent_child",
            ],
        }

    def _ensure_can_view(
        self,
        principal: Principal,
        document: KnowledgeDocument,
    ) -> None:
        if document.tenant_id != principal.tenant_id:
            raise AccessControlError("不能访问其它租户的文档")
        acl = DocumentAcl.from_payload(document.acl)
        if int(acl.classification) > int(principal.clearance):
            raise AccessControlError("当前用户密级不足，不能访问该文档")

    def _can_view(self, principal: Principal, document: KnowledgeDocument) -> bool:
        try:
            self._ensure_can_view(principal, document)
        except (AccessControlError, ValueError):
            return False
        return True

    def _visible_document_ids(self, principal: Principal) -> set[str]:
        values: set[str] = set()
        cursor = 0
        while True:
            page = self.ingestion.list_documents(
                tenant_id=principal.tenant_id,
                limit=500,
                offset=cursor,
            )
            if not page:
                return values
            values.update(
                item.document_id for item in page if self._can_view(principal, item)
            )
            cursor += len(page)
            if len(page) < 500:
                return values

    @staticmethod
    def _ensure_acl_within_clearance(
        principal: Principal,
        acl: DocumentAcl,
    ) -> None:
        if int(acl.classification) > int(principal.clearance):
            raise AccessControlError("不能授予高于自身密级的文档权限")


__all__ = ["KnowledgeAdminService"]
