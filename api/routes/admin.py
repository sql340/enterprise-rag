from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from access_control import AccessControlError, Principal
from api.AdminAccessController import AdminAccessController
from api.AdminAclRequest import AdminAclRequest
from api.AdminChunkConfigRequest import AdminChunkConfigRequest
from api.AdminJobRequest import AdminJobRequest
from api.AdminQualityRequest import AdminQualityRequest
from api.AdminRebuildRequest import AdminRebuildRequest
from knowledge_management import KnowledgeAdminService


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/admin", tags=["knowledge-admin"])


async def _admin_service(request: Request) -> KnowledgeAdminService:
    value = getattr(request.app.state, "admin_service", None)
    if value is not None:
        return value
    lock = getattr(request.app.state, "admin_service_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.admin_service_lock = lock
    async with lock:
        value = getattr(request.app.state, "admin_service", None)
        if value is not None:
            return value
        error = getattr(request.app.state, "admin_service_error", None)
        if error is not None:
            raise HTTPException(status_code=503, detail="知识管理服务初始化失败") from error
        factory = getattr(request.app.state, "admin_service_factory", None)
        if factory is None:
            raise HTTPException(status_code=503, detail="知识管理服务尚未配置")
        try:
            value = await asyncio.to_thread(factory)
        except Exception as exc:
            logger.exception("知识管理服务初始化失败")
            request.app.state.admin_service_error = exc
            raise HTTPException(status_code=503, detail="知识管理服务初始化失败") from exc
        request.app.state.admin_service = value
        return value


def _principal(request: Request) -> Principal:
    settings = getattr(request.app.state, "settings", None)
    controller = getattr(request.app.state, "admin_access_controller", None)
    if controller is None:
        controller = AdminAccessController(
            proxy_token=getattr(settings, "rag_acl_proxy_token", None),
            resolver=getattr(request.app.state, "principal_resolver", None),
        )
    if not controller.proxy_token:
        raise HTTPException(
            status_code=503,
            detail="管理接口要求配置 RAG_ACL_PROXY_TOKEN",
        )
    try:
        principal = controller.resolve(request.headers)
    except AccessControlError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="需要知识管理员角色")
    return principal


def _notify_job_worker(request: Request) -> None:
    worker = getattr(request.app.state, "job_worker", None)
    notify = getattr(worker, "notify", None)
    if callable(notify):
        notify()


@router.get("/capabilities", summary="管理端能力与支持文件类型")
async def capabilities(request: Request) -> dict[str, Any]:
    _principal(request)
    return KnowledgeAdminService.supported_file_types()


@router.get("/me", summary="当前管理身份")
async def me(request: Request) -> dict[str, Any]:
    principal = _principal(request)
    return {
        "tenant_id": principal.tenant_id,
        "user_id": principal.user_id,
        "roles": sorted(principal.roles),
        "groups": sorted(principal.groups),
        "clearance": principal.clearance.name.lower(),
        "display_name": principal.display_name,
    }


@router.post("/documents/upload", status_code=201, summary="上传知识文件")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_options: str = Form("{}"),
    classification: str = Form("internal"),
    allowed_roles: str = Form(""),
    allowed_groups: str = Form(""),
    allowed_users: str = Form(""),
    owner_id: str | None = Form(None),
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    settings = request.app.state.settings
    limit = int(settings.rag_admin_max_file_bytes)
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail="上传文件超过大小限制")
    try:
        options = _json_object(chunk_options, "chunk_options")
        acl = AdminAclRequest(
            classification=classification,
            allowed_roles=_split_values(allowed_roles),
            allowed_groups=_split_values(allowed_groups),
            allowed_users=_split_values(allowed_users),
            owner_id=owner_id,
        ).to_acl(principal)
        document = await asyncio.to_thread(
            service.upload_document,
            principal=principal,
            file_name=file.filename or "",
            content=content,
            acl=acl,
            chunk_strategy=chunk_strategy,
            chunk_options=options,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return document.to_dict()


@router.get("/documents", summary="列出知识文档")
async def list_documents(
    request: Request,
    status: str | None = None,
    quality_status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        items = await asyncio.to_thread(
            service.list_documents,
            principal=principal,
            status=status,
            quality_status=quality_status,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return {"items": [item.to_dict() for item in items], "count": len(items)}


@router.post("/documents/rebuild", summary="按质量状态批量重建")
async def rebuild_by_quality(
    payload: AdminRebuildRequest,
    request: Request,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        jobs, skipped = await asyncio.to_thread(
            service.rebuild_by_quality,
            principal=principal,
            quality_status=payload.quality_status,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    if jobs:
        _notify_job_worker(request)
    return {"jobs": [job.to_dict() for job in jobs], "skipped": skipped}


@router.get("/documents/{document_id}", summary="查看文档")
async def get_document(request: Request, document_id: str) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        document = await asyncio.to_thread(
            service.get_document,
            principal=principal,
            document_id=document_id,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return document.to_dict()


@router.post("/documents/{document_id}/preview", summary="解析与分块预览")
async def preview_document(request: Request, document_id: str) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        return await asyncio.to_thread(
            service.preview_document,
            principal=principal,
            document_id=document_id,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc


@router.patch("/documents/{document_id}/acl", summary="更新文档权限")
async def update_acl(
    payload: AdminAclRequest,
    request: Request,
    document_id: str,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        document = await asyncio.to_thread(
            service.update_acl,
            principal=principal,
            document_id=document_id,
            acl=payload.to_acl(principal),
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return document.to_dict()


@router.patch("/documents/{document_id}/quality", summary="更新文档质量状态")
async def update_quality(
    payload: AdminQualityRequest,
    request: Request,
    document_id: str,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        document = await asyncio.to_thread(
            service.update_quality,
            principal=principal,
            document_id=document_id,
            quality_status=payload.quality_status,
            note=payload.note,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return document.to_dict()


@router.patch("/documents/{document_id}/config", summary="更新文档分块配置")
async def update_chunk_config(
    payload: AdminChunkConfigRequest,
    request: Request,
    document_id: str,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        document = await asyncio.to_thread(
            service.update_chunk_config,
            principal=principal,
            document_id=document_id,
            chunk_strategy=payload.chunk_strategy,
            chunk_options=payload.chunk_options,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return document.to_dict()


@router.delete("/documents/{document_id}", summary="归档或删除文档")
async def delete_document(
    request: Request,
    document_id: str,
    delete_vectors: bool = False,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        document = await asyncio.to_thread(
            service.archive_document,
            principal=principal,
            document_id=document_id,
            delete_vectors=delete_vectors,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return document.to_dict()


@router.post("/documents/{document_id}/jobs", status_code=202, summary="创建构建任务")
async def create_job(
    payload: AdminJobRequest,
    request: Request,
    document_id: str,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        job = await asyncio.to_thread(
            service.create_job,
            principal=principal,
            document_id=document_id,
            force_reindex=payload.force_reindex,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    if job.status == "pending":
        _notify_job_worker(request)
    return job.to_dict()


@router.get("/jobs", summary="列出构建任务")
async def list_jobs(
    request: Request,
    document_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        jobs = await asyncio.to_thread(
            service.list_jobs,
            principal=principal,
            document_id=document_id,
            limit=limit,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return {"items": [job.to_dict() for job in jobs], "count": len(jobs)}


@router.get("/jobs/{job_id}", summary="查看构建任务")
async def get_job(request: Request, job_id: str) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        job = await asyncio.to_thread(
            service.get_job,
            principal=principal,
            job_id=job_id,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return job.to_dict()


@router.post("/jobs/{job_id}/retry", status_code=202, summary="重试失败任务")
async def retry_job(
    request: Request,
    job_id: str,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        job = await asyncio.to_thread(
            service.retry_job,
            principal=principal,
            job_id=job_id,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    if job.status == "pending":
        _notify_job_worker(request)
    return job.to_dict()


@router.get("/audit", summary="查看知识管理审计")
async def audit_events(
    request: Request,
    limit: int = 200,
) -> dict[str, Any]:
    principal = _principal(request)
    service = await _admin_service(request)
    try:
        events = await asyncio.to_thread(
            service.list_audit_events,
            principal=principal,
            limit=limit,
        )
    except Exception as exc:
        raise _to_http_error(exc) from exc
    return {"items": [event.to_dict() for event in events], "count": len(events)}


def _json_object(value: str, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} 必须是合法 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} 必须是 JSON 对象")
    return parsed


def _split_values(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _to_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AccessControlError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("知识管理请求失败")
    return HTTPException(status_code=500, detail="知识管理请求处理失败")


__all__ = ["router"]
