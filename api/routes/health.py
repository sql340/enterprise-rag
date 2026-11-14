from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["health"])


@router.get("/health", summary="服务存活检查")
async def health(request: Request) -> dict[str, str | None]:
    settings = getattr(request.app.state, "settings", None)
    service = getattr(request.app.state, "query_service", None)
    vector_store = getattr(getattr(service, "retriever", None), "vector_store", None)
    collection = getattr(vector_store, "collection_name", None)
    if collection is None:
        collection = getattr(settings, "qdrant_collection", None)
    return {"status": "ok", "collection": collection}


@router.get("/ready", summary="服务就绪检查")
async def ready(request: Request) -> JSONResponse:
    """检查检索索引、后台 worker、管理服务和外部依赖。"""

    state = request.app.state
    settings = getattr(state, "settings", None)
    service = getattr(state, "query_service", None)
    vector_store = getattr(getattr(service, "retriever", None), "vector_store", None)
    collection = getattr(vector_store, "collection_name", None)
    if collection is None:
        collection = getattr(settings, "qdrant_collection", None)

    search_status = getattr(state, "search_index_status", None)
    status_check = getattr(vector_store, "search_index_status", None)
    if callable(status_check):
        try:
            search_status = await asyncio.to_thread(status_check)
            state.search_index_status = search_status
        except Exception as exc:
            search_status = {
                "ready": False,
                "error": "search_index_check_failed",
                "detail": str(exc)[:500],
            }
            state.search_index_status = search_status
    if not isinstance(search_status, dict):
        search_status = {"ready": False, "error": "not_initialized"}
    search_ready = search_status.get("ready") is True

    worker_enabled = bool(getattr(state, "ingestion_worker_enabled", False))
    worker = getattr(state, "job_worker", None)
    worker_running = bool(getattr(worker, "running", False))
    admin_error = getattr(state, "admin_service_error", None)
    admin_ready = (
        getattr(state, "admin_service", None) is not None
        and admin_error is None
    )
    worker_ready = (not worker_enabled) or (worker_running and admin_ready)
    checker = getattr(state, "dependency_health_checker", None)
    dependencies: dict[str, dict] = {}
    if checker is not None:
        try:
            dependencies = await asyncio.to_thread(checker.check)
        except Exception as exc:
            dependencies = {
                "health_checker": {
                    "ready": False,
                    "detail": "health_check_failed",
                    "error": str(exc)[:500],
                }
            }
    dependencies_ready = all(
        bool(item.get("ready")) for item in dependencies.values()
    )
    dependency_required = bool(
        getattr(settings, "rag_dependency_health_fail_readiness", False)
    )
    ready_status = (
        search_ready
        and worker_ready
        and (dependencies_ready or not dependency_required)
    )

    if not ready_status:
        status_text = "not_ready"
    elif not dependencies_ready:
        status_text = "degraded"
    else:
        status_text = "ready"
    return JSONResponse(
        status_code=200 if ready_status else 503,
        content={
            "status": status_text,
            "collection": collection,
            "search_indexes": search_status,
            "ingestion_worker": {
                "enabled": worker_enabled,
                "running": worker_running,
                "ready": worker_ready,
            },
            "admin_service_ready": admin_ready,
            "dependencies": dependencies,
            "dependency_fail_readiness": dependency_required,
        },
    )


__all__ = ["router"]
