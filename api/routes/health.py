from __future__ import annotations

from fastapi import APIRouter, Request


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


__all__ = ["router"]
