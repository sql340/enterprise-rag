from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response


router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    registry = getattr(request.app.state, "metrics", None)
    if registry is None or not registry.enabled:
        raise HTTPException(status_code=404, detail="metrics 未启用")
    return Response(content=registry.render(), media_type=registry.content_type)


__all__ = ["router"]
