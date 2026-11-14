from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from access_control import AccessControlError, Principal, PrincipalResolver
from api.QueryRequest import QueryRequest
from api.sse import encode_sse_event
from observability import RagMetrics
from retrieval import RagQueryService, RagStreamEvent


logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post(
    "/v1/query",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
    summary="流式 RAG 问答",
)
async def query(payload: QueryRequest, request: Request) -> StreamingResponse:
    service = getattr(request.app.state, "query_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="查询服务尚未就绪")
    principal = _resolve_principal(request)

    return StreamingResponse(
        _stream_events(
            service,
            payload,
            request,
            principal,
            getattr(request.app.state, "metrics", None),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_events(
    service: RagQueryService,
    payload: QueryRequest,
    request: Request,
    principal: Principal | None,
    metrics: RagMetrics | None,
) -> AsyncIterator[str]:
    event_id = 0
    started_at = perf_counter()
    first_token_recorded = False
    outcome = "success"
    try:
        async for event in service.stream(
            payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            filters=payload.filters,
            include_sources=payload.include_sources,
            retrieval_mode=payload.retrieval_mode,
            principal=principal,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        ):
            if await request.is_disconnected():
                outcome = "disconnected"
                return
            if event.event == "delta" and not first_token_recorded:
                first_token_recorded = True
                if metrics is not None:
                    metrics.observe_query_first_token(perf_counter() - started_at)
            if event.event == "done":
                finish_reason = event.data.get("finish_reason")
                if finish_reason == "no_context":
                    outcome = "no_context"
            event_id += 1
            yield encode_sse_event(event, event_id)
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except GeneratorExit:
        outcome = "disconnected"
        raise
    except Exception:
        outcome = "error"
        logger.exception("RAG 流式查询失败")
        event_id += 1
        yield encode_sse_event(
            RagStreamEvent(
                "error",
                {
                    "code": "internal_error",
                    "message": "查询处理失败，请稍后重试。",
                },
            ),
            event_id,
        )
    finally:
        if metrics is not None:
            metrics.observe_query(
                outcome=outcome,
                duration_seconds=perf_counter() - started_at,
            )


def _resolve_principal(request: Request) -> Principal | None:
    settings = getattr(request.app.state, "settings", None)
    if settings is None or not settings.rag_acl_enabled:
        return None
    if not settings.rag_acl_proxy_token:
        _record_acl_denial(request, "proxy_token_missing")
        raise HTTPException(
            status_code=503,
            detail="ACL 已启用但未配置 RAG_ACL_PROXY_TOKEN，服务拒绝处理查询",
        )
    resolver = getattr(request.app.state, "principal_resolver", None)
    if resolver is None:
        resolver = PrincipalResolver(proxy_token=settings.rag_acl_proxy_token)
    try:
        return resolver.resolve(request.headers)
    except AccessControlError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _record_acl_denial(request: Request, reason: str) -> None:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.observe_acl_denial(reason)


__all__ = ["router"]
