from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestContext:
    """一次请求的日志关联信息。"""

    request_id: str
    trace_id: str


_current_context: ContextVar[RequestContext | None] = ContextVar(
    "rag_request_context",
    default=None,
)


def current_request_context() -> RequestContext | None:
    return _current_context.get()


def bind_request_context(context: RequestContext) -> Token[RequestContext | None]:
    return _current_context.set(context)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _current_context.reset(token)


def current_request_id() -> str | None:
    context = current_request_context()
    return context.request_id if context is not None else None


def current_trace_id() -> str | None:
    context = current_request_context()
    return context.trace_id if context is not None else None


__all__ = [
    "RequestContext",
    "bind_request_context",
    "current_request_context",
    "current_request_id",
    "current_trace_id",
    "reset_request_context",
]
