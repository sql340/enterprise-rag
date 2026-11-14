from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from .RequestContext import current_request_context


_EXTRA_FIELDS = (
    "event",
    "method",
    "route",
    "status",
    "duration_ms",
    "error_type",
    "operation",
    "outcome",
    "mode",
    "provider",
    "model",
    "job_id",
    "worker_id",
    "document_id",
    "attempt",
)


class JsonLogFormatter(logging.Formatter):
    """输出一行一个 JSON 对象，并自动附加请求关联 ID。"""

    def format(self, record: logging.LogRecord) -> str:
        context = current_request_context()
        payload: dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if context is not None:
            payload["request_id"] = context.request_id
            payload["trace_id"] = context.trace_id

        for field in _EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = _json_safe(getattr(record, field))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)[:8000]
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


__all__ = ["JsonLogFormatter"]
