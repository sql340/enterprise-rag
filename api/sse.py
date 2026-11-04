from __future__ import annotations

import json

from retrieval import RagStreamEvent


def encode_sse_event(event: RagStreamEvent, event_id: int) -> str:
    """把 RAG 事件编码为 SSE 帧。"""

    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    data_lines = "\n".join(f"data: {line}" for line in data.splitlines() or [""])
    return f"id: {event_id}\nevent: {event.event}\n{data_lines}\n\n"


__all__ = ["encode_sse_event"]
