from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable

from ..data_reader import Document
from ._types import _Section, _Span


LengthFunction = Callable[[str], int]
DocumentLike = Document | str | Mapping[str, Any]

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "…",
    ". ",
    "! ",
    "? ",
    "; ",
    "，",
    ", ",
    " ",
)


def _normalise_documents(
    documents: DocumentLike | Sequence[DocumentLike],
) -> list[Document]:
    if isinstance(documents, (Document, str, Mapping)) or hasattr(
        documents, "content"
    ):
        values: Iterable[DocumentLike] = [documents]  # type: ignore[list-item]
    else:
        values = documents
    return [_as_document(value) for value in values]


def _as_document(value: DocumentLike | Any) -> Document:
    if isinstance(value, Document):
        return value
    if isinstance(value, str):
        return Document(content=value)
    if isinstance(value, Mapping):
        content = value.get("content", value.get("page_content", ""))
        metadata = value.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError("document metadata must be a mapping")
        return Document(content=str(content or ""), metadata=dict(metadata))
    content = getattr(value, "content", getattr(value, "page_content", None))
    if content is None:
        raise TypeError(f"unsupported document input: {type(value).__name__}")
    metadata = getattr(value, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError("document metadata must be a mapping")
    return Document(content=str(content), metadata=dict(metadata))


def _normalise_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _trim_span(text: str, span: _Span) -> _Span:
    start = span.start
    end = min(span.end, len(text))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Span(start, end)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _callable_id(function: Callable[..., Any]) -> str:
    return (
        f"{getattr(function, '__module__', '')}."
        f"{getattr(function, '__qualname__', getattr(function, '__name__', 'callable'))}"
    )


def _overlapping_indices(
    spans: Sequence[_Span], start: int, end: int
) -> tuple[int | None, int | None]:
    matches = [
        index
        for index, span in enumerate(spans)
        if span.start < end and span.end > start
    ]
    if not matches:
        return None, None
    return matches[0], matches[-1]


def _dedupe_spans(spans: Sequence[_Span]) -> list[_Span]:
    unique: list[_Span] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        key = (span.start, span.end)
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique


def _sentence_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    punctuation = set("。！？!?；;…")
    closers = set("\"'”’）】]」』")
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        boundary = character in punctuation
        if character == "." and (index + 1 == len(text) or text[index + 1].isspace()):
            boundary = True
        if character == "\n":
            boundary = True

        if not boundary:
            index += 1
            continue

        boundary_end = index + 1
        while boundary_end < len(text) and text[boundary_end] in punctuation:
            boundary_end += 1
        while boundary_end < len(text) and text[boundary_end] in closers:
            boundary_end += 1
        span = _trim_span(text, _Span(start, boundary_end))
        if span.start < span.end:
            spans.append(span)
        start = boundary_end
        index = boundary_end

    tail = _trim_span(text, _Span(start, len(text)))
    if tail.start < tail.end:
        spans.append(tail)
    return spans


def _paragraph_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    cursor = 0
    for match in re.finditer(r"\n[ \t]*\n+", text):
        span = _trim_span(text, _Span(cursor, match.start()))
        if span.start < span.end:
            spans.append(span)
        cursor = match.end()
    tail = _trim_span(text, _Span(cursor, len(text)))
    if tail.start < tail.end:
        spans.append(tail)

    if len(spans) <= 1 and text.count("\n") >= 2:
        line_spans: list[_Span] = []
        cursor = 0
        for match in re.finditer(r"\n", text):
            span = _trim_span(text, _Span(cursor, match.start()))
            if span.start < span.end:
                line_spans.append(span)
            cursor = match.end()
        tail = _trim_span(text, _Span(cursor, len(text)))
        if tail.start < tail.end:
            line_spans.append(tail)
        if len(line_spans) > 1:
            return line_spans
    return spans


_MARKDOWN_HEADING_RE = re.compile(
    r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$"
)
_MARKDOWN_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _markdown_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    section_start = 0
    current_level = 0
    current_title = ""
    current_path: tuple[str, ...] = ()
    fence_marker: str | None = None
    offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        fence_match = _MARKDOWN_FENCE_RE.match(stripped)
        if fence_match is not None:
            marker = fence_match.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif fence_marker == marker:
                fence_marker = None
            offset += len(line)
            continue

        heading_match = (
            _MARKDOWN_HEADING_RE.match(stripped)
            if fence_marker is None
            else None
        )
        if heading_match is not None:
            previous = _trim_span(text, _Span(section_start, offset))
            if previous.start < previous.end:
                sections.append(
                    _Section(
                        span=previous,
                        level=current_level,
                        title=current_title,
                        heading_path=current_path,
                    )
                )

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_level = level
            current_title = title
            current_path = tuple(item[1] for item in heading_stack)
            section_start = offset
        offset += len(line)

    tail = _trim_span(text, _Span(section_start, len(text)))
    if tail.start < tail.end:
        sections.append(
            _Section(
                span=tail,
                level=current_level,
                title=current_title,
                heading_path=current_path,
            )
        )
    return sections
