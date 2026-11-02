from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from ..data_reader import Document
from .Chunk import Chunk
from ._types import _Draft, _Span
from ._utils import (
    DEFAULT_SEPARATORS,
    DocumentLike,
    LengthFunction,
    _callable_id,
    _hash_json,
    _normalise_documents,
    _normalise_text,
    _sha256,
    _trim_span,
)


class BaseChunker(ABC):
    """分块器基类。

    Args:
        chunk_size: 最大块大小。默认单位为字符。
        chunk_overlap: 相邻块重叠大小。默认取 ``min(64, chunk_size // 8)``；
            显式传入时必须小于 ``chunk_size``。
        min_chunk_size: 期望的最小块大小，用于尝试合并尾部过小块。
        length_function: 长度函数，默认 ``len``。可传入 tokenizer 计数函数。
    """

    strategy: ClassVar[str] = "base"

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int | None = None,
        min_chunk_size: int = 1,
        length_function: LengthFunction = len,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap is None:
            chunk_overlap = min(64, max(0, chunk_size // 8))
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size cannot be negative")
        if not callable(length_function):
            raise TypeError("length_function must be callable")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.length_function = length_function

    def split(self, documents: DocumentLike | Sequence[DocumentLike]) -> list[Chunk]:
        """对一个或多个文档执行分块。"""

        chunks: list[Chunk] = []
        for document in _normalise_documents(documents):
            text = _normalise_text(document.content)
            if not text.strip():
                continue
            normalised_document = Document(
                content=text, metadata=dict(document.metadata)
            )
            drafts = self._split_document(normalised_document)
            chunks.extend(self._build_chunks(normalised_document, drafts))
        return chunks

    def split_text(self, text: str) -> list[Chunk]:
        """对纯字符串执行分块。"""

        return self.split(Document(content=text))

    def __call__(
        self, documents: DocumentLike | Sequence[DocumentLike]
    ) -> list[Chunk]:
        return self.split(documents)

    @abstractmethod
    def _split_document(self, document: Document) -> list[_Draft]:
        """返回相对于 ``document.content`` 的草稿分块。"""

    def _build_chunks(self, document: Document, drafts: list[_Draft]) -> list[Chunk]:
        text = document.content
        valid_pairs: list[tuple[int, _Draft]] = []
        seen_spans: set[tuple[str, int, int, int | None]] = set()
        for index, draft in enumerate(drafts):
            if (
                draft.span.start >= draft.span.end
                or not text[draft.span.start : draft.span.end].strip()
            ):
                continue
            key = (
                draft.role,
                draft.span.start,
                draft.span.end,
                draft.parent_index,
            )
            if key in seen_spans:
                continue
            seen_spans.add(key)
            valid_pairs.append((index, draft))
        valid_drafts = [draft for _, draft in valid_pairs]
        if not valid_drafts:
            return []

        old_to_new = {
            old_index: new_index
            for new_index, (old_index, _) in enumerate(valid_pairs)
        }
        config_options = self._config_options()
        config_hash = _hash_json(config_options)
        source = str(
            document.metadata.get("source")
            or document.metadata.get("file_name")
            or ""
        )
        document_content_hash = _sha256(text)
        document_id = _sha256(f"{source}\0{document_content_hash}")[:32]
        chunk_count = len(valid_drafts)

        chunks: list[Chunk] = []
        chunk_ids: list[str] = []
        role_counts: dict[str, int] = {}
        for draft in valid_drafts:
            role_counts[draft.role] = role_counts.get(draft.role, 0) + 1
        role_positions: dict[str, int] = {}
        for index, draft in enumerate(valid_drafts):
            content = text[draft.span.start : draft.span.end]
            content_hash = _sha256(content)
            parent_id: str | None = None
            if draft.parent_index is not None:
                parent_position = old_to_new.get(draft.parent_index)
                if parent_position is not None:
                    parent_id = chunk_ids[parent_position]

            metadata = {
                **document.metadata,
                **draft.metadata,
                "document_id": document_id,
                "document_content_hash": document_content_hash,
                "chunk_index": index,
                "chunk_count": chunk_count,
                "chunk_strategy": self.strategy,
                "chunk_role": draft.role,
                "chunk_char_start": draft.span.start,
                "chunk_char_end": draft.span.end,
                "chunk_length": self.length_function(content),
                "chunk_content_hash": content_hash,
                "chunk_config_hash": config_hash,
            }
            if parent_id is not None:
                metadata["parent_id"] = parent_id
            role_position = role_positions.get(draft.role, 0)
            role_positions[draft.role] = role_position + 1
            metadata["chunk_role_index"] = role_position
            metadata["chunk_role_count"] = role_counts[draft.role]

            chunk_id = _sha256(
                "\0".join(
                    (
                        document_id,
                        self.strategy,
                        config_hash,
                        draft.role,
                        str(draft.span.start),
                        str(draft.span.end),
                        content_hash,
                        parent_id or "",
                    )
                )
            )[:32]
            chunk_ids.append(chunk_id)
            chunks.append(
                Chunk(
                    content=content,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=index,
                    char_start=draft.span.start,
                    char_end=draft.span.end,
                    strategy=self.strategy,
                    metadata=metadata,
                    content_hash=content_hash,
                    parent_id=parent_id,
                )
            )
        return chunks

    def _config_options(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "length_function": _callable_id(self.length_function),
        }

    def _length(self, text: str, span: _Span) -> int:
        return self.length_function(text[span.start : span.end])

    def _target_size(self) -> int:
        return max(1, self.chunk_size - self.chunk_overlap)

    def _max_end(
        self, text: str, start: int, end: int, max_units: int
    ) -> int:
        """寻找从 ``start`` 开始、长度不超过 ``max_units`` 的最远字符位置。"""

        if max_units <= 0 or start >= end:
            return start
        if self.length_function(text[start:end]) <= max_units:
            return end

        low = start + 1
        high = end
        best = start + 1
        while low <= high:
            middle = (low + high) // 2
            if self.length_function(text[start:middle]) <= max_units:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _overlap_start(
        self, text: str, start: int, end: int, overlap_units: int
    ) -> int:
        """寻找以 ``end`` 结尾、长度不超过 ``overlap_units`` 的最早起点。"""

        if overlap_units <= 0 or start >= end:
            return end
        if self.length_function(text[start:end]) <= overlap_units:
            return start

        low = start
        high = end - 1
        best = end
        while low <= high:
            middle = (low + high) // 2
            if self.length_function(text[middle:end]) <= overlap_units:
                best = middle
                high = middle - 1
            else:
                low = middle + 1
        return best

    def _suffix_start_for_max(
        self, text: str, start: int, end: int, max_units: int
    ) -> int:
        """寻找以 ``end`` 结尾、长度不超过 ``max_units`` 的最晚起点。"""

        if start >= end:
            return end
        if self.length_function(text[start:end]) <= max_units:
            return start

        low = start + 1
        high = end
        best = end
        while low <= high:
            middle = (low + high) // 2
            if self.length_function(text[middle:end]) <= max_units:
                best = middle
                high = middle - 1
            else:
                low = middle + 1
        return best

    def _fixed_spans(
        self, text: str, start: int, end: int
    ) -> list[_Span]:
        spans: list[_Span] = []
        cursor = start
        while cursor < end:
            chunk_end = self._max_end(
                text, cursor, end, max_units=self.chunk_size
            )
            span = _trim_span(text, _Span(cursor, chunk_end))
            if span.start < span.end:
                spans.append(span)
            if chunk_end >= end:
                break

            if self.chunk_overlap:
                next_cursor = self._overlap_start(
                    text,
                    cursor,
                    chunk_end,
                    overlap_units=self.chunk_overlap,
                )
            else:
                next_cursor = chunk_end
            cursor = max(cursor + 1, next_cursor)
            if cursor >= chunk_end:
                cursor = chunk_end
        return spans

    def _hard_spans(
        self, text: str, start: int, end: int, size: int
    ) -> list[_Span]:
        spans: list[_Span] = []
        cursor = start
        while cursor < end:
            chunk_end = self._max_end(text, cursor, end, max_units=size)
            span = _trim_span(text, _Span(cursor, chunk_end))
            if span.start < span.end:
                spans.append(span)
            if chunk_end >= end:
                break
            cursor = max(cursor + 1, chunk_end)
        return spans

    def _split_at_separator(
        self, text: str, start: int, end: int, separator: str
    ) -> list[_Span]:
        if not separator:
            return [_Span(start, end)]
        spans: list[_Span] = []
        cursor = start
        while cursor < end:
            index = text.find(separator, cursor, end)
            if index < 0:
                spans.append(_Span(cursor, end))
                break
            part_end = index + len(separator)
            spans.append(_Span(cursor, part_end))
            cursor = part_end
        return spans

    def _recursive_units(
        self,
        text: str,
        start: int,
        end: int,
        target_size: int,
        separators: Sequence[str],
    ) -> list[_Span]:
        initial = _trim_span(text, _Span(start, end))
        if initial.start >= initial.end:
            return []
        if self._length(text, initial) <= target_size:
            return [initial]
        if not separators:
            return self._hard_spans(text, initial.start, initial.end, target_size)

        separator = separators[0]
        remaining = separators[1:]
        parts = self._split_at_separator(
            text, initial.start, initial.end, separator
        )
        if len(parts) == 1:
            return self._recursive_units(
                text, initial.start, initial.end, target_size, remaining
            )

        units: list[_Span] = []
        for part in parts:
            trimmed = _trim_span(text, part)
            if trimmed.start >= trimmed.end:
                continue
            if self._length(text, trimmed) <= target_size:
                units.append(trimmed)
            else:
                units.extend(
                    self._recursive_units(
                        text,
                        trimmed.start,
                        trimmed.end,
                        target_size,
                        remaining,
                    )
                )
        return units

    def _merge_spans(
        self,
        text: str,
        units: Sequence[_Span],
        target_size: int,
        min_size: int,
    ) -> list[_Span]:
        if not units:
            return []

        merged: list[_Span] = []
        current = units[0]
        for unit in units[1:]:
            candidate = _Span(current.start, unit.end)
            if self._length(text, candidate) <= target_size:
                current = candidate
            else:
                merged.append(current)
                current = unit
        merged.append(current)

        if (
            len(merged) >= 2
            and min_size > 0
            and self._length(text, merged[-1]) < min_size
        ):
            combined = _Span(merged[-2].start, merged[-1].end)
            if self._length(text, combined) <= target_size:
                merged[-2:] = [combined]
        return merged

    def _apply_overlap(
        self, text: str, spans: Sequence[_Span]
    ) -> list[_Span]:
        if not spans or not self.chunk_overlap:
            return list(spans)

        result: list[_Span] = []
        for index, span in enumerate(spans):
            if index == 0:
                result.append(span)
                continue
            previous = spans[index - 1]
            overlap_start = self._overlap_start(
                text,
                previous.start,
                previous.end,
                overlap_units=self.chunk_overlap,
            )
            expanded_start = min(overlap_start, span.start)
            if self.length_function(
                text[expanded_start : span.end]
            ) > self.chunk_size:
                expanded_start = self._suffix_start_for_max(
                    text, expanded_start, span.end, self.chunk_size
                )
            result.append(_Span(expanded_start, span.end))
        return result

    def _split_region_recursive(
        self,
        text: str,
        start: int,
        end: int,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: Sequence[str] | None = None,
    ) -> list[_Span]:
        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else self.chunk_overlap
        )
        target = max(1, size - overlap)
        active_separators = (
            tuple(separators)
            if separators is not None
            else DEFAULT_SEPARATORS
        )
        units = self._recursive_units(
            text, start, end, target, active_separators
        )
        merged = self._merge_spans(
            text, units, target, min_size=self.min_chunk_size
        )
        return self._apply_overlap(text, merged)
