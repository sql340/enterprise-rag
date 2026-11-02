"""
企业 RAG 文本分块模块。

实现六种常用分块策略：
- ``fixed``：固定长度、可重叠分块。
- ``recursive``：按段落、换行、中英文句子等分隔符递归分块。
- ``sentence_window``：句子感知分块，可附加前后句窗口。
- ``paragraph``：自然段分块，过长段落自动递归切分。
- ``markdown_heading``：按 Markdown 标题层级分块，保留标题路径。
- ``parent_child``：父子分层分块，用子块检索、父块补充上下文。

所有策略都接受 ``Document``、字符串或它们的序列，返回稳定的 ``Chunk``
列表。默认按字符计数；传入 ``length_function`` 后可按 token 或其它单位
控制大小，例如::

    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    reader = Chunker(
        "recursive",
        chunk_size=512,
        chunk_overlap=64,
        length_function=lambda text: len(encoding.encode(text)),
    )

示例::

    from data_processing.data_reader import DataReader
    from data_processing.chunker import Chunker

    documents = DataReader().read("knowledge/demo.pdf")
    chunks = Chunker("recursive", chunk_size=600, chunk_overlap=80).split(documents)
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar

try:  # 支持作为 package 导入和从 data_processing 目录直接导入。
    from .data_reader import Document
except ImportError:  # pragma: no cover - 兼容直接执行/导入场景
    from data_reader import Document


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


class ChunkStrategy(str, Enum):
    """内置分块策略名称。"""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SENTENCE_WINDOW = "sentence_window"
    PARAGRAPH = "paragraph"
    MARKDOWN_HEADING = "markdown_heading"
    PARENT_CHILD = "parent_child"


@dataclass(slots=True)
class Chunk:
    """分块结果。

    ``content`` 可直接用于 embedding；``metadata`` 继承原 ``Document`` 的
    元数据，并增加分块位置、策略、标题路径和父子关系等字段。
    """

    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    char_start: int
    char_end: int
    strategy: str
    metadata: dict[str, Any]
    content_hash: str
    parent_id: str | None = None

    @property
    def page_content(self) -> str:
        """与常见 RAG 框架兼容的正文别名。"""

        return self.content

    @property
    def chunk_role(self) -> str:
        return str(self.metadata.get("chunk_role", "chunk"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "strategy": self.strategy,
            "content_hash": self.content_hash,
            "parent_id": self.parent_id,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class _Span:
    start: int
    end: int


@dataclass(slots=True)
class _Draft:
    span: _Span
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "chunk"
    parent_index: int | None = None


@dataclass(slots=True)
class _Section:
    span: _Span
    level: int
    title: str
    heading_path: tuple[str, ...]


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


class FixedSizeChunker(BaseChunker):
    """固定字符/token 长度分块，适合作为基线策略。"""

    strategy = "fixed"

    def _split_document(self, document: Document) -> list[_Draft]:
        text = document.content
        spans = self._fixed_spans(text, 0, len(text))
        if (
            len(spans) >= 2
            and self.min_chunk_size > 0
            and self._length(text, spans[-1]) < self.min_chunk_size
        ):
            combined = _Span(spans[-2].start, spans[-1].end)
            if self._length(text, combined) <= self.chunk_size:
                spans[-2:] = [combined]
        return [_Draft(span=span) for span in spans]


class RecursiveChunker(BaseChunker):
    """按分隔符优先级递归分块。"""

    strategy = "recursive"

    def __init__(
        self,
        *,
        separators: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.separators = tuple(
            separator
            for separator in (
                DEFAULT_SEPARATORS if separators is None else separators
            )
            if separator
        )

    def _config_options(self) -> dict[str, Any]:
        options = super()._config_options()
        options["separators"] = list(self.separators)
        return options

    def _split_document(self, document: Document) -> list[_Draft]:
        spans = self._split_region_recursive(
            document.content,
            0,
            len(document.content),
            separators=self.separators,
        )
        return [_Draft(span=span) for span in spans]


class SentenceWindowChunker(BaseChunker):
    """句子感知分块，并为每块附加可选的前后句窗口。"""

    strategy = "sentence_window"

    def __init__(
        self,
        *,
        sentence_window: int = 1,
        separators: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if sentence_window < 0:
            raise ValueError("sentence_window cannot be negative")
        super().__init__(**kwargs)
        self.sentence_window = sentence_window
        self.separators = tuple(
            separator
            for separator in (
                DEFAULT_SEPARATORS[1:]
                if separators is None
                else separators
            )
            if separator
        )

    def _config_options(self) -> dict[str, Any]:
        options = super()._config_options()
        options["sentence_window"] = self.sentence_window
        options["separators"] = list(self.separators)
        return options

    def _split_document(self, document: Document) -> list[_Draft]:
        text = document.content
        sentences = _sentence_spans(text)
        if not sentences:
            return []

        target = self._target_size()
        units: list[_Span] = []
        for sentence in sentences:
            if self._length(text, sentence) <= target:
                units.append(sentence)
            else:
                units.extend(
                    self._recursive_units(
                        text,
                        sentence.start,
                        sentence.end,
                        target,
                        self.separators,
                    )
                )
        merged = self._merge_spans(
            text, units, target, min_size=self.min_chunk_size
        )
        spans = self._apply_overlap(text, merged)

        drafts: list[_Draft] = []
        for span in spans:
            sentence_start, sentence_end = _overlapping_indices(
                sentences, span.start, span.end
            )
            if sentence_start is None or sentence_end is None:
                drafts.append(_Draft(span=span))
                continue

            metadata: dict[str, Any] = {
                "sentence_start_index": sentence_start,
                "sentence_end_index": sentence_end,
                "sentence_count": len(sentences),
            }
            if self.sentence_window > 0:
                window_start = max(0, sentence_start - self.sentence_window)
                window_end = min(
                    len(sentences) - 1,
                    sentence_end + self.sentence_window,
                )
                window_span = _Span(
                    sentences[window_start].start,
                    sentences[window_end].end,
                )
                metadata.update(
                    {
                        "window_size": self.sentence_window,
                        "window_char_start": window_span.start,
                        "window_char_end": window_span.end,
                        "window_sentence_start": window_start,
                        "window_sentence_end": window_end,
                        "window_content": text[window_span.start : window_span.end],
                    }
                )
            drafts.append(_Draft(span=span, metadata=metadata))
        return drafts


class ParagraphChunker(BaseChunker):
    """自然段分块，过长段落会自动递归拆分。"""

    strategy = "paragraph"

    def __init__(
        self,
        *,
        separators: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.separators = tuple(
            separator
            for separator in (
                DEFAULT_SEPARATORS[2:]
                if separators is None
                else separators
            )
            if separator
        )

    def _config_options(self) -> dict[str, Any]:
        options = super()._config_options()
        options["separators"] = list(self.separators)
        return options

    def _split_document(self, document: Document) -> list[_Draft]:
        text = document.content
        paragraphs = _paragraph_spans(text)
        if not paragraphs:
            return []

        target = self._target_size()
        units: list[_Span] = []
        for paragraph in paragraphs:
            if self._length(text, paragraph) <= target:
                units.append(paragraph)
            else:
                units.extend(
                    self._recursive_units(
                        text,
                        paragraph.start,
                        paragraph.end,
                        target,
                        self.separators,
                    )
                )
        merged = self._merge_spans(
            text, units, target, min_size=self.min_chunk_size
        )
        spans = self._apply_overlap(text, merged)

        drafts: list[_Draft] = []
        for span in spans:
            paragraph_start, paragraph_end = _overlapping_indices(
                paragraphs, span.start, span.end
            )
            metadata: dict[str, Any] = {"paragraph_count": len(paragraphs)}
            if paragraph_start is not None and paragraph_end is not None:
                metadata.update(
                    {
                        "paragraph_start_index": paragraph_start,
                        "paragraph_end_index": paragraph_end,
                    }
                )
            drafts.append(_Draft(span=span, metadata=metadata))
        return drafts


class MarkdownHeadingChunker(BaseChunker):
    """按 Markdown 标题层级分块，并保留标题路径。"""

    strategy = "markdown_heading"

    def __init__(
        self,
        *,
        separators: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.separators = tuple(
            separator
            for separator in (
                DEFAULT_SEPARATORS if separators is None else separators
            )
            if separator
        )

    def _config_options(self) -> dict[str, Any]:
        options = super()._config_options()
        options["separators"] = list(self.separators)
        return options

    def _split_document(self, document: Document) -> list[_Draft]:
        text = document.content
        sections = _markdown_sections(text)
        if not sections or not any(section.level > 0 for section in sections):
            spans = self._split_region_recursive(
                text,
                0,
                len(text),
                separators=self.separators,
            )
            return [
                _Draft(
                    span=span,
                    metadata={
                        "markdown_fallback": "recursive",
                        "heading_path": [],
                        "heading_path_text": "",
                    },
                )
                for span in spans
            ]

        drafts: list[_Draft] = []
        for section in sections:
            if self._length(text, section.span) <= self.chunk_size:
                section_spans = [section.span]
            else:
                section_spans = self._split_region_recursive(
                    text,
                    section.span.start,
                    section.span.end,
                    separators=self.separators,
                )
            section_count = len(section_spans)
            for section_index, span in enumerate(section_spans):
                drafts.append(
                    _Draft(
                        span=span,
                        metadata={
                            "heading_level": section.level,
                            "section_title": section.title,
                            "heading_path": list(section.heading_path),
                            "heading_path_text": " > ".join(section.heading_path),
                            "section_chunk_index": section_index,
                            "section_chunk_count": section_count,
                        },
                    )
                )
        return drafts


class ParentChildChunker(BaseChunker):
    """父子分层分块。

    ``chunk_size``/``chunk_overlap`` 控制子块；父块由
    ``parent_chunk_size``/``parent_chunk_overlap`` 控制。输出同时包含
    ``chunk_role=parent`` 和 ``chunk_role=child`` 的块，子块通过
    ``parent_id`` 指向父块。检索时建议索引 child，命中后再按 parent_id
    取回父块上下文。
    """

    strategy = "parent_child"

    def __init__(
        self,
        *,
        parent_chunk_size: int = 2048,
        parent_chunk_overlap: int = 0,
        separators: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be positive")
        if parent_chunk_overlap < 0:
            raise ValueError("parent_chunk_overlap cannot be negative")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError(
                "parent_chunk_overlap must be smaller than parent_chunk_size"
            )
        super().__init__(**kwargs)
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.separators = tuple(
            separator
            for separator in (
                DEFAULT_SEPARATORS if separators is None else separators
            )
            if separator
        )

    def _config_options(self) -> dict[str, Any]:
        options = super()._config_options()
        options.update(
            {
                "parent_chunk_size": self.parent_chunk_size,
                "parent_chunk_overlap": self.parent_chunk_overlap,
                "separators": list(self.separators),
            }
        )
        return options

    def _split_document(self, document: Document) -> list[_Draft]:
        text = document.content
        parent_spans = self._split_region_recursive(
            text,
            0,
            len(text),
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=self.separators,
        )
        parent_spans = _dedupe_spans(parent_spans)
        if not parent_spans:
            return []

        parent_drafts: list[_Draft] = []
        child_drafts: list[_Draft] = []
        for parent_index, parent_span in enumerate(parent_spans):
            parent_drafts.append(
                _Draft(
                    span=parent_span,
                    role="parent",
                    metadata={
                        "parent_index": parent_index,
                        "parent_char_start": parent_span.start,
                        "parent_char_end": parent_span.end,
                    },
                )
            )

            child_spans = self._split_region_recursive(
                text,
                parent_span.start,
                parent_span.end,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )
            child_spans = _dedupe_spans(child_spans)
            if not child_spans:
                child_spans = [parent_span]

            for child_index, child_span in enumerate(child_spans):
                child_drafts.append(
                    _Draft(
                        span=child_span,
                        role="child",
                        parent_index=parent_index,
                        metadata={
                            "parent_index": parent_index,
                            "parent_char_start": parent_span.start,
                            "parent_char_end": parent_span.end,
                            "child_index": child_index,
                            "child_count": len(child_spans),
                        },
                    )
                )

        for parent_index, draft in enumerate(parent_drafts):
            draft.metadata["child_count"] = sum(
                1
                for child in child_drafts
                if child.parent_index == parent_index
            )
        return parent_drafts + child_drafts


class Chunker:
    """分块器门面，通过策略名称调用具体实现。"""

    def __init__(
        self,
        strategy: str | ChunkStrategy = ChunkStrategy.RECURSIVE,
        **options: Any,
    ) -> None:
        self.strategy = _normalise_strategy(strategy)
        self._implementation = create_chunker(self.strategy, **options)

    @property
    def chunk_size(self) -> int:
        return self._implementation.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._implementation.chunk_overlap

    def split(self, documents: DocumentLike | Sequence[DocumentLike]) -> list[Chunk]:
        return self._implementation.split(documents)

    def split_text(self, text: str) -> list[Chunk]:
        return self._implementation.split_text(text)

    def __call__(
        self, documents: DocumentLike | Sequence[DocumentLike]
    ) -> list[Chunk]:
        return self.split(documents)


_STRATEGY_CLASSES: dict[str, type[BaseChunker]] = {
    ChunkStrategy.FIXED.value: FixedSizeChunker,
    ChunkStrategy.RECURSIVE.value: RecursiveChunker,
    ChunkStrategy.SENTENCE_WINDOW.value: SentenceWindowChunker,
    ChunkStrategy.PARAGRAPH.value: ParagraphChunker,
    ChunkStrategy.MARKDOWN_HEADING.value: MarkdownHeadingChunker,
    ChunkStrategy.PARENT_CHILD.value: ParentChildChunker,
}

_STRATEGY_ALIASES = {
    "fixed": "fixed",
    "fixed_size": "fixed",
    "fixed_char": "fixed",
    "recursive": "recursive",
    "recursive_character": "recursive",
    "sentence": "sentence_window",
    "sentence_window": "sentence_window",
    "sentence_aware": "sentence_window",
    "paragraph": "paragraph",
    "paragraph_aware": "paragraph",
    "markdown": "markdown_heading",
    "heading": "markdown_heading",
    "markdown_heading": "markdown_heading",
    "parent_child": "parent_child",
    "hierarchical": "parent_child",
    "parent": "parent_child",
}


def create_chunker(
    strategy: str | ChunkStrategy = ChunkStrategy.RECURSIVE,
    **options: Any,
) -> BaseChunker:
    """创建具体分块器实例。"""

    canonical = _normalise_strategy(strategy)
    return _STRATEGY_CLASSES[canonical](**options)


def chunk_documents(
    documents: DocumentLike | Sequence[DocumentLike],
    strategy: str | ChunkStrategy = ChunkStrategy.RECURSIVE,
    **options: Any,
) -> list[Chunk]:
    """使用指定策略对文档执行一次性分块。"""

    return Chunker(strategy, **options).split(documents)


def _normalise_strategy(strategy: str | ChunkStrategy) -> str:
    value = strategy.value if isinstance(strategy, ChunkStrategy) else str(strategy)
    key = value.strip().lower().replace("-", "_")
    canonical = _STRATEGY_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_STRATEGY_CLASSES))
        raise ValueError(f"unsupported chunk strategy: {strategy}; supported: {supported}")
    return canonical


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


__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkStrategy",
    "Chunker",
    "DEFAULT_SEPARATORS",
    "DocumentLike",
    "FixedSizeChunker",
    "MarkdownHeadingChunker",
    "ParagraphChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "chunk_documents",
    "create_chunker",
]
