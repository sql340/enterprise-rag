from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..data_reader import Document
from .BaseChunker import BaseChunker
from ._types import _Draft, _Span
from ._utils import (
    DEFAULT_SEPARATORS,
    _overlapping_indices,
    _paragraph_spans,
)


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
