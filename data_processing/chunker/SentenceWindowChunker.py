from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..data_reader import Document
from .BaseChunker import BaseChunker
from ._types import _Draft, _Span
from ._utils import (
    DEFAULT_SEPARATORS,
    _overlapping_indices,
    _sentence_spans,
)


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
