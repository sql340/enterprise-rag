from __future__ import annotations

from ..data_reader import Document
from .BaseChunker import BaseChunker
from ._types import _Draft, _Span


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
