from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..data_reader import Document
from .BaseChunker import BaseChunker
from ._types import _Draft
from ._utils import DEFAULT_SEPARATORS, _dedupe_spans


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
