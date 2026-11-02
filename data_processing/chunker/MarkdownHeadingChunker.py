from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..data_reader import Document
from .BaseChunker import BaseChunker
from ._types import _Draft
from ._utils import DEFAULT_SEPARATORS, _markdown_sections


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
