from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..data_reader import Document
from .BaseChunker import BaseChunker
from ._types import _Draft
from ._utils import DEFAULT_SEPARATORS


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
