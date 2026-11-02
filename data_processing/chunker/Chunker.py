from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .BaseChunker import BaseChunker
from .Chunk import Chunk
from .ChunkStrategy import ChunkStrategy
from .FixedSizeChunker import FixedSizeChunker
from .MarkdownHeadingChunker import MarkdownHeadingChunker
from .ParagraphChunker import ParagraphChunker
from .ParentChildChunker import ParentChildChunker
from .RecursiveChunker import RecursiveChunker
from .SentenceWindowChunker import SentenceWindowChunker
from ._utils import DocumentLike


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


def _normalise_strategy(strategy: str | ChunkStrategy) -> str:
    value = strategy.value if isinstance(strategy, ChunkStrategy) else str(strategy)
    key = value.strip().lower().replace("-", "_")
    canonical = _STRATEGY_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(_STRATEGY_CLASSES))
        raise ValueError(f"unsupported chunk strategy: {strategy}; supported: {supported}")
    return canonical


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
