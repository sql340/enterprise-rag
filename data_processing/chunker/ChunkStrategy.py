from __future__ import annotations

from enum import Enum


class ChunkStrategy(str, Enum):
    """内置分块策略名称。"""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SENTENCE_WINDOW = "sentence_window"
    PARAGRAPH = "paragraph"
    MARKDOWN_HEADING = "markdown_heading"
    PARENT_CHILD = "parent_child"
