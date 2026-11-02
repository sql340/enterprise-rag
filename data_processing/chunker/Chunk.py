from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
