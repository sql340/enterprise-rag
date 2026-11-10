from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .KnowledgeDocument import KnowledgeDocument


class KnowledgeBuildFingerprint:
    """文档构建指纹：内容、解析器、分块配置与向量模型共同决定是否重建。"""

    @classmethod
    def build(
        cls,
        document: "KnowledgeDocument",
        *,
        embedding_fingerprint: str | None = None,
        graph_extractor_version: str | None = None,
    ) -> str:
        payload = {
            "source": document.source,
            "file_type": document.file_type,
            "content_hash": document.content_hash,
            "parser_version": document.parser_version,
            "chunk_strategy": document.chunk_strategy,
            "chunk_options": document.chunk_options,
            "embedding_fingerprint": (
                document.embedding_fingerprint
                if embedding_fingerprint is None
                else str(embedding_fingerprint)
            ),
            "graph_extractor_version": graph_extractor_version or "",
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def matches(
        cls,
        document: "KnowledgeDocument",
        *,
        embedding_fingerprint: str | None = None,
        graph_extractor_version: str | None = None,
    ) -> bool:
        stored = str(document.build_fingerprint or "").strip()
        if not stored:
            return False
        return stored == cls.build(
            document,
            embedding_fingerprint=embedding_fingerprint,
            graph_extractor_version=graph_extractor_version,
        )


__all__ = ["KnowledgeBuildFingerprint"]
