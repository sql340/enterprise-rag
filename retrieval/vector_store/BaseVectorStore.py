from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from data_processing.chunker import Chunk

from .VectorSearchResult import VectorSearchResult


class BaseVectorStore(ABC):
    """向量库接口，便于后续替换 Qdrant 或接入其它实现。"""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """确保 collection 存在且向量维度一致。"""

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> int:
        """按稳定 ID 幂等写入知识块向量。"""

    @abstractmethod
    def sync_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str,
        scope: str = "source",
        delete_stale: bool = True,
    ) -> int:
        """写入新向量，并可选删除同一来源下已经过期的知识块。"""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """执行向量相似度检索。"""

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        """删除一个文档版本下的全部向量。"""

    @abstractmethod
    def count(self) -> int:
        """返回 collection 中的向量数量。"""

    @abstractmethod
    def close(self) -> None:
        """关闭底层客户端。"""


__all__ = ["BaseVectorStore"]
