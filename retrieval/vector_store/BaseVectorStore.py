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
        expected_chunk_ids: Sequence[str] | None = None,
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

    def merge_filters(self, *filters: Any) -> Any:
        """把访问控制条件和调用方过滤条件合并为 AND 关系。"""

        values = [value for value in filters if value is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return tuple(values)

    @property
    def supports_text_search(self) -> bool:
        """是否支持基于 payload 文本索引的 BM25 检索。"""

        return False

    def text_search(
        self,
        query_text: str,
        *,
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """执行 BM25 文本检索；不支持的实现应保持默认异常。"""

        raise NotImplementedError("当前向量库不支持 BM25 文本检索")

    def list_chunk_records(
        self,
        *,
        scope_field: str,
        scope_value: str,
    ) -> dict[str, dict[str, Any]]:
        """返回同步范围内 chunk_id 到关键 payload 的映射。"""

        raise NotImplementedError("当前向量库不支持按范围读取知识块")

    def prune_stale_chunks(
        self,
        *,
        scope_field: str,
        scope_value: str,
        expected_chunk_ids: Sequence[str],
    ) -> int:
        """删除同步范围内不在预期 chunk_id 集合中的旧块。"""

        raise NotImplementedError("当前向量库不支持清理过期知识块")

    def update_payload_by_document(
        self,
        document_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """原地更新文档下全部知识块的 payload，不重新计算向量。"""

        raise NotImplementedError("当前向量库不支持原地更新文档 payload")

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
