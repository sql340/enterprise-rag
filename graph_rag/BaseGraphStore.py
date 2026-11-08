from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .GraphEntity import GraphEntity
from .GraphRelation import GraphRelation
from .GraphSearchHit import GraphSearchHit


class BaseGraphStore(ABC):
    """知识图谱存储接口。"""

    @abstractmethod
    def is_chunk_current(
        self,
        *,
        tenant_id: str,
        chunk_id: str,
        content_hash: str,
        extractor_version: str,
    ) -> bool:
        """判断知识块的图谱是否已按相同内容和抽取版本构建。"""

    @abstractmethod
    def replace_chunk_graph(
        self,
        *,
        tenant_id: str,
        document_id: str,
        chunk_id: str,
        content_hash: str,
        extractor_version: str,
        entities: Sequence[GraphEntity],
        relations: Sequence[GraphRelation],
    ) -> None:
        """原子替换一个知识块关联的实体和关系。"""

    @abstractmethod
    def search_chunk_ids(
        self,
        question: str,
        *,
        tenant_id: str,
        limit: int,
        max_hops: int,
        entity_limit: int,
    ) -> list[GraphSearchHit]:
        """按问题中出现的实体检索知识块，并可选扩展邻居实体。"""

    @abstractmethod
    def delete_by_document(self, *, tenant_id: str, document_id: str) -> None:
        """删除文档对应的图谱数据。"""

    @abstractmethod
    def prune_stale_chunks(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids: Sequence[str],
    ) -> int:
        """删除文档下不在预期集合中的图谱知识块。"""

    @abstractmethod
    def count_entities(self, *, tenant_id: str | None = None) -> int:
        """返回实体数量。"""

    @abstractmethod
    def count_relations(self, *, tenant_id: str | None = None) -> int:
        """返回关系数量。"""

    @abstractmethod
    def close(self) -> None:
        """释放存储资源。"""


__all__ = ["BaseGraphStore"]
