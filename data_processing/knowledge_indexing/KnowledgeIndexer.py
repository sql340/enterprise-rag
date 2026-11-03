from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from data_processing.chunker import Chunk, Chunker, DocumentLike
from retrieval.vector_store import BaseVectorStore

from .BaseEmbedding import BaseEmbedding
from .IndexingResult import IndexingResult
from .KnowledgeIndexError import KnowledgeIndexError


class KnowledgeIndexer:
    """把 ``Chunk`` 批量向量化并写入向量库。

    默认使用 ``source`` 作为同步范围：同一文件的全部知识块写入完成后，
    删除该文件下不属于本次结果的旧块。这样文件内容更新后，即使生成了
    新的 ``document_id``，旧版本向量也会被清理。

    Args:
        embedding: Embedding 提供者。
        vector_store: 向量库实现。
        sync_scope: ``source`` 或 ``document``。按文件更新用 ``source``；
            只处理单个文档版本或部分页面时用 ``document``。
        delete_stale: 是否删除同步范围内本次未写入的旧块。
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        *,
        sync_scope: str = "source",
        delete_stale: bool = True,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.sync_scope = self._normalise_scope(sync_scope)
        self.delete_stale = delete_stale

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        sync_scope: str | None = None,
        delete_stale: bool | None = None,
    ) -> IndexingResult:
        """向量化并幂等写入知识块。"""

        if not chunks:
            raise ValueError("chunks 不能为空")
        values = [chunk for chunk in chunks if chunk.content.strip()]
        if not values:
            raise ValueError("chunks 中没有可向量化的文本")

        scope = self._normalise_scope(sync_scope or self.sync_scope)
        should_delete_stale = (
            self.delete_stale if delete_stale is None else delete_stale
        )
        groups = self._group_chunks(values, scope)
        indexed = 0
        collection_ready = False

        for (scope_field, scope_value), group_chunks in groups.items():
            vectors = self.embedding.embed_documents(
                [chunk.content for chunk in group_chunks]
            )
            if not vectors:
                raise KnowledgeIndexError(
                    f"Embedding 未返回向量: {scope_field}={scope_value}"
                )
            if not collection_ready:
                self.vector_store.ensure_collection(len(vectors[0]))
                collection_ready = True

            store_scope = "source" if scope_field == "source" else "document"
            indexed += self.vector_store.sync_chunks(
                group_chunks,
                vectors,
                embedding_model=self.embedding.model_name,
                scope=store_scope,
                delete_stale=should_delete_stale,
            )

        return IndexingResult(
            indexed_chunks=indexed,
            document_ids=tuple(sorted({chunk.document_id for chunk in values})),
            sources=tuple(
                sorted(
                    {
                        str(chunk.metadata["source"])
                        for chunk in values
                        if chunk.metadata.get("source")
                    }
                )
            ),
            collection_name=self._collection_name,
        )

    def index_documents(
        self,
        documents: DocumentLike | Sequence[DocumentLike],
        chunker: Chunker,
        **options: Any,
    ) -> IndexingResult:
        """先分块再入库的便捷方法。"""

        return self.index_chunks(chunker.split(documents), **options)

    @property
    def _collection_name(self) -> str:
        return str(getattr(self.vector_store, "collection_name", ""))

    @staticmethod
    def _normalise_scope(scope: str) -> str:
        value = scope.strip().lower()
        if value in {"source", "file"}:
            return "source"
        if value in {"document", "document_id"}:
            return "document"
        raise ValueError("sync_scope 必须是 source 或 document")

    @staticmethod
    def _group_chunks(
        chunks: Sequence[Chunk],
        scope: str,
    ) -> dict[tuple[str, str], list[Chunk]]:
        groups: dict[tuple[str, str], list[Chunk]] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source")
            if scope == "source" and source:
                key = ("source", str(source))
            else:
                key = ("document_id", chunk.document_id)
            groups.setdefault(key, []).append(chunk)
        return groups


__all__ = ["KnowledgeIndexer"]
