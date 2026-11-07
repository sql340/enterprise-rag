from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from access_control import DocumentAcl
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
        skip_existing: 是否跳过同一模型中 chunk_id 与 content_hash 均未变化
            的知识块，避免重复 Embedding。
        acl: 默认文档权限；写入时合并进每个知识块的 payload。
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        *,
        sync_scope: str = "source",
        delete_stale: bool = True,
        skip_existing: bool = True,
        acl: DocumentAcl | None = None,
    ) -> None:
        if acl is not None and not isinstance(acl, DocumentAcl):
            raise TypeError("acl 必须是 DocumentAcl 或 None")
        self.embedding = embedding
        self.vector_store = vector_store
        self.sync_scope = self._normalise_scope(sync_scope)
        self.delete_stale = delete_stale
        self.skip_existing = skip_existing
        self.acl = acl

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        sync_scope: str | None = None,
        delete_stale: bool | None = None,
        force_reindex: bool | None = None,
        acl: DocumentAcl | None = None,
    ) -> IndexingResult:
        """向量化并幂等写入知识块。"""

        if not chunks:
            raise ValueError("chunks 不能为空")
        active_acl = acl or self.acl
        if active_acl is not None and not isinstance(active_acl, DocumentAcl):
            raise TypeError("acl 必须是 DocumentAcl 或 None")
        values = [chunk for chunk in chunks if chunk.content.strip()]
        if not values:
            raise ValueError("chunks 中没有可向量化的文本")
        if active_acl is not None:
            acl_payload = active_acl.to_payload()
            values = [
                replace(chunk, metadata={**chunk.metadata, **acl_payload})
                for chunk in values
            ]

        scope = self._normalise_scope(sync_scope or self.sync_scope)
        should_delete_stale = (
            self.delete_stale if delete_stale is None else delete_stale
        )
        should_skip_existing = (
            self.skip_existing if force_reindex is None else not force_reindex
        )
        groups = self._group_chunks(values, scope)
        embedded = 0
        skipped = 0
        stale_deleted = 0

        for (scope_field, scope_value), group_chunks in groups.items():
            existing = (
                self.vector_store.list_chunk_records(
                    scope_field=scope_field,
                    scope_value=scope_value,
                )
                if should_skip_existing
                else {}
            )
            to_embed = [
                chunk
                for chunk in group_chunks
                if not self._is_unchanged(
                    chunk,
                    existing.get(chunk.chunk_id),
                )
            ]
            skipped += len(group_chunks) - len(to_embed)

            if to_embed:
                vectors = self.embedding.embed_documents(
                    [chunk.content for chunk in to_embed]
                )
                if not vectors:
                    raise KnowledgeIndexError(
                        f"Embedding 未返回向量: {scope_field}={scope_value}"
                    )
                embedded += self.vector_store.upsert_chunks(
                    to_embed,
                    vectors,
                    embedding_model=self.embedding.model_name,
                )

            if should_delete_stale:
                stale_deleted += self.vector_store.prune_stale_chunks(
                    scope_field=scope_field,
                    scope_value=scope_value,
                    expected_chunk_ids=[chunk.chunk_id for chunk in group_chunks],
                )

        return IndexingResult(
            indexed_chunks=embedded,
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
            total_chunks=len(values),
            embedded_chunks=embedded,
            skipped_chunks=skipped,
            stale_deleted=stale_deleted,
        )

    def index_documents(
        self,
        documents: DocumentLike | Sequence[DocumentLike],
        chunker: Chunker,
        **options: Any,
    ) -> IndexingResult:
        """先分块再入库的便捷方法。"""

        return self.index_chunks(chunker.split(documents), **options)

    def update_acl_by_document(
        self,
        document_id: str,
        acl: DocumentAcl,
    ) -> None:
        """原地更新一个文档的 ACL，不重新计算向量。"""

        if not isinstance(acl, DocumentAcl):
            raise TypeError("acl 必须是 DocumentAcl")
        self.vector_store.update_payload_by_document(
            document_id,
            acl.to_payload(),
        )

    @property
    def _collection_name(self) -> str:
        return str(getattr(self.vector_store, "collection_name", ""))

    def _is_unchanged(
        self,
        chunk: Chunk,
        existing: dict[str, Any] | None,
    ) -> bool:
        if not existing:
            return False
        return (
            str(existing.get("embedding_model") or "") == self.embedding.model_name
            and str(existing.get("content_hash") or "") == chunk.content_hash
            and DocumentAcl.payload_signature(existing)
            == DocumentAcl.payload_signature(chunk.metadata)
        )

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
