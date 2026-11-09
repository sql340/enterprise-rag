from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from access_control import DocumentAcl
from data_processing.chunker import Chunk, Chunker, DocumentLike, TenantChunkScope

from .BaseGraphExtractor import BaseGraphExtractor
from .BaseGraphStore import BaseGraphStore
from .GraphIndexingResult import GraphIndexingResult


class GraphIndexer:
    """把知识块中的实体关系写入图谱。

    图谱构建独立于向量入库，但读取相同的 ``chunk_id``。未变化的知识块会按
    ``content_hash + extractor_version`` 跳过，避免重复调用抽取模型。
    """

    def __init__(
        self,
        graph_store: BaseGraphStore,
        extractor: BaseGraphExtractor,
        *,
        force_reindex: bool = False,
        continue_on_error: bool = True,
        acl: DocumentAcl | None = None,
    ) -> None:
        if acl is not None and not isinstance(acl, DocumentAcl):
            raise TypeError("acl 必须是 DocumentAcl 或 None")
        self.graph_store = graph_store
        self.extractor = extractor
        self.force_reindex = force_reindex
        self.continue_on_error = continue_on_error
        self.acl = acl

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        force_reindex: bool | None = None,
        acl: DocumentAcl | None = None,
    ) -> GraphIndexingResult:
        if not chunks:
            raise ValueError("chunks 不能为空")
        active_acl = acl or self.acl
        if active_acl is not None and not isinstance(active_acl, DocumentAcl):
            raise TypeError("acl 必须是 DocumentAcl 或 None")
        force = self.force_reindex if force_reindex is None else force_reindex
        indexed = 0
        skipped = 0
        entity_count = 0
        relation_count = 0
        failures: list[dict[str, str]] = []

        for chunk in chunks:
            tenant_id = (
                active_acl.tenant_id
                if active_acl is not None
                else str(chunk.metadata.get("tenant_id") or "").strip()
            )
            if not tenant_id:
                error = "知识块缺少 tenant_id，不能构建图谱"
                if not self.continue_on_error:
                    raise ValueError(error)
                failures.append({"chunk_id": chunk.chunk_id, "error": error})
                continue
            try:
                scoped_chunk = TenantChunkScope.apply([chunk], tenant_id)[0]
            except ValueError as exc:
                if not self.continue_on_error:
                    raise
                failures.append({"chunk_id": chunk.chunk_id, "error": str(exc)})
                continue

            if not force and self.graph_store.is_chunk_current(
                tenant_id=tenant_id,
                chunk_id=scoped_chunk.chunk_id,
                content_hash=scoped_chunk.content_hash,
                extractor_version=self.extractor.extractor_version,
            ):
                skipped += 1
                continue

            try:
                extraction = self.extractor.extract(scoped_chunk)
                self.graph_store.replace_chunk_graph(
                    tenant_id=tenant_id,
                    document_id=scoped_chunk.document_id,
                    chunk_id=scoped_chunk.chunk_id,
                    content_hash=scoped_chunk.content_hash,
                    extractor_version=self.extractor.extractor_version,
                    entities=extraction.entities,
                    relations=extraction.relations,
                )
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                failures.append(
                    {"chunk_id": chunk.chunk_id, "error": str(exc)}
                )
                continue

            indexed += 1
            entity_count += len(extraction.entities)
            relation_count += len(extraction.relations)

        return GraphIndexingResult(
            indexed_chunks=indexed,
            skipped_chunks=skipped,
            entity_count=entity_count,
            relation_count=relation_count,
            failures=tuple(failures),
        )

    def index_documents(
        self,
        documents: DocumentLike | Sequence[DocumentLike],
        chunker: Chunker,
        **options: Any,
    ) -> GraphIndexingResult:
        return self.index_chunks(chunker.split(documents), **options)

    def delete_by_document(self, *, tenant_id: str, document_id: str) -> None:
        self.graph_store.delete_by_document(
            tenant_id=tenant_id,
            document_id=document_id,
        )

    def prune_stale_chunks(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids: Sequence[str],
    ) -> int:
        return self.graph_store.prune_stale_chunks(
            tenant_id=tenant_id,
            document_id=document_id,
            expected_chunk_ids=expected_chunk_ids,
        )


__all__ = ["GraphIndexer"]
