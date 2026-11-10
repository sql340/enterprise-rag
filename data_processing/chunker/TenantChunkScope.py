from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .Chunk import Chunk


class TenantChunkScope:
    """把知识块 ID 收敛到租户作用域，避免跨租户覆盖。"""

    SCOPE_VERSION = "v1"

    @classmethod
    def apply(cls, chunks: Sequence[Chunk], tenant_id: str) -> list[Chunk]:
        tenant = str(tenant_id).strip()
        if not tenant:
            raise ValueError("tenant_id 不能为空")
        values = list(chunks)
        if not values:
            return []

        scoped_flags = [bool(chunk.metadata.get("tenant_scoped")) for chunk in values]
        if any(scoped_flags) and not all(scoped_flags):
            raise ValueError("不能混用已作用域和未作用域的知识块")
        if all(scoped_flags):
            cls._validate_existing_scope(values, tenant)
            return values

        for chunk in values:
            existing_tenant = str(chunk.metadata.get("tenant_id") or "").strip()
            if existing_tenant and existing_tenant != tenant:
                raise ValueError(
                    f"知识块 {chunk.chunk_id} 的 tenant_id 与目标租户不一致"
                )

        chunk_ids = {chunk.chunk_id for chunk in values}
        document_ids = {chunk.document_id for chunk in values}
        parent_ids = {
            chunk.parent_id for chunk in values if chunk.parent_id is not None
        }
        chunk_id_map = {
            value: cls._scoped_id(tenant, "chunk", value) for value in chunk_ids
        }
        document_id_map = {
            value: cls._scoped_id(tenant, "document", value)
            for value in document_ids
        }
        parent_id_map = {
            value: cls._scoped_id(tenant, "chunk", value) for value in parent_ids
        }

        scoped: list[Chunk] = []
        for chunk in values:
            parent_id = (
                parent_id_map.get(chunk.parent_id)
                if chunk.parent_id is not None
                else None
            )
            document_id = document_id_map[chunk.document_id]
            metadata = {
                **chunk.metadata,
                "tenant_id": tenant,
                "tenant_scoped": True,
                "tenant_scope_version": cls.SCOPE_VERSION,
                "document_id": document_id,
            }
            if parent_id is not None:
                metadata["parent_id"] = parent_id
            scoped.append(
                Chunk(
                    content=chunk.content,
                    chunk_id=chunk_id_map[chunk.chunk_id],
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    strategy=chunk.strategy,
                    metadata=metadata,
                    content_hash=chunk.content_hash,
                    parent_id=parent_id,
                )
            )
        return scoped

    @classmethod
    def _scoped_id(cls, tenant_id: str, kind: str, value: str) -> str:
        payload = f"tenant-scope:{cls.SCOPE_VERSION}\0{tenant_id}\0{kind}\0{value}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _validate_existing_scope(chunks: Sequence[Chunk], tenant_id: str) -> None:
        for chunk in chunks:
            current = str(chunk.metadata.get("tenant_id") or "").strip()
            if current != tenant_id:
                raise ValueError(
                    f"知识块 {chunk.chunk_id} 的 tenant_id 与目标租户不一致"
                )


__all__ = ["TenantChunkScope"]
