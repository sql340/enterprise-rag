from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any

from data_processing.chunker import Chunk

from .BaseVectorStore import BaseVectorStore
from .VectorSearchResult import VectorSearchResult

try:
    from qdrant_client import QdrantClient, models
except ImportError as exc:  # pragma: no cover - 由构造时给出安装提示
    QdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    _QDRANT_IMPORT_ERROR: ImportError | None = exc
else:
    _QDRANT_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """向量库操作失败。"""


class QdrantDependencyError(VectorStoreError, ImportError):
    """缺少 qdrant-client 依赖。"""


class CollectionDimensionMismatchError(VectorStoreError, ValueError):
    """collection 的向量维度与当前模型不一致。"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


class QdrantVectorStore(BaseVectorStore):
    """基于 Qdrant 的向量存储实现。

    Args:
        collection_name: collection 名称。切换 Embedding 模型时应使用新名称。
        url: Qdrant HTTP 地址；传 ``:memory:`` 可使用进程内测试实例。
        api_key: Qdrant Cloud 或开启鉴权时的 API Key。
        path: 本地持久化目录；设置后优先于 ``url``。
        client: 已创建的 QdrantClient，便于测试或复用连接。
        prefer_grpc: 是否优先使用 gRPC。
        timeout: 请求超时秒数。
        distance: ``cosine``、``dot``、``euclid`` 或 ``manhattan``。
        on_disk_payload: 是否将 payload 存储在磁盘上。
        create_payload_indexes: 是否创建常用过滤字段索引。
        batch_size: 单次 upsert 的点数量。
        text_index_field: 开启 BM25 评分的 payload 文本字段。
        enable_text_index: 是否创建 BM25 文本索引。
    """

    DEFAULT_PAYLOAD_INDEXES = (
        "document_id",
        "source",
        "file_name",
        "file_type",
        "chunk_strategy",
        "chunk_role",
        "embedding_model",
    )

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        path: str | Path | None = None,
        client: Any = None,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
        distance: str = "cosine",
        on_disk_payload: bool = True,
        create_payload_indexes: bool = True,
        batch_size: int = 128,
        payload_index_fields: Sequence[str] | None = None,
        text_index_field: str = "content",
        enable_text_index: bool = True,
        **client_kwargs: Any,
    ) -> None:
        if QdrantClient is None or models is None:
            raise QdrantDependencyError(
                "未安装 qdrant-client，请执行: pip install qdrant-client"
            ) from _QDRANT_IMPORT_ERROR
        if not collection_name.strip():
            raise ValueError("collection_name 不能为空")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if timeout <= 0:
            raise ValueError("timeout 必须为正数")
        if not text_index_field.strip():
            raise ValueError("text_index_field 不能为空")

        self.collection_name = collection_name
        self.distance = self._resolve_distance(distance)
        self.batch_size = batch_size
        self.on_disk_payload = on_disk_payload
        self.create_payload_indexes = create_payload_indexes
        self.payload_index_fields = tuple(
            payload_index_fields or self.DEFAULT_PAYLOAD_INDEXES
        )
        self.text_index_field = text_index_field
        self.enable_text_index = enable_text_index
        self._text_index_available = enable_text_index
        self._owns_client = client is None
        self._dimension: int | None = None
        self._collection_verified = False
        self._payload_indexes_ready = False
        self._closed = False

        if client is not None:
            self.client = client
        elif path is not None:
            self.client = QdrantClient(path=str(Path(path).expanduser()))
        elif url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
                **client_kwargs,
            )

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def supports_text_search(self) -> bool:
        return self._text_index_available

    @staticmethod
    def _resolve_distance(distance: str) -> Any:
        values = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
            "euclidean": models.Distance.EUCLID,
            "manhattan": models.Distance.MANHATTAN,
        }
        try:
            return values[distance.strip().lower()]
        except KeyError as exc:
            raise ValueError(
                "distance 必须是 cosine、dot、euclid 或 manhattan"
            ) from exc

    def ensure_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        if self._collection_verified and self._dimension == dimension:
            self._ensure_payload_indexes()
            return

        exists = self.client.collection_exists(self.collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=self.distance,
                ),
                on_disk_payload=self.on_disk_payload,
            )
        else:
            info = self.client.get_collection(self.collection_name)
            current_dimension = self._collection_dimension(info)
            if current_dimension is not None and current_dimension != dimension:
                raise CollectionDimensionMismatchError(
                    f"collection {self.collection_name!r} 的向量维度为 "
                    f"{current_dimension}，当前模型输出维度为 {dimension}；"
                    "请为不同模型使用新的 collection 名称"
                )

        self._dimension = dimension
        self._collection_verified = True
        self._ensure_payload_indexes()

    @staticmethod
    def _collection_dimension(info: Any) -> int | None:
        vectors = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        if isinstance(vectors, dict):
            if len(vectors) != 1:
                raise VectorStoreError(
                    "当前实现只支持单个未命名向量字段的 collection"
                )
            vectors = next(iter(vectors.values()))
        size = getattr(vectors, "size", None)
        return int(size) if size is not None else None

    def _ensure_payload_indexes(self) -> None:
        if self._payload_indexes_ready or not self.create_payload_indexes:
            return
        for field_name in self.payload_index_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        if self.enable_text_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=self.text_index_field,
                    field_schema=models.TextIndexParams(
                        type=models.TextIndexType.TEXT,
                        tokenizer=models.TokenizerType.MULTILINGUAL,
                        lowercase=True,
                        scoring=True,
                    ),
                    wait=True,
                )
            except Exception as exc:
                self._text_index_available = False
                logger.warning(
                    "创建 Qdrant BM25 文本索引失败，hybrid 将回退 dense: %s",
                    exc,
                )
        self._payload_indexes_ready = True

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> int:
        chunk_values = list(chunks)
        vector_values = [list(map(float, vector)) for vector in vectors]
        if len(chunk_values) != len(vector_values):
            raise ValueError(
                f"chunks 与 vectors 数量不一致: {len(chunk_values)} != "
                f"{len(vector_values)}"
            )
        if not chunk_values:
            return 0

        dimension = len(vector_values[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in vector_values):
            raise ValueError("所有向量必须具有相同且非零的维度")
        self.ensure_collection(dimension)

        points = [
            models.PointStruct(
                id=self.point_id_for_chunk(chunk.chunk_id),
                vector=vector,
                payload=self._chunk_payload(chunk, embedding_model),
            )
            for chunk, vector in zip(chunk_values, vector_values, strict=True)
        ]
        for start in range(0, len(points), self.batch_size):
            batch = points[start : start + self.batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"写入 Qdrant collection {self.collection_name!r} 失败: {exc}"
                ) from exc
        return len(points)

    @classmethod
    def point_id_for_chunk(cls, chunk_id: str) -> str:
        """把 Chunk 的 32 位 hex ID 转为 Qdrant 支持的 UUID。"""

        try:
            return str(uuid.UUID(str(chunk_id)))
        except (ValueError, AttributeError, TypeError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"enterprise-rag:{chunk_id}"))

    def _chunk_payload(self, chunk: Chunk, embedding_model: str) -> dict[str, Any]:
        payload = _json_safe(dict(chunk.metadata))
        if not isinstance(payload, dict):
            payload = {}
        payload.update(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "chunk_strategy": chunk.strategy,
                "chunk_role": chunk.chunk_role,
                "content_hash": chunk.content_hash,
                "parent_id": chunk.parent_id,
                "embedding_model": embedding_model,
            }
        )
        return payload

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
        chunk_values = list(chunks)
        if not chunk_values:
            return 0

        scope_field, scope_value = self._scope_identity(chunk_values, scope)
        indexed = self.upsert_chunks(
            chunk_values,
            vectors,
            embedding_model=embedding_model,
        )
        if not delete_stale:
            return indexed

        active_chunk_ids = (
            list(expected_chunk_ids)
            if expected_chunk_ids is not None
            else [chunk.chunk_id for chunk in chunk_values]
        )
        self.prune_stale_chunks(
            scope_field=scope_field,
            scope_value=scope_value,
            expected_chunk_ids=active_chunk_ids,
        )
        return indexed

    @staticmethod
    def _scope_identity(chunks: Sequence[Chunk], scope: str) -> tuple[str, str]:
        normalised = scope.strip().lower()
        if normalised in {"document", "document_id"}:
            values = {chunk.document_id for chunk in chunks}
            field_name = "document_id"
        elif normalised in {"source", "file"}:
            if any(not chunk.metadata.get("source") for chunk in chunks):
                raise ValueError("按 source 同步时，所有 chunks 都必须包含 source")
            values = {str(chunk.metadata["source"]) for chunk in chunks}
            field_name = "source"
        else:
            raise ValueError("scope 必须是 document 或 source")
        if not values:
            raise ValueError(f"chunks 缺少可用于 {normalised} 同步的标识")
        if len(values) != 1:
            raise ValueError(
                f"一次 sync_chunks 只能同步一个 {normalised} 范围，"
                f"实际包含 {len(values)} 个"
            )
        return field_name, next(iter(values))

    def _scroll_chunk_ids(self, field_name: str, value: str) -> set[str]:
        return set(self.list_chunk_records(scope_field=field_name, scope_value=value))

    def list_chunk_records(
        self,
        *,
        scope_field: str,
        scope_value: str,
    ) -> dict[str, dict[str, Any]]:
        if not scope_field.strip() or not scope_value:
            raise ValueError("scope_field 和 scope_value 不能为空")
        if not self.client.collection_exists(self.collection_name):
            return {}

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=scope_field,
                    match=models.MatchValue(value=scope_value),
                )
            ]
        )
        records_by_chunk_id: dict[str, dict[str, Any]] = {}
        offset: Any = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=["chunk_id", "embedding_model", "content_hash"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                if payload.get("chunk_id"):
                    records_by_chunk_id[str(payload["chunk_id"])] = dict(payload)
            if offset is None:
                return records_by_chunk_id

    def prune_stale_chunks(
        self,
        *,
        scope_field: str,
        scope_value: str,
        expected_chunk_ids: Sequence[str],
    ) -> int:
        expected = {str(chunk_id) for chunk_id in expected_chunk_ids}
        if not expected:
            raise ValueError("expected_chunk_ids 不能为空，避免误删整个同步范围")
        existing = set(
            self.list_chunk_records(
                scope_field=scope_field,
                scope_value=scope_value,
            )
        )
        stale_chunk_ids = existing - expected
        if not stale_chunk_ids:
            return 0

        stale_point_ids = [
            self.point_id_for_chunk(chunk_id)
            for chunk_id in sorted(stale_chunk_ids)
        ]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=stale_point_ids,
            wait=True,
        )
        return len(stale_point_ids)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, Any] | Any | None = None,
    ) -> list[VectorSearchResult]:
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        vector = list(map(float, query_vector))
        if not vector:
            raise ValueError("query_vector 不能为空")
        if not self.client.collection_exists(self.collection_name):
            raise VectorStoreError(
                f"Qdrant collection 不存在: {self.collection_name!r}"
            )
        current_dimension = self._dimension or self._load_collection_dimension()
        if current_dimension is not None and len(vector) != current_dimension:
            raise CollectionDimensionMismatchError(
                f"查询向量维度 {len(vector)} 与 collection 维度 "
                f"{current_dimension} 不一致"
            )

        query_filter = self._build_filter(filters)
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                points = response.points
            else:  # pragma: no cover - 兼容旧版 qdrant-client
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant 检索 {self.collection_name!r} 失败: {exc}"
            ) from exc

        return [
            VectorSearchResult(
                id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in points
        ]

    def text_search(
        self,
        query_text: str,
        *,
        limit: int = 5,
        filters: Mapping[str, Any] | Any | None = None,
    ) -> list[VectorSearchResult]:
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        if not self._text_index_available:
            raise NotImplementedError("Qdrant 文本索引未启用")
        if not self.client.collection_exists(self.collection_name):
            raise VectorStoreError(
                f"Qdrant collection 不存在: {self.collection_name!r}"
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=models.TextQuery(text=query_text.strip()),
                using=self.text_index_field,
                limit=limit,
                query_filter=self._build_filter(filters),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant BM25 检索 {self.collection_name!r} 失败: {exc}"
            ) from exc

        return [
            VectorSearchResult(
                id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
                retrieval_mode="keyword",
                route_scores={"keyword": float(point.score)},
            )
            for point in response.points
        ]

    def _load_collection_dimension(self) -> int | None:
        info = self.client.get_collection(self.collection_name)
        self._dimension = self._collection_dimension(info)
        self._collection_verified = True
        return self._dimension

    def _build_filter(self, filters: Mapping[str, Any] | Any | None) -> Any:
        if filters is None:
            return None
        if isinstance(filters, models.Filter):
            return filters
        if not isinstance(filters, Mapping):
            raise TypeError("filters 必须是字典或 qdrant_client.models.Filter")

        conditions: list[Any] = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                values = list(value)
                if not values:
                    raise ValueError(f"过滤字段 {key} 的候选值不能为空")
                conditions.append(
                    models.FieldCondition(
                        key=str(key),
                        match=models.MatchAny(any=values),
                    )
                )
            elif isinstance(value, Mapping):
                raise ValueError(
                    f"过滤字段 {key} 不支持字典值；复杂条件请传入 Qdrant Filter"
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=str(key),
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions) if conditions else None

    def delete_by_document(self, document_id: str) -> None:
        if not document_id:
            raise ValueError("document_id 不能为空")
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )

    def delete_by_source(self, source: str) -> None:
        if not source:
            raise ValueError("source 不能为空")
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source),
                    )
                ]
            ),
            wait=True,
        )

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(result.count)

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_client:
            self.client.close()
        self._closed = True

    def __enter__(self) -> "QdrantVectorStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


__all__ = [
    "CollectionDimensionMismatchError",
    "QdrantDependencyError",
    "QdrantVectorStore",
    "VectorStoreError",
]
