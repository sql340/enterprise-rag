from __future__ import annotations

from .BaseGraphStore import BaseGraphStore
from .GraphSearchHit import GraphSearchHit


class GraphRetriever:
    """从知识图谱召回与问题实体相关的知识块 ID。"""

    def __init__(
        self,
        graph_store: BaseGraphStore,
        *,
        max_hops: int = 1,
        entity_limit: int = 20,
    ) -> None:
        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops 必须在 1 到 3 之间")
        if entity_limit <= 0:
            raise ValueError("entity_limit 必须为正整数")
        self.graph_store = graph_store
        self.max_hops = max_hops
        self.entity_limit = entity_limit

    def retrieve(
        self,
        question: str,
        *,
        tenant_id: str,
        limit: int = 20,
        max_hops: int | None = None,
        entity_limit: int | None = None,
    ) -> list[GraphSearchHit]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 不能为空")
        if not str(tenant_id).strip():
            raise ValueError("tenant_id 不能为空")
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        return self.graph_store.search_chunk_ids(
            question.strip(),
            tenant_id=str(tenant_id).strip(),
            limit=limit,
            max_hops=self.max_hops if max_hops is None else max_hops,
            entity_limit=self.entity_limit if entity_limit is None else entity_limit,
        )

    def close(self) -> None:
        self.graph_store.close()


__all__ = ["GraphRetriever"]
