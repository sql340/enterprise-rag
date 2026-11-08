from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .BaseGraphStore import BaseGraphStore
from .GraphEntity import GraphEntity
from .GraphRelation import GraphRelation
from .GraphSearchHit import GraphSearchHit
from ._utils import contains_entity, normalise_text


class SqliteGraphStore(BaseGraphStore):
    """SQLite 知识图谱存储，适合单机部署和后续替换为 Neo4j。"""

    def __init__(self, path: str | Path = ".cache/knowledge_graph.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    normalised_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (tenant_id, normalised_name)
                );

                CREATE TABLE IF NOT EXISTS graph_entity_aliases (
                    entity_id INTEGER NOT NULL,
                    alias TEXT NOT NULL,
                    normalised_alias TEXT NOT NULL,
                    PRIMARY KEY (entity_id, normalised_alias),
                    FOREIGN KEY (entity_id) REFERENCES graph_entities(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS graph_chunk_entities (
                    tenant_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    entity_id INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, chunk_id, entity_id),
                    FOREIGN KEY (entity_id) REFERENCES graph_entities(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS graph_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    source_entity_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    target_entity_id INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    UNIQUE (
                        tenant_id,
                        source_entity_id,
                        relation_type,
                        target_entity_id,
                        chunk_id
                    ),
                    FOREIGN KEY (source_entity_id) REFERENCES graph_entities(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (target_entity_id) REFERENCES graph_entities(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS graph_chunk_state (
                    tenant_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, chunk_id)
                );

                CREATE INDEX IF NOT EXISTS idx_graph_entities_tenant_name
                    ON graph_entities(tenant_id, normalised_name);
                CREATE INDEX IF NOT EXISTS idx_graph_chunk_entities_entity
                    ON graph_chunk_entities(tenant_id, entity_id);
                CREATE INDEX IF NOT EXISTS idx_graph_relations_source
                    ON graph_relations(tenant_id, source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_graph_relations_target
                    ON graph_relations(tenant_id, target_entity_id);
                CREATE INDEX IF NOT EXISTS idx_graph_state_document
                    ON graph_chunk_state(tenant_id, document_id);
                """
            )

    def is_chunk_current(
        self,
        *,
        tenant_id: str,
        chunk_id: str,
        content_hash: str,
        extractor_version: str,
    ) -> bool:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT content_hash, extractor_version
                FROM graph_chunk_state
                WHERE tenant_id = ? AND chunk_id = ?
                """,
                (tenant_id, chunk_id),
            ).fetchone()
        return bool(
            row
            and row["content_hash"] == content_hash
            and row["extractor_version"] == extractor_version
        )

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
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM graph_chunk_entities WHERE tenant_id = ? AND chunk_id = ?",
                (tenant_id, chunk_id),
            )
            self._connection.execute(
                "DELETE FROM graph_relations WHERE tenant_id = ? AND chunk_id = ?",
                (tenant_id, chunk_id),
            )

            entity_ids: dict[str, int] = {}
            for entity in entities:
                entity_id = self._upsert_entity(tenant_id, entity)
                entity_ids[entity.normalised_name] = entity_id
                for alias in entity.aliases:
                    entity_ids.setdefault(normalise_text(alias), entity_id)
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO graph_entity_aliases
                            (entity_id, alias, normalised_alias)
                        VALUES (?, ?, ?)
                        """,
                        (entity_id, alias, normalise_text(alias)),
                    )
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO graph_chunk_entities
                        (tenant_id, chunk_id, document_id, entity_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (tenant_id, chunk_id, document_id, entity_id),
                )

            for relation in relations:
                source_id = entity_ids.get(normalise_text(relation.source))
                target_id = entity_ids.get(normalise_text(relation.target))
                if source_id is None or target_id is None or source_id == target_id:
                    continue
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO graph_relations
                        (
                            tenant_id,
                            source_entity_id,
                            relation_type,
                            target_entity_id,
                            chunk_id,
                            document_id,
                            description
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        source_id,
                        relation.relation_type,
                        target_id,
                        chunk_id,
                        document_id,
                        relation.description,
                    ),
                )

            self._connection.execute(
                """
                INSERT INTO graph_chunk_state
                    (
                        tenant_id,
                        chunk_id,
                        document_id,
                        content_hash,
                        extractor_version,
                        updated_at
                    )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(tenant_id, chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    content_hash = excluded.content_hash,
                    extractor_version = excluded.extractor_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    tenant_id,
                    chunk_id,
                    document_id,
                    content_hash,
                    extractor_version,
                ),
            )

    def _upsert_entity(self, tenant_id: str, entity: GraphEntity) -> int:
        self._connection.execute(
            """
            INSERT INTO graph_entities
                (tenant_id, normalised_name, name, entity_type, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(tenant_id, normalised_name) DO UPDATE SET
                name = excluded.name,
                entity_type = excluded.entity_type,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                tenant_id,
                entity.normalised_name,
                entity.name,
                entity.entity_type,
            ),
        )
        row = self._connection.execute(
            """
            SELECT id FROM graph_entities
            WHERE tenant_id = ? AND normalised_name = ?
            """,
            (tenant_id, entity.normalised_name),
        ).fetchone()
        if row is None:  # pragma: no cover - 由 SQLite 约束保证
            raise RuntimeError("实体写入后无法读取")
        return int(row["id"])

    def search_chunk_ids(
        self,
        question: str,
        *,
        tenant_id: str,
        limit: int,
        max_hops: int,
        entity_limit: int,
    ) -> list[GraphSearchHit]:
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        if entity_limit <= 0:
            raise ValueError("entity_limit 必须为正整数")
        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops 必须在 1 到 3 之间")

        question_normalised = normalise_text(question)
        if not question_normalised:
            return []

        with self._lock:
            candidates = self._find_matching_entities(
                question_normalised,
                tenant_id=tenant_id,
                entity_limit=entity_limit,
            )
            if not candidates:
                return []

            chunk_scores: dict[str, float] = defaultdict(float)
            chunk_documents: dict[str, str] = {}
            chunk_entities: dict[str, set[str]] = defaultdict(set)
            self._collect_chunks(
                tenant_id=tenant_id,
                entity_weights={
                    entity_id: 1.0 + min(len(name) / 100, 0.5)
                    for entity_id, name in candidates.items()
                },
                entity_names={entity_id: name for entity_id, name in candidates.items()},
                chunk_scores=chunk_scores,
                chunk_documents=chunk_documents,
                chunk_entities=chunk_entities,
            )

            visited = set(candidates)
            frontier = {
                entity_id: 1.0 + min(len(name) / 100, 0.5)
                for entity_id, name in candidates.items()
            }
            for _hop in range(1, max_hops):
                neighbours = self._expand_entities(
                    tenant_id=tenant_id,
                    frontier=frontier,
                    visited=visited,
                    limit=entity_limit * 8,
                )
                if not neighbours:
                    break
                names = self._entity_names(tenant_id, neighbours)
                self._collect_chunks(
                    tenant_id=tenant_id,
                    entity_weights=neighbours,
                    entity_names=names,
                    chunk_scores=chunk_scores,
                    chunk_documents=chunk_documents,
                    chunk_entities=chunk_entities,
                )
                visited.update(neighbours)
                frontier = dict(neighbours)

        ranked = sorted(
            chunk_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
        return [
            GraphSearchHit(
                chunk_id=chunk_id,
                document_id=chunk_documents.get(chunk_id, ""),
                score=round(score, 8),
                matched_entities=tuple(sorted(chunk_entities[chunk_id])),
            )
            for chunk_id, score in ranked
        ]

    def _find_matching_entities(
        self,
        question: str,
        *,
        tenant_id: str,
        entity_limit: int,
    ) -> dict[int, str]:
        rows = self._connection.execute(
            """
            SELECT * FROM (
                SELECT e.id, e.name, e.normalised_name AS matched_name
                FROM graph_entities e
                WHERE e.tenant_id = ? AND instr(?, e.normalised_name) > 0
                UNION
                SELECT e.id, e.name, a.normalised_alias AS matched_name
                FROM graph_entity_aliases a
                JOIN graph_entities e ON e.id = a.entity_id
                WHERE e.tenant_id = ? AND instr(?, a.normalised_alias) > 0
            )
            ORDER BY LENGTH(matched_name) DESC
            LIMIT ?
            """,
            (tenant_id, question, tenant_id, question, entity_limit * 4),
        ).fetchall()

        matched: dict[int, tuple[str, str]] = {}
        for row in rows:
            matched_name = str(row["matched_name"])
            if not contains_entity(question, matched_name):
                continue
            entity_id = int(row["id"])
            current = matched.get(entity_id)
            if current is None or len(matched_name) > len(current[1]):
                matched[entity_id] = (str(row["name"]), matched_name)

        ordered = sorted(
            matched.items(),
            key=lambda item: (-len(item[1][1]), item[1][1]),
        )[:entity_limit]
        return {entity_id: value[0] for entity_id, value in ordered}

    def _collect_chunks(
        self,
        *,
        tenant_id: str,
        entity_weights: dict[int, float],
        entity_names: dict[int, str],
        chunk_scores: dict[str, float],
        chunk_documents: dict[str, str],
        chunk_entities: dict[str, set[str]],
    ) -> None:
        if not entity_weights:
            return
        placeholders = ",".join("?" for _ in entity_weights)
        rows = self._connection.execute(
            f"""
            SELECT chunk_id, document_id, entity_id
            FROM graph_chunk_entities
            WHERE tenant_id = ? AND entity_id IN ({placeholders})
            """,
            (tenant_id, *entity_weights),
        ).fetchall()
        for row in rows:
            entity_id = int(row["entity_id"])
            chunk_id = str(row["chunk_id"])
            chunk_scores[chunk_id] += entity_weights[entity_id]
            chunk_documents.setdefault(chunk_id, str(row["document_id"]))
            if entity_id in entity_names:
                chunk_entities[chunk_id].add(entity_names[entity_id])

    def _expand_entities(
        self,
        *,
        tenant_id: str,
        frontier: dict[int, float],
        visited: set[int],
        limit: int,
    ) -> dict[int, float]:
        if not frontier or limit <= 0:
            return {}
        placeholders = ",".join("?" for _ in frontier)
        rows = self._connection.execute(
            f"""
            SELECT source_entity_id, target_entity_id
            FROM graph_relations
            WHERE tenant_id = ?
              AND (
                  source_entity_id IN ({placeholders})
                  OR target_entity_id IN ({placeholders})
              )
            LIMIT ?
            """,
            (tenant_id, *frontier, *frontier, limit),
        ).fetchall()

        neighbours: dict[int, float] = {}
        for row in rows:
            source = int(row["source_entity_id"])
            target = int(row["target_entity_id"])
            if source in frontier and target not in visited:
                neighbours[target] = max(
                    neighbours.get(target, 0.0),
                    frontier[source] * 0.5,
                )
            if target in frontier and source not in visited:
                neighbours[source] = max(
                    neighbours.get(source, 0.0),
                    frontier[target] * 0.5,
                )
        return dict(sorted(neighbours.items(), key=lambda item: -item[1])[:limit])

    def _entity_names(self, tenant_id: str, entity_ids: dict[int, float]) -> dict[int, str]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self._connection.execute(
            f"""
            SELECT id, name FROM graph_entities
            WHERE tenant_id = ? AND id IN ({placeholders})
            """,
            (tenant_id, *entity_ids),
        ).fetchall()
        return {int(row["id"]): str(row["name"]) for row in rows}

    def delete_by_document(self, *, tenant_id: str, document_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM graph_relations WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            self._connection.execute(
                "DELETE FROM graph_chunk_entities WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            self._connection.execute(
                "DELETE FROM graph_chunk_state WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            self._cleanup_orphan_entities(tenant_id)

    def prune_stale_chunks(
        self,
        *,
        tenant_id: str,
        document_id: str,
        expected_chunk_ids: Sequence[str],
    ) -> int:
        expected = {str(chunk_id) for chunk_id in expected_chunk_ids if str(chunk_id)}
        if not expected:
            raise ValueError("expected_chunk_ids 不能为空，避免误删整个文档图谱")
        with self._lock, self._connection:
            rows = self._connection.execute(
                """
                SELECT chunk_id FROM graph_chunk_state
                WHERE tenant_id = ? AND document_id = ?
                """,
                (tenant_id, document_id),
            ).fetchall()
            stale = {str(row["chunk_id"]) for row in rows} - expected
            if not stale:
                return 0
            placeholders = ",".join("?" for _ in stale)
            parameters = (tenant_id, document_id, *stale)
            self._connection.execute(
                f"""
                DELETE FROM graph_relations
                WHERE tenant_id = ? AND document_id = ?
                  AND chunk_id IN ({placeholders})
                """,
                parameters,
            )
            self._connection.execute(
                f"""
                DELETE FROM graph_chunk_entities
                WHERE tenant_id = ? AND document_id = ?
                  AND chunk_id IN ({placeholders})
                """,
                parameters,
            )
            self._connection.execute(
                f"""
                DELETE FROM graph_chunk_state
                WHERE tenant_id = ? AND document_id = ?
                  AND chunk_id IN ({placeholders})
                """,
                parameters,
            )
            self._cleanup_orphan_entities(tenant_id)
        return len(stale)

    def _cleanup_orphan_entities(self, tenant_id: str) -> None:
        self._connection.execute(
            """
            DELETE FROM graph_entities
            WHERE tenant_id = ?
              AND id NOT IN (
                  SELECT source_entity_id FROM graph_relations WHERE tenant_id = ?
                  UNION
                  SELECT target_entity_id FROM graph_relations WHERE tenant_id = ?
                  UNION
                  SELECT entity_id FROM graph_chunk_entities WHERE tenant_id = ?
              )
            """,
            (tenant_id, tenant_id, tenant_id, tenant_id),
        )

    def count_entities(self, *, tenant_id: str | None = None) -> int:
        with self._lock:
            if tenant_id is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM graph_entities"
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM graph_entities WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
        return int(row["count"]) if row else 0

    def count_relations(self, *, tenant_id: str | None = None) -> int:
        with self._lock:
            if tenant_id is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM graph_relations"
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM graph_relations WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
        return int(row["count"]) if row else 0

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None  # type: ignore[assignment]


__all__ = ["SqliteGraphStore"]
