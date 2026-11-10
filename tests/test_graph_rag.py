from __future__ import annotations

import unittest
import uuid

from qdrant_client import QdrantClient

from access_control import AccessControlError, Classification, DocumentAcl, Principal
from api.QueryRequest import QueryRequest
from configs import RagSettings
from data_processing.chunker import Chunk, TenantChunkScope
from data_processing.knowledge_indexing import BaseEmbedding
from graph_rag import (
    BaseGraphExtractor,
    GraphEntity,
    GraphExtraction,
    GraphExtractionError,
    GraphIndexer,
    GraphRelation,
    GraphRetriever,
    OpenAICompatibleGraphExtractor,
    SqliteGraphStore,
)
from retrieval import KnowledgeRetriever
from retrieval.vector_store import QdrantVectorStore


class FakeGraphExtractor(BaseGraphExtractor):
    @property
    def extractor_version(self) -> str:
        return "fake-graph-v1"

    def extract(self, chunk: Chunk) -> GraphExtraction:
        entities = [GraphEntity("Alice", "person")]
        if "Project Atlas" in chunk.content:
            entities.append(GraphEntity("Project Atlas", "project"))
        relations = ()
        if len(entities) == 2:
            relations = (GraphRelation("Alice", "leads", "Project Atlas"),)
        return GraphExtraction(entities=tuple(entities), relations=relations)


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-graph-embedding"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


def _chunk(
    *,
    document_id: str,
    content: str,
    tenant_id: str = "tenant-a",
    acl: DocumentAcl | None = None,
) -> Chunk:
    chunk_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{document_id}:{content}",
    ).hex
    resolved_acl = acl or DocumentAcl(tenant_id=tenant_id, owner_id="u1")
    return Chunk(
        content=content,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        char_start=0,
        char_end=len(content),
        strategy="recursive",
        content_hash=f"hash-{document_id}",
        metadata={
            "source": f"/knowledge/{document_id}.md",
            "file_type": "markdown",
            **resolved_acl.to_payload(),
        },
    )


class SqliteGraphStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SqliteGraphStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def _write(self, chunk_id: str, document_id: str, entities, relations=()):
        self.store.replace_chunk_graph(
            tenant_id="tenant-a",
            document_id=document_id,
            chunk_id=chunk_id,
            content_hash="hash",
            extractor_version="v1",
            entities=entities,
            relations=relations,
        )

    def test_entity_alias_and_hop_expansion(self) -> None:
        self._write(
            "c1",
            "d1",
            [
                GraphEntity("Alice", "person", aliases=("爱丽丝",)),
                GraphEntity("Project Atlas", "project"),
            ],
            [GraphRelation("Alice", "leads", "Project Atlas")],
        )
        self._write("c2", "d2", [GraphEntity("Project Atlas", "project")])

        direct = self.store.search_chunk_ids(
            "Alice 负责什么？",
            tenant_id="tenant-a",
            limit=10,
            max_hops=1,
            entity_limit=10,
        )
        expanded = self.store.search_chunk_ids(
            "Alice 负责什么？",
            tenant_id="tenant-a",
            limit=10,
            max_hops=2,
            entity_limit=10,
        )
        alias = self.store.search_chunk_ids(
            "爱丽丝负责什么？",
            tenant_id="tenant-a",
            limit=10,
            max_hops=1,
            entity_limit=10,
        )

        self.assertEqual({hit.chunk_id for hit in direct}, {"c1"})
        self.assertEqual({hit.chunk_id for hit in expanded}, {"c1", "c2"})
        self.assertEqual({hit.chunk_id for hit in alias}, {"c1"})

    def test_english_entity_does_not_match_word_fragment(self) -> None:
        self._write("c1", "d1", [GraphEntity("Alice", "person")])

        hits = self.store.search_chunk_ids(
            "Malice is unrelated",
            tenant_id="tenant-a",
            limit=10,
            max_hops=1,
            entity_limit=10,
        )

        self.assertEqual(hits, [])

    def test_tenant_isolation_and_document_deletion(self) -> None:
        self._write("c1", "d1", [GraphEntity("Alice", "person")])

        other_tenant = self.store.search_chunk_ids(
            "Alice",
            tenant_id="tenant-b",
            limit=10,
            max_hops=1,
            entity_limit=10,
        )
        self.assertEqual(other_tenant, [])
        self.assertEqual(self.store.count_entities(tenant_id="tenant-a"), 1)

        self.store.delete_by_document(tenant_id="tenant-a", document_id="d1")

        self.assertEqual(self.store.count_entities(tenant_id="tenant-a"), 0)
        self.assertEqual(self.store.count_relations(tenant_id="tenant-a"), 0)
        self.assertEqual(
            self.store.search_chunk_ids(
                "Alice",
                tenant_id="tenant-a",
                limit=10,
                max_hops=1,
                entity_limit=10,
            ),
            [],
        )

    def test_chunk_state_supports_incremental_skip(self) -> None:
        self.assertFalse(
            self.store.is_chunk_current(
                tenant_id="tenant-a",
                chunk_id="c1",
                content_hash="hash",
                extractor_version="v1",
            )
        )

    def test_prune_stale_chunks_keeps_expected_blocks(self) -> None:
        self._write("c1", "d1", [GraphEntity("Alice", "person")])
        self._write("c2", "d1", [GraphEntity("Project Atlas", "project")])

        deleted = self.store.prune_stale_chunks(
            tenant_id="tenant-a",
            document_id="d1",
            expected_chunk_ids=["c1"],
        )

        self.assertEqual(deleted, 1)
        self.assertEqual(
            {hit.chunk_id for hit in self.store.search_chunk_ids(
                "Alice",
                tenant_id="tenant-a",
                limit=10,
                max_hops=1,
                entity_limit=10,
            )},
            {"c1"},
        )
        self.assertEqual(self.store.count_entities(tenant_id="tenant-a"), 1)
        self._write("c1", "d1", [GraphEntity("Alice", "person")])

        self.assertTrue(
            self.store.is_chunk_current(
                tenant_id="tenant-a",
                chunk_id="c1",
                content_hash="hash",
                extractor_version="v1",
            )
        )
        self.assertFalse(
            self.store.is_chunk_current(
                tenant_id="tenant-a",
                chunk_id="c1",
                content_hash="changed",
                extractor_version="v1",
            )
        )


class GraphIndexerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SqliteGraphStore(":memory:")
        self.indexer = GraphIndexer(self.store, FakeGraphExtractor())

    def tearDown(self) -> None:
        self.store.close()

    def test_unchanged_chunks_are_skipped(self) -> None:
        chunks = [
            _chunk(document_id="d1", content="Alice leads Project Atlas"),
            _chunk(document_id="d2", content="Alice writes docs"),
        ]

        first = self.indexer.index_chunks(chunks)
        second = self.indexer.index_chunks(chunks)

        self.assertEqual(first.indexed_chunks, 2)
        self.assertEqual(first.skipped_chunks, 0)
        self.assertEqual(first.entity_count, 3)
        self.assertEqual(first.relation_count, 1)
        self.assertEqual(second.indexed_chunks, 0)
        self.assertEqual(second.skipped_chunks, 2)

    def test_missing_tenant_is_reported(self) -> None:
        chunk = _chunk(document_id="d1", content="Alice")
        chunk.metadata.pop("tenant_id")

        result = self.indexer.index_chunks([chunk])

        self.assertEqual(result.indexed_chunks, 0)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("tenant_id", result.failures[0]["error"])

    def test_explicit_acl_supplies_tenant_for_graph_build(self) -> None:
        chunk = _chunk(document_id="d1", content="Alice")
        chunk.metadata.pop("tenant_id")
        acl = DocumentAcl(tenant_id="tenant-a", owner_id="u1")

        result = self.indexer.index_chunks([chunk], acl=acl)

        self.assertEqual(result.indexed_chunks, 1)
        self.assertEqual(result.failures, ())


class OpenAICompatibleGraphExtractorTest(unittest.TestCase):
    def test_parse_fenced_json_and_missing_endpoint_entity(self) -> None:
        extractor = OpenAICompatibleGraphExtractor("test-model")

        extraction = extractor.parse_response(
            """```json
            {
              "entities": [{"name": "Alice", "type": "person", "aliases": ["爱丽丝"]}],
              "relations": [{"source": "Alice", "relation": "leads", "target": "Project Atlas"}]
            }
            ```"""
        )

        self.assertEqual({entity.name for entity in extraction.entities}, {"Alice", "Project Atlas"})
        self.assertEqual(len(extraction.relations), 1)

    def test_reject_non_json_response(self) -> None:
        extractor = OpenAICompatibleGraphExtractor("test-model")

        with self.assertRaises(GraphExtractionError):
            extractor.parse_response("not json")


class GraphRagIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vector_store = QdrantVectorStore(
            "graph_rag_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
            enable_text_index=False,
        )
        self.graph_store = SqliteGraphStore(":memory:")
        self.graph_retriever = GraphRetriever(self.graph_store, max_hops=1)
        self.acl = DocumentAcl(
            tenant_id="tenant-a",
            classification=Classification.INTERNAL,
            owner_id="u1",
        )
        self.chunks = [
            _chunk(
                document_id="d1",
                content="Alice leads Project Atlas",
                acl=self.acl,
            ),
            _chunk(
                document_id="d2",
                content="Project Atlas budget is 100",
                acl=self.acl,
            ),
            _chunk(
                document_id="d3",
                content="Alice private notes",
                acl=DocumentAcl(
                    tenant_id="tenant-a",
                    classification=Classification.INTERNAL,
                    allowed_users=frozenset({"u2"}),
                    owner_id="u2",
                ),
            ),
        ]
        self.scoped_chunks = TenantChunkScope.apply(self.chunks, "tenant-a")
        self.document_ids = {
            scoped.document_id: original.document_id
            for scoped, original in zip(
                self.scoped_chunks,
                self.chunks,
                strict=True,
            )
        }
        self.vector_store.upsert_chunks(
            self.scoped_chunks,
            [[1.0, 0.0] for _ in self.scoped_chunks],
            embedding_model="fake-graph-embedding",
        )
        GraphIndexer(self.graph_store, FakeGraphExtractor()).index_chunks(
            self.scoped_chunks
        )
        self.retriever = KnowledgeRetriever(
            FakeEmbedding(),
            self.vector_store,
            retrieval_mode="hybrid_graph",
            graph_retriever=self.graph_retriever,
            acl_enabled=True,
        )
        self.principal = Principal(
            tenant_id="tenant-a",
            user_id="u1",
            clearance=Classification.INTERNAL,
        )

    def tearDown(self) -> None:
        self.retriever.close()

    def _logical_document_ids(self, results) -> set[str]:
        return {
            self.document_ids.get(result.document_id, result.document_id)
            for result in results
        }

    def test_graph_route_excludes_unauthorized_chunks(self) -> None:
        results = self.retriever.retrieve(
            "Alice",
            top_k=10,
            retrieval_mode="graph",
            principal=self.principal,
        )

        self.assertEqual(
            self._logical_document_ids(results),
            {"d1", "d2"},
        )
        self.assertTrue(all("graph" in result.route_scores for result in results))

    def test_hybrid_graph_fuses_graph_with_original_rag(self) -> None:
        results = self.retriever.retrieve(
            "Alice",
            top_k=10,
            principal=self.principal,
        )

        self.assertEqual(
            self._logical_document_ids(results),
            {"d1", "d2"},
        )
        self.assertTrue(all(result.retrieval_mode == "hybrid_rrf" for result in results))
        self.assertTrue(all("dense" in result.route_scores for result in results))
        self.assertTrue(all("graph" in result.route_scores for result in results))

    def test_graph_route_expands_candidates_before_acl_limits_results(self) -> None:
        private_acl = DocumentAcl(
            tenant_id="tenant-a",
            classification=Classification.INTERNAL,
            allowed_users=frozenset({"u2"}),
            owner_id="u2",
        )
        private_chunks = [
            _chunk(
                document_id=f"private-{index}",
                content=f"Alice leads Project Atlas {index}",
                acl=private_acl,
            )
            for index in range(8)
        ]
        scoped_private_chunks = TenantChunkScope.apply(private_chunks, "tenant-a")
        self.vector_store.upsert_chunks(
            scoped_private_chunks,
            [[1.0, 0.0] for _ in scoped_private_chunks],
            embedding_model="fake-graph-embedding",
        )
        GraphIndexer(
            self.graph_store,
            FakeGraphExtractor(),
        ).index_chunks(scoped_private_chunks)

        results = self.retriever.retrieve(
            "Alice Project Atlas",
            top_k=1,
            retrieval_mode="graph",
            principal=self.principal,
        )

        self.assertEqual(len(results), 1)
        self.assertIn(
            self.document_ids.get(results[0].document_id, results[0].document_id),
            {"d1", "d2"},
        )
        self.assertEqual(results[0].route_ranks["graph"], 1)

    def test_graph_route_requires_principal(self) -> None:
        with self.assertRaises(AccessControlError):
            self.retriever.retrieve(
                "Alice",
                top_k=10,
                retrieval_mode="graph",
            )

    def test_query_request_accepts_hybrid_graph_mode(self) -> None:
        request = QueryRequest(question="Alice", retrieval_mode="hybrid_graph")

        self.assertEqual(request.retrieval_mode, "hybrid_graph")


class RagSettingsGraphValidationTest(unittest.TestCase):
    def test_graph_mode_can_disable_dense_and_keyword_weights(self) -> None:
        settings = RagSettings(
            rag_retrieval_mode="graph",
            rag_dense_weight=0,
            rag_keyword_weight=0,
            rag_graph_weight=1.0,
        )

        self.assertEqual(settings.rag_retrieval_mode, "graph")

    def test_hybrid_graph_rejects_all_zero_weights(self) -> None:
        with self.assertRaises(ValueError):
            RagSettings(
                rag_retrieval_mode="hybrid_graph",
                rag_dense_weight=0,
                rag_keyword_weight=0,
                rag_graph_weight=0,
            )

    def test_graph_mode_requires_graph_enabled(self) -> None:
        with self.assertRaises(ValueError):
            RagSettings(
                rag_retrieval_mode="graph",
                rag_graph_enabled=False,
            )


if __name__ == "__main__":
    unittest.main()
