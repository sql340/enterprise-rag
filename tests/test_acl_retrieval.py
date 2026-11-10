from __future__ import annotations

import unittest
import uuid

from qdrant_client import QdrantClient

from access_control import AccessControlError, Classification, DocumentAcl, Principal
from data_processing.chunker import Chunk
from data_processing.knowledge_indexing import BaseEmbedding, KnowledgeIndexer
from retrieval import KnowledgeRetriever
from retrieval.vector_store import QdrantVectorStore


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake-acl-embedding"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


class AclRetrievalIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantVectorStore(
            "acl_retrieval_test",
            client=QdrantClient(location=":memory:"),
            create_payload_indexes=False,
        )
        self.indexer = KnowledgeIndexer(
            FakeEmbedding(),
            self.store,
            sync_scope="document",
        )
        self.retriever = KnowledgeRetriever(
            FakeEmbedding(),
            self.store,
            retrieval_mode="dense",
            acl_enabled=True,
        )
        self.documents = self._build_documents()

    def _build_documents(self) -> dict[str, str]:
        documents: dict[str, str] = {}

        def index(
            name: str,
            acl: DocumentAcl,
            *,
            file_type: str = "markdown",
        ) -> None:
            document_id = f"doc-{name}"
            result = self._index_document(document_id, acl, file_type=file_type)
            documents[name] = result.document_ids[0]

        index(
            "public",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.PUBLIC,
                owner_id="public-owner",
            ),
        )
        index(
            "owner",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.INTERNAL,
                owner_id="u1",
            ),
        )
        index(
            "explicit",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.INTERNAL,
                allowed_users=frozenset({"u2"}),
                owner_id="owner-2",
                acl_version=2,
            ),
        )
        index(
            "role",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.INTERNAL,
                allowed_roles=frozenset({"finance"}),
                owner_id="owner-3",
            ),
        )
        index(
            "group",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.CONFIDENTIAL,
                allowed_groups=frozenset({"audit"}),
                owner_id="owner-4",
            ),
        )
        index(
            "secret",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.SECRET,
                owner_id="u-secret",
            ),
        )
        index(
            "archived",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.PUBLIC,
                owner_id="u1",
                document_status="archived",
            ),
        )
        index(
            "deleted",
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.INTERNAL,
                allowed_users=frozenset({"u2"}),
                owner_id="u2",
                document_status="deleted",
            ),
        )
        index(
            "other-tenant",
            DocumentAcl(
                tenant_id="tenant-b",
                classification=Classification.PUBLIC,
                owner_id="u2",
            ),
        )
        legacy = self._index_document("doc-legacy", None, tenant_id="tenant-a")
        documents["legacy"] = legacy.document_ids[0]
        return documents

    def _index_document(
        self,
        document_id: str,
        acl: DocumentAcl | None,
        *,
        tenant_id: str | None = None,
        file_type: str = "markdown",
    ):
        chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, document_id).hex
        content = f"policy {document_id}"
        chunk = Chunk(
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
                "file_type": file_type,
            },
        )
        return self.indexer.index_chunks(
            [chunk],
            acl=acl,
            tenant_id=acl.tenant_id if acl is not None else tenant_id,
        )

    def _visible_names(self, principal: Principal) -> set[str]:
        results = self.retriever.retrieve("policy", top_k=20, principal=principal)
        visible_document_ids = {result.document_id for result in results}
        return {
            name
            for name, document_id in self.documents.items()
            if document_id in visible_document_ids
        }

    def test_permission_matrix_is_enforced_in_qdrant(self) -> None:
        cases = {
            "public reader": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="public-reader",
                    clearance=Classification.PUBLIC,
                ),
                {"public"},
            ),
            "owner": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="u1",
                    clearance=Classification.INTERNAL,
                ),
                {"public", "owner"},
            ),
            "explicit user": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="u2",
                    clearance=Classification.INTERNAL,
                ),
                {"public", "explicit"},
            ),
            "role": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="finance-user",
                    roles={"finance"},
                    clearance=Classification.INTERNAL,
                ),
                {"public", "role"},
            ),
            "group": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="audit-user",
                    groups={"audit"},
                    clearance=Classification.CONFIDENTIAL,
                ),
                {"public", "group"},
            ),
            "secret clearance": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="u-secret",
                    clearance=Classification.SECRET,
                ),
                {"public", "secret"},
            ),
            "tenant admin": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="admin",
                    roles={"knowledge_admin"},
                    clearance=Classification.SECRET,
                ),
                {"public", "owner", "explicit", "role", "group", "secret"},
            ),
            "clearance is still required": (
                Principal(
                    tenant_id="tenant-a",
                    user_id="u2",
                    clearance=Classification.PUBLIC,
                ),
                {"public"},
            ),
            "other tenant": (
                Principal(
                    tenant_id="tenant-b",
                    user_id="u2",
                    clearance=Classification.SECRET,
                ),
                {"other-tenant"},
            ),
        }

        for name, (principal, expected) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self._visible_names(principal), expected)

    def test_acl_update_takes_effect_without_reindexing_vectors(self) -> None:
        store_count = self.store.count()
        self.indexer.update_acl_by_document(
            self.documents["explicit"],
            DocumentAcl(
                tenant_id="tenant-a",
                classification=Classification.INTERNAL,
                allowed_users=frozenset({"u3"}),
                owner_id="owner-2",
                acl_version=3,
            ),
        )

        u2 = Principal(
            tenant_id="tenant-a",
            user_id="u2",
            clearance=Classification.INTERNAL,
        )
        u3 = Principal(
            tenant_id="tenant-a",
            user_id="u3",
            clearance=Classification.INTERNAL,
        )

        self.assertEqual(self._visible_names(u2), {"public"})
        self.assertEqual(self._visible_names(u3), {"public", "explicit"})
        self.assertEqual(self.store.count(), store_count)

    def test_client_filters_are_anded_with_acl(self) -> None:
        u2 = Principal(
            tenant_id="tenant-a",
            user_id="u2",
            clearance=Classification.INTERNAL,
        )

        matching = self.retriever.retrieve(
            "policy",
            top_k=20,
            principal=u2,
            filters={"file_type": "markdown"},
        )
        forged_tenant = self.retriever.retrieve(
            "policy",
            top_k=20,
            principal=u2,
            filters={"tenant_id": "tenant-b"},
        )

        self.assertEqual(
            {result.document_id for result in matching},
            {self.documents["public"], self.documents["explicit"]},
        )
        self.assertEqual(forged_tenant, [])

    def test_acl_enabled_requires_principal(self) -> None:
        with self.assertRaises(AccessControlError):
            self.retriever.retrieve("policy", principal=None)


if __name__ == "__main__":
    unittest.main()
