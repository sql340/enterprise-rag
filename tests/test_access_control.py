from __future__ import annotations

import unittest

from pydantic import ValidationError

from access_control import (
    ACL_PAYLOAD_FIELDS,
    AccessControlError,
    Classification,
    DocumentAcl,
    Principal,
    PrincipalResolver,
)
from api.QueryRequest import QueryRequest


class ClassificationTest(unittest.TestCase):
    def test_parse_aliases_and_numeric_values(self) -> None:
        self.assertEqual(Classification.parse("PUBLIC"), Classification.PUBLIC)
        self.assertEqual(Classification.parse(" confidential "), Classification.CONFIDENTIAL)
        self.assertEqual(Classification.parse(3), Classification.SECRET)

    def test_reject_unknown_or_out_of_range_value(self) -> None:
        with self.assertRaises(ValueError):
            Classification.parse("restricted")
        with self.assertRaises(ValueError):
            Classification.parse(4)


class PrincipalTest(unittest.TestCase):
    def test_normalise_roles_and_groups(self) -> None:
        principal = Principal(
            tenant_id=" tenant-a ",
            user_id=" User-1 ",
            roles="Finance, Viewer",
            groups="Audit",
            clearance="internal",
        )

        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.user_id, "User-1")
        self.assertEqual(principal.roles, frozenset({"finance", "viewer"}))
        self.assertEqual(principal.groups, frozenset({"audit"}))
        self.assertEqual(principal.clearance, Classification.INTERNAL)

    def test_missing_identity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Principal(tenant_id="", user_id="u1")
        with self.assertRaises(ValueError):
            Principal(tenant_id="t1", user_id="")


class DocumentAclPolicyTest(unittest.TestCase):
    def _acl(self, **overrides) -> DocumentAcl:
        values = {
            "tenant_id": "tenant-a",
            "classification": Classification.INTERNAL,
            "owner_id": "owner",
            "acl_version": 2,
        }
        values.update(overrides)
        return DocumentAcl(**values)

    def test_public_document_requires_sufficient_clearance(self) -> None:
        acl = self._acl(classification=Classification.PUBLIC)
        public_user = Principal(
            tenant_id="tenant-a",
            user_id="reader",
            clearance=Classification.PUBLIC,
        )
        internal_user = Principal(
            tenant_id="tenant-a",
            user_id="reader",
            clearance=Classification.INTERNAL,
        )

        self.assertTrue(acl.allows(public_user))
        self.assertTrue(acl.allows(internal_user))
        self.assertFalse(acl.allows(Principal(tenant_id="tenant-b", user_id="reader")))

    def test_visibility_requires_an_allow_rule(self) -> None:
        acl = self._acl(
            allowed_users=frozenset({"explicit-user"}),
            allowed_roles=frozenset({"finance"}),
            allowed_groups=frozenset({"audit"}),
        )

        self.assertTrue(
            acl.allows(
                Principal(tenant_id="tenant-a", user_id="owner", clearance="internal")
            )
        )
        self.assertTrue(
            acl.allows(
                Principal(
                    tenant_id="tenant-a",
                    user_id="explicit-user",
                    clearance="internal",
                )
            )
        )
        self.assertTrue(
            acl.allows(
                Principal(
                    tenant_id="tenant-a",
                    user_id="finance-user",
                    roles={"finance"},
                    clearance="internal",
                )
            )
        )
        self.assertTrue(
            acl.allows(
                Principal(
                    tenant_id="tenant-a",
                    user_id="audit-user",
                    groups={"audit"},
                    clearance="internal",
                )
            )
        )
        self.assertFalse(
            acl.allows(
                Principal(tenant_id="tenant-a", user_id="unknown", clearance="internal")
            )
        )

    def test_classification_archived_and_tenant_are_hard_limits(self) -> None:
        acl = self._acl(classification=Classification.CONFIDENTIAL)
        internal_user = Principal(
            tenant_id="tenant-a",
            user_id="owner",
            clearance=Classification.INTERNAL,
        )
        confidential_user = Principal(
            tenant_id="tenant-a",
            user_id="owner",
            clearance=Classification.CONFIDENTIAL,
        )

        self.assertFalse(acl.allows(internal_user))
        self.assertTrue(acl.allows(confidential_user))
        self.assertFalse(
            self._acl(document_status="archived").allows(confidential_user)
        )
        self.assertFalse(
            self._acl(document_status="deleted").allows(confidential_user)
        )
        self.assertFalse(
            self._acl(tenant_id="tenant-b").allows(confidential_user)
        )

    def test_admin_still_obeys_tenant_clearance_and_status(self) -> None:
        acl = self._acl(classification=Classification.CONFIDENTIAL)
        admin = Principal(
            tenant_id="tenant-a",
            user_id="admin",
            roles={"knowledge_admin"},
            clearance=Classification.SECRET,
        )
        admin_without_clearance = Principal(
            tenant_id="tenant-a",
            user_id="admin",
            roles={"knowledge_admin"},
            clearance=Classification.INTERNAL,
        )

        self.assertTrue(acl.allows(admin))
        self.assertFalse(acl.allows(admin_without_clearance))
        self.assertFalse(self._acl(document_status="archived").allows(admin))
        self.assertFalse(self._acl(tenant_id="tenant-b").allows(admin))

    def test_payload_round_trip_and_signature(self) -> None:
        acl = self._acl(
            classification=Classification.CONFIDENTIAL,
            allowed_users=frozenset({"User-2"}),
            allowed_roles="Finance,Legal",
            allowed_groups=frozenset({"Audit"}),
        )

        restored = DocumentAcl.from_payload(acl.to_payload())

        self.assertEqual(restored, acl)
        self.assertEqual(
            DocumentAcl.payload_signature(acl.to_payload()),
            DocumentAcl.payload_signature(restored.to_payload()),
        )
        self.assertIsNone(DocumentAcl.payload_signature({"content": "legacy"}))


class PrincipalResolverTest(unittest.TestCase):
    def test_resolve_trusted_headers(self) -> None:
        resolver = PrincipalResolver(proxy_token="secret-token")

        principal = resolver.resolve(
            {
                "X-Rag-Proxy-Token": "secret-token",
                "X-Tenant-Id": "tenant-a",
                "X-User-Id": "User-1",
                "X-User-Roles": "Finance, Knowledge_Admin",
                "X-User-Groups": "Audit",
                "X-User-Clearance": "confidential",
                "X-User-Name": "Alice",
            }
        )

        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.user_id, "User-1")
        self.assertEqual(principal.roles, frozenset({"finance", "knowledge_admin"}))
        self.assertEqual(principal.groups, frozenset({"audit"}))
        self.assertEqual(principal.clearance, Classification.CONFIDENTIAL)
        self.assertEqual(principal.display_name, "Alice")

    def test_reject_missing_identity_or_invalid_proxy_token(self) -> None:
        resolver = PrincipalResolver(proxy_token="secret-token")

        with self.assertRaises(AccessControlError):
            resolver.resolve({"X-Rag-Proxy-Token": "wrong"})
        with self.assertRaises(AccessControlError):
            resolver.resolve(
                {
                    "X-Rag-Proxy-Token": "secret-token",
                    "X-Tenant-Id": "tenant-a",
                }
            )

    def test_reject_unknown_clearance(self) -> None:
        resolver = PrincipalResolver()

        with self.assertRaises(AccessControlError):
            resolver.resolve(
                {
                    "X-Tenant-Id": "tenant-a",
                    "X-User-Id": "user-1",
                    "X-User-Clearance": "top-secret",
                }
            )


class QueryRequestAclTest(unittest.TestCase):
    def test_client_filters_cannot_override_acl_fields(self) -> None:
        for field in ACL_PAYLOAD_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    QueryRequest(question="policy", filters={field: "forged"})

    def test_non_acl_filters_remain_supported(self) -> None:
        request = QueryRequest(
            question="policy",
            filters={"file_type": ["pdf", "markdown"], "chunk_role": "child"},
        )

        self.assertEqual(
            request.filters,
            {"file_type": ["pdf", "markdown"], "chunk_role": "child"},
        )


if __name__ == "__main__":
    unittest.main()
