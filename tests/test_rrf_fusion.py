from __future__ import annotations

import unittest

from retrieval import RrfFusion
from retrieval.vector_store import VectorSearchResult


def _hit(point_id: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        id=point_id,
        score=score,
        payload={"chunk_id": point_id, "content": point_id},
    )


class RrfFusionTest(unittest.TestCase):
    def test_fuses_overlapping_routes_by_reciprocal_rank(self) -> None:
        fusion = RrfFusion(k=60)
        results = fusion.fuse(
            {
                "dense": [_hit("a", 0.91), _hit("b", 0.88), _hit("c", 0.80)],
                "keyword": [_hit("b", 8.2), _hit("d", 5.1), _hit("a", 3.0)],
            },
            limit=4,
            weights={"dense": 1.0, "keyword": 0.8},
        )

        self.assertEqual([item.id for item in results], ["b", "a", "c", "d"])
        self.assertEqual(results[0].retrieval_mode, "hybrid_rrf")
        self.assertEqual(results[0].route_ranks, {"dense": 2, "keyword": 1})
        self.assertAlmostEqual(results[0].route_scores["dense"], 0.88)
        self.assertAlmostEqual(results[0].route_scores["keyword"], 8.2)

    def test_limit_and_empty_routes(self) -> None:
        fusion = RrfFusion(k=10)
        self.assertEqual(fusion.fuse({}, limit=5), [])
        results = fusion.fuse(
            {"dense": [_hit("a", 1.0), _hit("b", 0.9)]},
            limit=1,
        )
        self.assertEqual([item.id for item in results], ["a"])


if __name__ == "__main__":
    unittest.main()
