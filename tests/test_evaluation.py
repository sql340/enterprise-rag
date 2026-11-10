from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_processing.knowledge_indexing import BaseEmbedding
from evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationReportStore,
    EvaluationRunner,
    RetrievalMetrics,
)
from retrieval.vector_store import VectorSearchResult


class FakeEmbedding(BaseEmbedding):
    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 2

    def _embed_texts(self, texts, *, query):
        return [[1.0, 0.0] for _ in texts]


class FakeRetriever:
    def retrieve(self, question, *, top_k=5, retrieval_mode="dense", **kwargs):
        if "policy" in question:
            values = [
                VectorSearchResult("x", 0.7, {"chunk_id": "x", "document_id": "dx"}),
                VectorSearchResult("c1", 0.6, {"chunk_id": "c1", "document_id": "d1"}),
            ]
        else:
            values = [
                VectorSearchResult("d2", 0.9, {"chunk_id": "c9", "document_id": "d2"}),
            ]
        if retrieval_mode == "hybrid":
            values.reverse()
        return values[:top_k]


class EvaluationTest(unittest.TestCase):
    def test_retrieval_metrics(self) -> None:
        case = EvaluationCase(case_id="q1", question="q", expected_chunk_ids=("c1",))
        results = [
            VectorSearchResult("x", 0.9, {"chunk_id": "x"}),
            VectorSearchResult("c1", 0.8, {"chunk_id": "c1"}),
        ]

        metrics = RetrievalMetrics.calculate(case, results, k=2)

        self.assertAlmostEqual(metrics["recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.5)
        self.assertAlmostEqual(metrics["hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["ndcg_at_k"], 1 / 1.584962500721156)

    def test_dataset_jsonl_and_runner_modes(self) -> None:
        dataset = EvaluationDataset.from_json(
            json.dumps(
                {
                    "name": "smoke",
                    "cases": [
                        {
                            "case_id": "q1",
                            "question": "policy question",
                            "expected_chunk_ids": ["c1"],
                        },
                        {
                            "case_id": "q2",
                            "question": "other question",
                            "expected_document_ids": ["d2"],
                        },
                    ],
                }
            )
        )
        report = EvaluationRunner(FakeRetriever()).run(
            dataset,
            modes=("dense", "hybrid"),
            top_k=2,
        )

        self.assertEqual(report.modes, ("dense", "hybrid"))
        self.assertEqual(report.summary["dense"]["cases"], 2.0)
        self.assertGreater(report.summary["hybrid"]["mrr"], 0)
        self.assertEqual(len(report.cases), 4)

    def test_jsonl_dataset_and_report_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "case_id": "q1",
                                "question": "policy",
                                "expected_chunk_ids": ["c1"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "q2",
                                "question": "other",
                                "expected_document_ids": ["d2"],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            dataset = EvaluationDataset.from_file(path)
            report = EvaluationRunner(FakeRetriever()).run(
                dataset,
                modes=("dense",),
                top_k=2,
            )
            store = EvaluationReportStore()
            target = store.save(report, Path(temp_dir) / "report.json")
            payload = store.load(target)

        self.assertEqual(len(dataset.cases), 2)
        self.assertEqual(payload["dataset_name"], "cases")
        self.assertIn("summary", payload)


if __name__ == "__main__":
    unittest.main()
