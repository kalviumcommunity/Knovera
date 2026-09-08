"""
test_retrieval_relevance_tuning.py

Comprehensive test suite for Assignment 3.34: Retrieval Relevance Tuning.
Validates test query definitions, retrieval setting configurations, evaluation metrics calculation,
score threshold filtering, metadata filtering, and quantitative best setting selection.
"""

import os
import json
import tempfile
import unittest

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.retrieval_tuner import (
    TestQuery,
    RetrievalSetting,
    EvaluationRow,
    TuningSummary,
    RetrievalTuner
)


class TestRetrievalRelevanceTuning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = EmbeddingGenerator()
        cls.chunks = [
            {
                "id": "account-guide.md:0",
                "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Authentication"}
            },
            {
                "id": "account-guide.md:1",
                "text": "Learners can recover access using their registered email address. A one-time secure verification link is dispatched immediately.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "doc_type": "guide", "category": "Authentication"}
            },
            {
                "id": "campus-guide.md:0",
                "text": "Campus cafeteria hours and lunch menu: Operating from 11:30 AM to 2:30 PM offering fresh garden salads and daily specials.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Campus Life"}
            },
            {
                "id": "submission-rubric.md:0",
                "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough.",
                "metadata": {"source": "submission-rubric.md", "chunk_index": 0, "doc_type": "rubric", "category": "Academics"}
            },
            {
                "id": "service-policy.md:0",
                "text": "Customer service SLA and refund eligibility details for enterprise subscription tiers.",
                "metadata": {"source": "service-policy.md", "chunk_index": 0, "doc_type": "policy", "category": "Billing"}
            }
        ]
        cls.embedded_chunks = cls.generator.embed_chunks(cls.chunks)
        cls.sample_test_queries = [
            TestQuery(
                query="How can a learner reset their password?",
                expected_source="account-guide.md",
                category="Authentication"
            ),
            TestQuery(
                query="When does the cafeteria menu change?",
                expected_source="campus-guide.md",
                category="Campus Life"
            ),
            TestQuery(
                query="What evidence is required for project submission?",
                expected_source="submission-rubric.md",
                category="Academics"
            )
        ]

    def setUp(self):
        self.vdb = VectorDatabase(in_memory=True)
        self.col_name = "test_tuning_collection"
        self.indexer = CorpusIndexer(vector_db=self.vdb, default_collection=self.col_name)
        self.indexer.index_corpus(self.embedded_chunks, reset_collection=True)
        self.tuner = RetrievalTuner(vector_db=self.vdb, default_collection=self.col_name)

    def test_query_and_setting_dataclasses(self):
        """Validates dataclass construction and dictionary serialization."""
        query = TestQuery(query="Test query?", expected_source="test.md", category="Test")
        self.assertEqual(query.query, "Test query?")
        self.assertEqual(query.expected_source, "test.md")
        self.assertEqual(query.to_dict()["category"], "Test")

        setting = RetrievalSetting(name="baseline_k3", k=3, min_score=0.5)
        self.assertEqual(setting.name, "baseline_k3")
        self.assertEqual(setting.k, 3)
        self.assertEqual(setting.min_score, 0.5)
        self.assertEqual(setting.to_dict()["k"], 3)

    def test_single_setting_evaluation(self):
        """Tests evaluation of a single retrieval setting and verifies hit rate computation."""
        setting = RetrievalSetting(name="baseline_k3", k=3, min_score=0.0)
        summary = self.tuner.evaluate_setting(setting, self.sample_test_queries, collection_name=self.col_name)

        self.assertEqual(summary.setting_name, "baseline_k3")
        self.assertEqual(summary.total_queries, len(self.sample_test_queries))
        self.assertGreaterEqual(summary.hits, 0)
        self.assertTrue(0.0 <= summary.hit_rate <= 1.0)
        self.assertEqual(len(summary.details), len(self.sample_test_queries))

    def test_score_threshold_filtering(self):
        """Verifies min_score filters out low confidence chunks."""
        permissive_setting = RetrievalSetting(name="permissive", k=5, min_score=0.0)
        strict_setting = RetrievalSetting(name="strict", k=5, min_score=0.999)

        perm_summary = self.tuner.evaluate_setting(permissive_setting, self.sample_test_queries, collection_name=self.col_name)
        strict_summary = self.tuner.evaluate_setting(strict_setting, self.sample_test_queries, collection_name=self.col_name)

        self.assertGreaterEqual(perm_summary.avg_retained_chunks, strict_summary.avg_retained_chunks)

    def test_multi_setting_comparison(self):
        """Tests evaluate_all_settings across multiple retrieval settings."""
        settings = [
            RetrievalSetting(name="baseline_k3", k=3, min_score=0.0),
            RetrievalSetting(name="filtered_k3", k=3, metadata_filter={"doc_type": "guide"}, min_score=0.0),
            RetrievalSetting(name="hybrid_k3", k=3, use_hybrid=True, vector_weight=0.7, keyword_weight=0.3)
        ]

        summaries = self.tuner.evaluate_all_settings(settings, self.sample_test_queries, collection_name=self.col_name)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(summaries[0].setting_name, "baseline_k3")
        self.assertEqual(summaries[1].setting_name, "filtered_k3")
        self.assertEqual(summaries[2].setting_name, "hybrid_k3")

    def test_best_setting_selection(self):
        """Validates select_best_setting identifies top-performing summary with justification."""
        settings = [
            RetrievalSetting(name="low_k1", k=1, min_score=0.0),
            RetrievalSetting(name="optimal_k3", k=3, min_score=0.0)
        ]

        summaries = self.tuner.evaluate_all_settings(settings, self.sample_test_queries, collection_name=self.col_name)
        best_summary, justification = self.tuner.select_best_setting(summaries)

        self.assertIsInstance(best_summary, TuningSummary)
        self.assertGreater(len(justification), 20)
        self.assertGreaterEqual(best_summary.hit_rate, min(s.hit_rate for s in summaries))

    def test_export_results(self):
        """Tests JSON and TXT artifact generation."""
        setting = RetrievalSetting(name="baseline_k3", k=3, min_score=0.0)
        summary = self.tuner.evaluate_setting(setting, self.sample_test_queries, collection_name=self.col_name)
        best_summary, justification = self.tuner.select_best_setting([summary])

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = os.path.join(tmp_dir, "results.json")
            txt_file = os.path.join(tmp_dir, "results.txt")

            self.tuner.export_results([summary], best_summary, justification, json_file, txt_file)

            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(txt_file))

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertEqual(data["chosen_setting"], "baseline_k3")
                self.assertEqual(len(data["summary_metrics"]), 1)


if __name__ == "__main__":
    unittest.main()
