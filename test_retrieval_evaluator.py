"""
test_retrieval_evaluator.py

Comprehensive test suite for Assignment 3.36: Retrieval Evaluation & Recall Testing.
Validates labelled query set handling, recall@k and precision@k metrics calculation,
multi-k comparison, failure inspection, and audit report generation.
"""

import os
import json
import tempfile
import unittest

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.corpus_indexer import CorpusIndexer
from src.retrieval_evaluator import (
    LabelledQuery,
    QueryEvaluationResult,
    AggregateMetrics,
    RetrievalEvaluator
)


class TestRetrievalEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = EmbeddingGenerator()
        cls.chunks = [
            {
                "id": "account-guide.md:0",
                "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "category": "Authentication"}
            },
            {
                "id": "account-guide.md:1",
                "text": "Password recovery email dispatch: Learners receive a single-use secure link to recover their registered account password.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "category": "Authentication"}
            },
            {
                "id": "submission-rubric.md:0",
                "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough.",
                "metadata": {"source": "submission-rubric.md", "chunk_index": 0, "category": "Academics"}
            },
            {
                "id": "campus-guide.md:0",
                "text": "Campus cafeteria hours and daily lunch menu rotation: Operating from 11:30 AM to 2:30 PM offering fresh garden salads.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 0, "category": "Campus Life"}
            },
            {
                "id": "distractor.md:0",
                "text": "Software engineering design patterns and clean architectural abstractions for Python development.",
                "metadata": {"source": "distractor.md", "chunk_index": 0, "category": "General"}
            }
        ]
        cls.embedded_chunks = cls.generator.embed_chunks(cls.chunks)
        cls.sample_labelled_queries = [
            LabelledQuery(
                query="How can a learner reset their password?",
                relevant_chunk_ids={"account-guide.md:0", "account-guide.md:1"},
                category="Authentication"
            ),
            LabelledQuery(
                query="What evidence is required for project submission?",
                relevant_chunk_ids={"submission-rubric.md:0"},
                category="Academics"
            ),
            LabelledQuery(
                query="When does the cafeteria menu change?",
                relevant_chunk_ids={"campus-guide.md:0"},
                category="Campus Life"
            )
        ]

    def setUp(self):
        self.vdb = VectorDatabase(in_memory=True)
        self.col_name = "test_eval_collection"
        self.indexer = CorpusIndexer(vector_db=self.vdb, default_collection=self.col_name)
        self.indexer.index_corpus(self.embedded_chunks, reset_collection=True)
        self.evaluator = RetrievalEvaluator(vector_db=self.vdb, default_collection=self.col_name)

    def test_labelled_query_and_metrics_dataclasses(self):
        """Validates dataclass instantiation and dictionary serialization."""
        item = LabelledQuery(
            query="How to reset password?",
            relevant_chunk_ids=["account.md:0", "account.md:1"],
            category="Auth"
        )
        self.assertEqual(item.query, "How to reset password?")
        self.assertIsInstance(item.relevant_chunk_ids, set)
        self.assertIn("account.md:0", item.relevant_chunk_ids)
        self.assertEqual(item.to_dict()["category"], "Auth")

        result = QueryEvaluationResult(
            query="Test query",
            retrieved_ids=["chunk_1", "chunk_2"],
            relevant_chunk_ids=["chunk_1"],
            hits=["chunk_1"],
            recall=1.0,
            precision=0.5,
            f1_score=0.6667,
            reciprocal_rank=1.0,
            hit_at_k=True,
            scores=[0.9, 0.4],
            sources=["doc1", "doc2"]
        )
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 0.5)
        self.assertTrue(result.to_dict()["hit_at_k"])

    def test_single_query_recall_precision_evaluation(self):
        """Tests evaluation of a single query and verifies recall and precision calculations."""
        query_item = {
            "query": "How can a learner reset their password?",
            "relevant_chunk_ids": {"account-guide.md:0", "account-guide.md:1"},
            "category": "Authentication"
        }

        result = self.evaluator.evaluate_query(query_item=query_item, k=5, collection_name=self.col_name)

        self.assertEqual(result.query, query_item["query"])
        self.assertGreater(len(result.retrieved_ids), 0)
        self.assertTrue("account-guide.md:0" in result.hits or "account-guide.md:1" in result.hits)
        self.assertGreater(result.recall, 0.0)
        self.assertTrue(0.0 <= result.precision <= 1.0)
        self.assertTrue(result.hit_at_k)
        self.assertGreater(result.reciprocal_rank, 0.0)

    def test_batch_query_evaluation_aggregate_metrics(self):
        """Tests evaluate_queries across batch dataset and verifies summary metrics."""
        metrics = self.evaluator.evaluate_queries(
            labelled_queries=self.sample_labelled_queries,
            k=5,
            collection_name=self.col_name
        )

        self.assertIsInstance(metrics, AggregateMetrics)
        self.assertEqual(metrics.total_queries, len(self.sample_labelled_queries))
        self.assertEqual(metrics.k, 5)
        self.assertTrue(0.0 <= metrics.avg_recall <= 1.0)
        self.assertTrue(0.0 <= metrics.avg_precision <= 1.0)
        self.assertGreater(metrics.hit_rate, 0.0)
        self.assertEqual(metrics.passed_queries_count + metrics.failed_queries_count, len(self.sample_labelled_queries))
        self.assertEqual(len(metrics.results), len(self.sample_labelled_queries))

    def test_evaluate_across_k_progression(self):
        """Verifies that recall increases or stays stable as top-k cutoff increases."""
        k_results = self.evaluator.evaluate_across_k(
            labelled_queries=self.sample_labelled_queries,
            k_values=[1, 3, 5],
            collection_name=self.col_name
        )

        self.assertEqual(set(k_results.keys()), {1, 3, 5})
        self.assertGreaterEqual(k_results[5].avg_recall, k_results[1].avg_recall)

    def test_failure_inspection_and_diagnosis(self):
        """Tests failure cause diagnosis when top-k is smaller than relevant chunk set."""
        multi_chunk_query = LabelledQuery(
            query="How can a learner reset their password?",
            relevant_chunk_ids={"account-guide.md:0", "account-guide.md:1"}
        )

        result_k1 = self.evaluator.evaluate_query(query_item=multi_chunk_query, k=1, collection_name=self.col_name)
        self.assertLess(result_k1.recall, 1.0)
        self.assertIsNotNone(result_k1.failure_cause)
        self.assertIn("TOO_SMALL_K", result_k1.failure_cause)

        metrics_k1 = self.evaluator.evaluate_queries([multi_chunk_query], k=1, collection_name=self.col_name)
        failures = metrics_k1.failure_analysis
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["query"], multi_chunk_query.query)
        self.assertIn("actionable_recommendation", failures[0])

    def test_export_results(self):
        """Tests JSON and TXT export functionality."""
        metrics = self.evaluator.evaluate_queries(self.sample_labelled_queries, k=5, collection_name=self.col_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = os.path.join(tmp_dir, "eval_results.json")
            txt_file = os.path.join(tmp_dir, "eval_audit.log")

            self.evaluator.export_results(metrics, json_file, txt_file)

            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(txt_file))

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertEqual(data["assignment"], "3.36 Retrieval Evaluation & Recall Testing")
                self.assertEqual(data["total_queries"], len(self.sample_labelled_queries))
                self.assertIn("summary_metrics", data)

            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("RETRIEVAL EVALUATION & RECALL TESTING", content)
                self.assertIn("Average Recall@5", content)


if __name__ == "__main__":
    unittest.main()
