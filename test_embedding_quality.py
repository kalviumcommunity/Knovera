"""
test_embedding_quality.py

Comprehensive Unit Test Suite for Assignment 3.29: Embedding Quality Checks & Sanity Tests.
Tests:
- Known query-chunk relevance ranking
- Confirmation that related chunks rank above unrelated chunks
- Positive score margin evaluation
- Detection and diagnostic explanation of edge/surprising cases
- Detection of model mismatch vector corruption
- Full sanity report generation metrics
"""

import os
import unittest
from typing import List, Dict, Any

from src.embedding_generator import EmbeddingGenerator, cosine_similarity
from src.embedding_quality_checker import EmbeddingQualityChecker


class TestEmbeddingQualityChecks(unittest.TestCase):

    def setUp(self):
        self.generator = EmbeddingGenerator()
        self.checker = EmbeddingQualityChecker(generator=self.generator)

        self.corpus = [
            {
                "id": "account-guide#chunk_0",
                "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Recovery"}
            },
            {
                "id": "account-guide#chunk_1",
                "text": "Learners can recover access using their registered email. A one-time verification link is dispatched.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Verification"}
            },
            {
                "id": "service-policy#chunk_0",
                "text": "Knovera Customer Service & Refund Policy: Enterprise clients are eligible for full refunds within 14 days.",
                "metadata": {"source": "service-policy.md", "chunk_index": 0, "section": "Refund Eligibility"}
            },
            {
                "id": "campus-guide#chunk_0",
                "text": "Campus cafeteria hours and lunch menu: The cafeteria operates from 11:30 AM to 2:30 PM offering Italian pasta and fresh salads.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Dining Hours"}
            },
            {
                "id": "dev-guide#chunk_0",
                "text": "Knovera Python SDK Quickstart: Install knovera-sdk via pip and configure KNOVERA_API_KEY environment variable.",
                "metadata": {"source": "dev-guide.md", "chunk_index": 0, "section": "SDK Installation"}
            }
        ]
        self.corpus_records = self.generator.embed_chunks(self.corpus)

    def test_known_query_retrieves_expected_source(self):
        """Verify that a known query retrieves the expected source at Rank #1."""
        test_case = {
            "query": "How can a learner reset their account password?",
            "expected_source": "account-guide.md"
        }
        result = self.checker.evaluate_test_case(test_case, self.corpus_records)

        self.assertTrue(result["passed"])
        self.assertEqual(result["target_rank"], 1)
        self.assertEqual(result["top_source"], "account-guide.md")
        self.assertGreater(result["top_score"], 0.40)

    def test_related_ranks_strictly_above_unrelated(self):
        """Verify that related chunk score is strictly higher than unrelated distractor chunks."""
        test_case = {
            "query": "When does the cafeteria open and what is for lunch?",
            "expected_source": "campus-guide.md"
        }
        result = self.checker.evaluate_test_case(test_case, self.corpus_records)

        self.assertTrue(result["passed"])
        self.assertGreater(result["margin_vs_distractor"], 0.0)
        
        # Campus dining chunk should score higher than Python SDK chunk
        ranked = self.checker.rank_corpus_for_query(test_case["query"], self.corpus_records)
        campus_score = next(r["score"] for r in ranked if r["metadata"]["source"] == "campus-guide.md")
        dev_score = next(r["score"] for r in ranked if r["metadata"]["source"] == "dev-guide.md")
        self.assertGreater(campus_score, dev_score)

    def test_full_sanity_suite_execution(self):
        """Verify executing a full suite returns valid aggregate metrics and Hit@1 > 80%."""
        test_cases = [
            {"query": "Password reset steps for users", "expected_source": "account-guide.md"},
            {"query": "Campus lunch food menu pasta", "expected_source": "campus-guide.md"},
            {"query": "14-day refund policy eligibility", "expected_source": "service-policy.md"},
            {"query": "Installing the Python API client SDK", "expected_source": "dev-guide.md"}
        ]
        suite = self.checker.run_sanity_suite(test_cases, self.corpus_records)

        self.assertEqual(suite["total_tests"], 4)
        self.assertEqual(suite["passed"], 4)
        self.assertEqual(suite["failed"], 0)
        self.assertEqual(suite["hit_at_1_rate"], 100.0)
        self.assertEqual(suite["mrr"], 1.0)
        self.assertGreater(suite["avg_positive_margin"], 0.0)

    def test_mismatched_model_simulation_breaks_ranking(self):
        """Verify that querying with a mismatched / arbitrary vector space breaks retrieval."""
        diagnosis = self.checker.simulate_mismatched_model_ranking(
            query="How can a learner reset their account password?",
            corpus_records=self.corpus_records,
            expected_source="account-guide.md"
        )
        self.assertTrue(diagnosis["correct_passed"])
        self.assertIn("Mismatched model embeddings", diagnosis["diagnosis"])


if __name__ == "__main__":
    unittest.main()
