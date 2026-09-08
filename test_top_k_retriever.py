"""
test_top_k_retriever.py

Comprehensive test suite for Assignment 3.32: Similarity Search & Top-K Retrieval.
Validates model alignment, top-k retrieval, score calculation, metadata payload integrity,
and multi-k comparative analysis.
"""

import unittest
from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.top_k_retriever import TopKRetriever


class TestTopKRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.generator = EmbeddingGenerator()
        cls.chunks = [
            {
                "id": "account-guide.md:0",
                "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Recovery", "doc_title": "Account Admin Guide", "category": "Authentication"}
            },
            {
                "id": "account-guide.md:1",
                "text": "Learners can recover access using their registered email address. A one-time secure verification link is dispatched immediately.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Verification", "doc_title": "Account Admin Guide", "category": "Authentication"}
            },
            {
                "id": "account-guide.md:2",
                "text": "Two-factor authentication (2FA) is mandatory for enterprise admin consoles. Register TOTP authenticator app or hardware key.",
                "metadata": {"source": "account-guide.md", "chunk_index": 2, "section": "Multi-Factor Security", "doc_title": "Account Admin Guide", "category": "Authentication"}
            },
            {
                "id": "service-policy.md:0",
                "text": "Knovera Customer Service & Refund Policy: Enterprise clients are eligible for full subscription refunds within 14 days of provisioning.",
                "metadata": {"source": "service-policy.md", "chunk_index": 0, "section": "Refund Eligibility", "doc_title": "Service Policy Guide", "category": "Billing"}
            },
            {
                "id": "campus-guide.md:0",
                "text": "Campus cafeteria hours and lunch menu: Operating from 11:30 AM to 2:30 PM offering fresh garden salads and daily specials.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Dining Hours", "doc_title": "Campus Guide", "category": "Campus Life"}
            }
        ]
        cls.embedded_chunks = cls.generator.embed_chunks(cls.chunks)

    def setUp(self):
        self.vdb = VectorDatabase(in_memory=True)
        self.col_name = "test_top_k_collection"
        self.indexer = CorpusIndexer(vector_db=self.vdb, default_collection=self.col_name)
        self.indexer.index_corpus(self.embedded_chunks, reset_collection=True)
        self.retriever = TopKRetriever(vector_db=self.vdb)

    def test_top_k_retriever_initialization(self):
        """Tests default initialization of TopKRetriever."""
        retriever = TopKRetriever()
        self.assertIsNotNone(retriever.generator)
        self.assertIsNotNone(retriever.vector_db)
        self.assertEqual(retriever.default_collection, "knovera_top_k_chunks")

    def test_model_alignment_verification(self):
        """Tests model and dimension alignment verification."""
        alignment_info = self.retriever.verify_model_alignment(collection_name=self.col_name)
        self.assertTrue(alignment_info["is_aligned"])
        self.assertEqual(alignment_info["status"], "ALIGNED")
        self.assertEqual(alignment_info["query_dimension"], alignment_info["collection_expected_dimension"])
        self.assertEqual(alignment_info["collection_record_count"], 5)

    def test_single_top_k_retrieval(self):
        """Task 1-3: Tests single top-k similarity search returning scores and metadata."""
        query = "How can a learner reset their password?"
        results = self.retriever.retrieve(query=query, k=3, collection_name=self.col_name)

        self.assertEqual(len(results), 3)

        # Verify score ranking order (descending similarity)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Verify top result is password reset chunk
        top_result = results[0]
        self.assertEqual(top_result["rank"], 1)
        self.assertTrue("password" in top_result["text"].lower() or "recover" in top_result["text"].lower())
        self.assertEqual(top_result["metadata"]["source"], "account-guide.md")
        self.assertIn(top_result["metadata"]["chunk_index"], (0, 1))

        # Verify payload schema match
        for res in results:
            self.assertIn("rank", res)
            self.assertIn("score", res)
            self.assertIn("distance", res)
            self.assertIn("text", res)
            self.assertIn("metadata", res)
            self.assertIn("source", res["metadata"])
            self.assertIn("chunk_index", res["metadata"])
            self.assertIn("id", res)

    def test_demonstrate_changing_k(self):
        """Task 4: Tests running the query across k=1, k=3, and k=5."""
        query = "How can a learner reset their password?"
        comparison = self.retriever.compare_k(query=query, k_values=[1, 3, 5], collection_name=self.col_name)

        self.assertEqual(comparison["query"], query)
        self.assertEqual(len(comparison["runs"]), 3)

        run_k1 = comparison["runs"][0]
        run_k3 = comparison["runs"][1]
        run_k5 = comparison["runs"][2]

        self.assertEqual(run_k1["k"], 1)
        self.assertEqual(len(run_k1["results"]), 1)
        self.assertEqual(run_k3["k"], 3)
        self.assertEqual(len(run_k3["results"]), 3)
        self.assertEqual(run_k5["k"], 5)
        self.assertEqual(len(run_k5["results"]), 5)

        # Top result should be identical across all k values
        top_k1_id = run_k1["results"][0]["id"]
        top_k3_id = run_k3["results"][0]["id"]
        top_k5_id = run_k5["results"][0]["id"]
        self.assertEqual(top_k1_id, top_k3_id)
        self.assertEqual(top_k3_id, top_k5_id)

        # Context character count must expand with larger k
        self.assertLess(run_k1["total_context_characters"], run_k3["total_context_characters"])
        self.assertLess(run_k3["total_context_characters"], run_k5["total_context_characters"])

    def test_edge_cases_and_input_validation(self):
        """Tests handling of empty strings, whitespace, and invalid k values."""
        # Empty / whitespace query
        self.assertEqual(self.retriever.retrieve("", k=3, collection_name=self.col_name), [])
        self.assertEqual(self.retriever.retrieve("   ", k=3, collection_name=self.col_name), [])

        # Invalid k <= 0
        with self.assertRaises(ValueError):
            self.retriever.retrieve("Valid query", k=0, collection_name=self.col_name)

        with self.assertRaises(ValueError):
            self.retriever.retrieve("Valid query", k=-5, collection_name=self.col_name)


if __name__ == "__main__":
    unittest.main()
