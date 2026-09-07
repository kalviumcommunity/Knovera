"""
test_hybrid_retriever.py

Unit test suite for Assignment 3.33: Metadata Filtering & Hybrid Search.
Tests metadata filtering, keyword scoring mechanics, weighted hybrid ranking,
precision gain calculation, and edge cases using ephemeral ChromaDB.
"""

import unittest
from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.hybrid_retriever import (
    HybridRetriever,
    keyword_score,
    hybrid_rank,
    extract_search_tokens
)


class TestHybridRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create ephemeral in-memory vector database and embedding generator
        cls.generator = EmbeddingGenerator()
        cls.vector_db = VectorDatabase(in_memory=True, embedding_generator=cls.generator)
        cls.indexer = CorpusIndexer(vector_db=cls.vector_db, default_collection="test_hybrid_corpus")
        cls.retriever = HybridRetriever(
            vector_db=cls.vector_db,
            embedding_generator=cls.generator,
            default_collection="test_hybrid_corpus"
        )

        # Benchmark corpus spanning distinct domains with intentional distractors
        cls.raw_corpus = [
            {
                "id": "auth_reset_guide_0",
                "text": "Password reset instructions: Navigate to /reset-password, enter your registered email, and verify via 6-digit OTP code.",
                "metadata": {
                    "source": "account-guide.md",
                    "section": "Account access",
                    "category": "Authentication",
                    "user_role": "Learner",
                    "doc_type": "guide"
                }
            },
            {
                "id": "auth_otp_troubleshoot_1",
                "text": "Troubleshooting account access: If the password reset OTP expires or fails with error code ERR-AUTH-902, request a new verification token.",
                "metadata": {
                    "source": "account-guide.md",
                    "section": "Account access",
                    "category": "Authentication",
                    "user_role": "Learner",
                    "doc_type": "guide"
                }
            },
            {
                "id": "it_campus_security_policy_0",
                "text": "Campus IT security policy SEC-POL-101: Passwords must be at least 12 characters and changed every 90 days. For staff compliance.",
                "metadata": {
                    "source": "campus-it-policy.md",
                    "section": "IT Security",
                    "category": "IT Policy",
                    "user_role": "Staff",
                    "doc_type": "policy"
                }
            },
            {
                "id": "campus_wifi_guide_0",
                "text": "Campus WiFi credentials setup: Connect to Knovera-Secure using your learner credentials and accept the security certificate.",
                "metadata": {
                    "source": "campus-wifi.md",
                    "section": "Network Access",
                    "category": "Campus IT",
                    "user_role": "Learner",
                    "doc_type": "guide"
                }
            },
            {
                "id": "billing_refund_policy_0",
                "text": "Subscription billing and refund SLA: Refund requests submitted within 14 days of purchase will be processed to the original payment method.",
                "metadata": {
                    "source": "billing-policy.md",
                    "section": "Refunds",
                    "category": "Billing",
                    "user_role": "Customer",
                    "doc_type": "policy"
                }
            },
            {
                "id": "dev_sdk_error_codes_0",
                "text": "Developer API Error Codes: ERR-AUTH-902 denotes an invalid or expired authentication token. Regenerate API key in Developer Portal.",
                "metadata": {
                    "source": "developer-api.md",
                    "section": "API Errors",
                    "category": "Developer Tools",
                    "user_role": "Developer",
                    "doc_type": "reference"
                }
            }
        ]

        # Embed and index test corpus
        cls.embedded_corpus = cls.generator.embed_chunks(cls.raw_corpus)
        cls.indexer.index_corpus(cls.embedded_corpus, collection_name="test_hybrid_corpus", reset_collection=True)

    def test_extract_search_tokens(self):
        """Test token extraction preserves hyphenated error codes and drops stop words."""
        tokens = extract_search_tokens("What are the steps for ERR-AUTH-902 and SEC-POL-101 in CS-101?")
        self.assertIn("err-auth-902", tokens)
        self.assertIn("sec-pol-101", tokens)
        self.assertIn("cs-101", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("for", tokens)

    def test_keyword_score_methods(self):
        """Test keyword scoring for frequency, count, presence, and exact boundaries."""
        sample_text = "Error ERR-AUTH-902 occurred. Please fix error ERR-AUTH-902 immediately."
        
        count_score = keyword_score(sample_text, ["ERR-AUTH-902"], method="count")
        self.assertEqual(count_score, 2.0)

        presence_score = keyword_score(sample_text, ["ERR-AUTH-902", "unrelated_term"], method="presence")
        self.assertEqual(presence_score, 0.5)

        freq_score = keyword_score(sample_text, ["ERR-AUTH-902"], method="frequency")
        self.assertGreater(freq_score, 0.5)

        # Exact match boundary check
        word_text = "The cat categorized the caterpillar."
        exact_score = keyword_score(word_text, ["cat"], exact_match=True, method="count")
        non_exact_score = keyword_score(word_text, ["cat"], exact_match=False, method="count")
        self.assertEqual(exact_score, 1.0)
        self.assertEqual(non_exact_score, 3.0)

    def test_hybrid_rank_weight_blending(self):
        """Test that hybrid_rank correctly fuses vector scores with keyword bonuses."""
        mock_results = [
            {"id": "doc_a", "score": 0.90, "text": "General information about platform accounts and login settings."},
            {"id": "doc_b", "score": 0.70, "text": "Error troubleshooting guide for code ERR-AUTH-902 token refresh."}
        ]

        # Pure vector (weight 1.0, 0.0) leaves doc_a first
        res_pure_vec = hybrid_rank(mock_results, keywords=["ERR-AUTH-902"], vector_weight=1.0, keyword_weight=0.0)
        self.assertEqual(res_pure_vec[0]["id"], "doc_a")

        # Hybrid weighting (0.4 vector, 0.6 keyword) boosts doc_b to top because of exact token match
        res_hybrid = hybrid_rank(mock_results, keywords=["ERR-AUTH-902"], vector_weight=0.4, keyword_weight=0.6)
        self.assertEqual(res_hybrid[0]["id"], "doc_b")
        self.assertGreater(res_hybrid[0]["hybrid_score"], res_hybrid[1]["hybrid_score"])

    def test_task1_metadata_filter(self):
        """Task 1: Use metadata filter to restrict retrieval to section='Account access'."""
        query = "What are the password reset steps?"
        filtered = self.retriever.retrieve(
            query=query,
            top_k=5,
            metadata_filter={"section": "Account access"}
        )

        self.assertTrue(len(filtered) > 0)
        for item in filtered:
            self.assertEqual(item["metadata"]["section"], "Account access")
            self.assertEqual(item["metadata"]["category"], "Authentication")

    def test_task2_compare_filtered_and_unfiltered_results(self):
        """Task 2: Compare filtered and unfiltered results for the same query."""
        query = "What are the password reset steps?"
        
        unfiltered = self.retriever.retrieve(query=query, top_k=3, metadata_filter=None)
        filtered = self.retriever.retrieve(query=query, top_k=3, metadata_filter={"section": "Account access"})

        unfiltered_sections = [item["metadata"].get("section") for item in unfiltered]
        filtered_sections = [item["metadata"].get("section") for item in filtered]

        # Filtered results must strictly be from 'Account access'
        self.assertTrue(all(sec == "Account access" for sec in filtered_sections))
        # Unfiltered results may contain other sections or campus IT policies
        self.assertEqual(len(filtered_sections), 2)  # exactly 2 chunks exist in Account access

    def test_task3_hybrid_matching_exact_code(self):
        """Task 3: Combine vector search with keyword matching for exact error code ERR-AUTH-902."""
        query = "How do I fix ERR-AUTH-902 authentication error?"
        
        # In developer context vs account guide context
        dev_filtered_hybrid = self.retriever.hybrid_search(
            query=query,
            keywords=["ERR-AUTH-902"],
            top_k=2,
            metadata_filter={"category": "Developer Tools"}
        )
        self.assertEqual(len(dev_filtered_hybrid), 1)
        self.assertEqual(dev_filtered_hybrid[0]["id"], "dev_sdk_error_codes_0")
        self.assertIn("ERR-AUTH-902", dev_filtered_hybrid[0]["text"])

    def test_task4_demonstrate_improved_precision(self):
        """Task 4: Demonstrate improved precision by scoping retrieval to category='Authentication'."""
        query = "How do I reset my credentials and handle expired passwords?"
        
        comparison = self.retriever.compare_retrieval_modes(
            query=query,
            metadata_filter={"category": "Authentication"},
            top_k=3,
            target_criteria={"category": "Authentication"}
        )

        precision_unfiltered = comparison["precision_metrics"]["unfiltered_vector_precision"]
        precision_filtered = comparison["precision_metrics"]["filtered_vector_precision"]

        # Filtered retrieval achieves 100% precision on target domain
        self.assertEqual(precision_filtered, 1.0)
        self.assertGreaterEqual(precision_filtered, precision_unfiltered)

    def test_edge_cases(self):
        """Test non-matching filter, empty keyword list, and empty query."""
        # Filter matching zero items returns empty list gracefully
        no_match = self.retriever.retrieve(
            query="password",
            top_k=3,
            metadata_filter={"category": "NonExistentCategory"}
        )
        self.assertEqual(len(no_match), 0)

        # Empty keywords fallback to vector score
        empty_kw_res = hybrid_rank([{"id": "c1", "score": 0.85, "text": "sample"}], keywords=[])
        self.assertEqual(len(empty_kw_res), 1)
        self.assertEqual(empty_kw_res[0]["keyword_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
