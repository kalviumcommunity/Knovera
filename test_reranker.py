"""
test_reranker.py

Unit test suite for Assignment 3.35: Chunk Re-Ranking for Precision.
Tests Two-Stage Retrieval (K_initial > K_final), cross-scoring relevance evaluation,
rank promotions (rank deltas), precision improvements, and edge cases.
"""

import unittest
from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.reranker import ChunkReranker


class TestChunkReranker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create ephemeral in-memory vector database and embedding generator
        cls.generator = EmbeddingGenerator()
        cls.vector_db = VectorDatabase(in_memory=True, embedding_generator=cls.generator)
        cls.indexer = CorpusIndexer(vector_db=cls.vector_db, default_collection="test_rerank_corpus")
        cls.reranker = ChunkReranker(
            vector_db=cls.vector_db,
            embedding_generator=cls.generator,
            default_collection="test_rerank_corpus"
        )

        # Realistic 10-chunk corpus containing project guidelines, distractors, and exact evidence criteria
        cls.raw_corpus = [
            {
                "id": "proj_overview_0",
                "text": "Project Overview and Milestones: All capstone projects must adhere to weekly agile sprint deadlines and milestone deliverables.",
                "metadata": {"source": "project-overview.md", "chunk_index": 0, "section": "Overview"}
            },
            {
                "id": "proj_team_etiquette_1",
                "text": "Team collaboration etiquette: Keep branch names descriptive, write clean commit messages, and review teammate pull requests promptly.",
                "metadata": {"source": "team-guide.md", "chunk_index": 1, "section": "Git Workflow"}
            },
            {
                "id": "proj_submission_evidence_2",
                "text": "Mandatory Submission Evidence & Verification: Every project submission must include two mandatory items: 1. A public GitHub PR URL showing meaningful commits, and 2. A 3-5 minute video screen recording walkthrough explaining code and answering questions.",
                "metadata": {"source": "submission-policy.md", "chunk_index": 2, "section": "Submission Evidence"}
            },
            {
                "id": "proj_grading_rubric_3",
                "text": "Grading Rubric Breakdown: Final grades are weighted 40% on test coverage, 30% on architecture design, and 30% on code quality standards.",
                "metadata": {"source": "rubric.md", "chunk_index": 3, "section": "Grading"}
            },
            {
                "id": "proj_late_policy_4",
                "text": "Late Submissions and Extensions: Submissions received after the deadline without prior medical exemption incur a 10% daily grade deduction.",
                "metadata": {"source": "submission-policy.md", "chunk_index": 4, "section": "Late Policy"}
            },
            {
                "id": "campus_dining_menu_5",
                "text": "Campus dining options: The campus cafeteria serves breakfast from 7:30 AM to 10:00 AM and offers vegan meals daily.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 5, "section": "Dining"}
            },
            {
                "id": "wifi_setup_guide_6",
                "text": "Campus WiFi setup: Connect to Knovera-Secure using your student ID credentials and accept the security certificate.",
                "metadata": {"source": "wifi.md", "chunk_index": 6, "section": "Network"}
            },
            {
                "id": "account_pwd_reset_7",
                "text": "Password reset instructions: Navigate to /reset-password, enter registered email, and verify with 6-digit OTP code.",
                "metadata": {"source": "account.md", "chunk_index": 7, "section": "Authentication"}
            },
            {
                "id": "dev_sdk_api_8",
                "text": "Developer API client configuration: Initialize the Knovera client with base_url and API key environment variables.",
                "metadata": {"source": "dev-api.md", "chunk_index": 8, "section": "Developer Tools"}
            },
            {
                "id": "billing_refund_sla_9",
                "text": "Billing refund SLA: Refund requests submitted within 14 days are processed to the original payment method within 3-5 business days.",
                "metadata": {"source": "billing.md", "chunk_index": 9, "section": "Refunds"}
            }
        ]

        # Embed and index test corpus
        cls.embedded_corpus = cls.generator.embed_chunks(cls.raw_corpus)
        cls.indexer.index_corpus(cls.embedded_corpus, collection_name="test_rerank_corpus", reset_collection=True)

    def test_task1_retrieve_larger_candidate_set(self):
        """Task 1: Retrieve candidate set (K_initial = 10) larger than final context (K_final = 3)."""
        query = "What evidence is required for project submission?"
        
        pipeline_res = self.reranker.retrieve_and_rerank(
            query=query,
            initial_k=10,
            final_k=3,
            collection_name="test_rerank_corpus"
        )

        self.assertEqual(pipeline_res["initial_k"], 10)
        self.assertEqual(pipeline_res["final_k"], 3)
        self.assertEqual(len(pipeline_res["stage_1_all_candidates"]), 10)
        self.assertEqual(len(pipeline_res["stage_2_final_selected"]), 3)

    def test_task2_re_ranking_scoring_and_ordering(self):
        """Task 2: Re-rank candidates by relevance score (0.0 - 10.0) to query."""
        query = "What evidence is required for project submission?"
        
        candidates = [
            {"id": "doc_general", "score": 0.75, "text": "General project overview and weekly agile sprint schedules."},
            {"id": "doc_evidence", "score": 0.60, "text": "Mandatory Submission Evidence: Must include GitHub PR URL and 3-5 minute video screen recording walkthrough."}
        ]

        reranked = self.reranker.rerank_candidates(query=query, candidates=candidates, top_k=2)

        # Assert doc_evidence with exact requirements receives highest rerank_score
        self.assertEqual(reranked[0]["id"], "doc_evidence")
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])
        self.assertEqual(reranked[0]["rerank_rank"], 1)
        self.assertEqual(reranked[0]["initial_rank"], 2)
        self.assertEqual(reranked[0]["rank_delta"], 1)  # Promoted from rank 2 to 1 (+1)

    def test_task3_show_improved_top_results(self):
        """Task 3: Show that top result after re-ranking is more directly relevant than initial vector order."""
        query = "What evidence is required for project submission?"
        
        pipeline_res = self.reranker.retrieve_and_rerank(
            query=query,
            initial_k=8,
            final_k=3,
            collection_name="test_rerank_corpus"
        )

        final_top_1 = pipeline_res["stage_2_final_selected"][0]

        # The exact submission evidence chunk must be Rank #1 after re-ranking
        self.assertEqual(final_top_1["id"], "proj_submission_evidence_2")
        self.assertIn("GitHub PR URL", final_top_1["text"])
        self.assertIn("video screen recording", final_top_1["text"])
        self.assertGreater(final_top_1["rerank_score"], 4.0)

    def test_task4_compare_before_and_after_ordering(self):
        """Task 4: Compare before-and-after ordering including vector scores, rerank scores, and deltas."""
        query = "What evidence is required for project submission?"
        
        pipeline_res = self.reranker.retrieve_and_rerank(
            query=query,
            initial_k=10,
            final_k=3,
            collection_name="test_rerank_corpus"
        )

        initial_top = pipeline_res["stage_1_initial_top_k"]
        reranked_top = pipeline_res["stage_2_final_selected"]

        self.assertEqual(len(initial_top), 3)
        self.assertEqual(len(reranked_top), 3)

        # Verify fields exist on re-ranked objects
        for r in reranked_top:
            self.assertIn("vector_score", r)
            self.assertIn("rerank_score", r)
            self.assertIn("initial_rank", r)
            self.assertIn("rerank_rank", r)
            self.assertIn("rank_delta", r)

    def test_cost_and_latency_metrics(self):
        """Test latency tracking and token/cost estimation metrics."""
        query = "What are the password reset steps?"
        res = self.reranker.retrieve_and_rerank(
            query=query,
            initial_k=5,
            final_k=2,
            collection_name="test_rerank_corpus"
        )

        metrics = res["metrics"]
        self.assertIn("stage_1_retrieval_latency_ms", metrics)
        self.assertIn("stage_2_rerank_latency_ms", metrics)
        self.assertIn("total_pipeline_latency_ms", metrics)
        self.assertGreater(metrics["total_pipeline_latency_ms"], 0.0)
        self.assertEqual(metrics["candidates_scored"], 5)
        self.assertGreater(metrics["estimated_tokens_processed"], 0)

    def test_edge_cases(self):
        """Test empty candidate pool and zero query string."""
        empty_res = self.reranker.rerank_candidates(query="test", candidates=[])
        self.assertEqual(empty_res, [])

        # Low relevance candidate scoring
        score_low = self.reranker.score_chunk("What is the refund policy?", "The cafeteria serves breakfast.")
        score_high = self.reranker.score_chunk("What is the refund policy?", "Refunds are processed in 14 days.")
        self.assertLess(score_low, score_high)


if __name__ == "__main__":
    unittest.main()
