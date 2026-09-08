"""
test_hallucination_guardrail.py

Unit Test Suite for Assignment 3.41: Hallucination Guardrails & Refusal Handling.
Validates:
- Weak retrieval detection (empty, low similarity score, insufficient supporting chunks)
- Safe refusal response generation with status codes (refused_empty_context, refused_low_similarity)
- Relevance threshold checks (MIN_TOP_SCORE = 0.72, MIN_SUPPORTING_CHUNKS = 1)
- Preservation of confident grounded answers when strong context exists
- Class-based guardrail telemetry tracking
- Dynamic threshold tuning helper
- RAGPipeline guarded_query integration
"""

import sys
import unittest
from pathlib import Path

# Ensure Knovera root is in sys.path
_root_dir = str(Path(__file__).resolve().parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.hallucination_guardrail import (
    retrieval_is_strong,
    evaluate_retrieval_quality,
    guarded_answer,
    HallucinationGuardrail,
    tune_guardrail_thresholds,
    MIN_TOP_SCORE,
    MIN_SUPPORTING_CHUNKS,
    STANDARD_SAFE_REFUSAL
)
from src.rag_pipeline import RAGPipeline


class TestHallucinationGuardrail(unittest.TestCase):
    def setUp(self):
        self.strong_chunks = [
            {
                "id": "doc1#chunk_0",
                "text": "All project submissions require a public GitHub PR link and a 3-minute video.",
                "source": "doc1.md",
                "metadata": {"source": "doc1.md", "chunk_id": "doc1#chunk_0"},
                "score": 0.8800
            },
            {
                "id": "doc1#chunk_1",
                "text": "Automated evaluation requires a minimum pass mark of 60%.",
                "source": "doc1.md",
                "metadata": {"source": "doc1.md", "chunk_id": "doc1#chunk_1"},
                "score": 0.7900
            }
        ]
        self.low_score_chunks = [
            {
                "id": "refund#chunk_0",
                "text": "Returns are processed within 14 business days for standard retail hardware items.",
                "source": "refund.md",
                "metadata": {"source": "refund.md"},
                "score": 0.4200
            }
        ]
        self.empty_chunks = []

    def test_retrieval_is_strong_true(self):
        self.assertTrue(retrieval_is_strong(self.strong_chunks, min_top_score=0.72))

    def test_retrieval_is_strong_empty(self):
        self.assertFalse(retrieval_is_strong(self.empty_chunks))

    def test_retrieval_is_strong_low_score(self):
        self.assertFalse(retrieval_is_strong(self.low_score_chunks, min_top_score=0.72))

    def test_retrieval_is_strong_insufficient_chunks(self):
        # Top score matches, but min_supporting_chunks = 2 (strong_chunks has 2, so True)
        self.assertTrue(retrieval_is_strong(self.strong_chunks, min_top_score=0.72, min_supporting_chunks=2))
        # If we demand 3 supporting chunks, it should return False
        self.assertFalse(retrieval_is_strong(self.strong_chunks, min_top_score=0.72, min_supporting_chunks=3))

    def test_evaluate_retrieval_quality_empty(self):
        res = evaluate_retrieval_quality(self.empty_chunks)
        self.assertFalse(res["is_strong"])
        self.assertEqual(res["status_code"], "refused_empty_context")
        self.assertEqual(res["refusal_reason"], "NO_CHUNKS_RETRIEVED")

    def test_evaluate_retrieval_quality_low_similarity(self):
        res = evaluate_retrieval_quality(self.low_score_chunks, min_top_score=0.72)
        self.assertFalse(res["is_strong"])
        self.assertEqual(res["status_code"], "refused_low_similarity")
        self.assertIn("TOP_SCORE_BELOW_THRESHOLD", res["refusal_reason"])

    def test_evaluate_retrieval_quality_insufficient_support(self):
        single_strong = [self.strong_chunks[0]]
        res = evaluate_retrieval_quality(single_strong, min_top_score=0.72, min_supporting_chunks=2)
        self.assertFalse(res["is_strong"])
        self.assertEqual(res["status_code"], "refused_insufficient_support")

    def test_evaluate_retrieval_quality_strong(self):
        res = evaluate_retrieval_quality(self.strong_chunks, min_top_score=0.72)
        self.assertTrue(res["is_strong"])
        self.assertEqual(res["status_code"], "answered")

    def test_guarded_answer_refusal_empty(self):
        res = guarded_answer("What is the refund policy?", self.empty_chunks)
        self.assertTrue(res["guardrail_triggered"])
        self.assertEqual(res["answer"], STANDARD_SAFE_REFUSAL)
        self.assertEqual(res["status"], "refused_empty_context")
        self.assertEqual(res["sources"], [])
        self.assertEqual(res["citations"], {})

    def test_guarded_answer_refusal_low_score(self):
        res = guarded_answer("What is the refund policy?", self.low_score_chunks, min_top_score=0.72)
        self.assertTrue(res["guardrail_triggered"])
        self.assertEqual(res["answer"], STANDARD_SAFE_REFUSAL)
        self.assertEqual(res["status"], "refused_low_similarity")

    def test_guarded_answer_confident_pass(self):
        res = guarded_answer("What evidence is required for submission?", self.strong_chunks, min_top_score=0.72)
        self.assertFalse(res["guardrail_triggered"])
        self.assertEqual(res["status"], "answered")
        self.assertIn("submission", res["answer"].lower())

    def test_guarded_answer_custom_refusal_msg(self):
        custom_msg = "Custom refusal: Insufficient context available."
        res = guarded_answer("Unknown topic?", self.empty_chunks, refusal_message=custom_msg)
        self.assertEqual(res["answer"], custom_msg)

    def test_guarded_answer_invalid_query(self):
        with self.assertRaises(ValueError):
            guarded_answer("   ", self.strong_chunks)

    def test_hallucination_guardrail_class_telemetry(self):
        guard = HallucinationGuardrail(min_top_score=0.72)
        guard.generate("Question 1", self.strong_chunks)
        guard.generate("Question 2", self.low_score_chunks)
        guard.generate("Question 3", self.empty_chunks)

        telemetry = guard.get_telemetry()
        self.assertEqual(telemetry["total_evaluated"], 3)
        self.assertEqual(telemetry["total_answered"], 1)
        self.assertEqual(telemetry["total_refused"], 2)
        self.assertAlmostEqual(telemetry["refusal_rate"], 2/3, places=2)

    def test_tune_guardrail_thresholds(self):
        dataset = [
            {"query": "Q1", "retrieved_chunks": self.strong_chunks, "is_answerable": True},
            {"query": "Q2", "retrieved_chunks": self.low_score_chunks, "is_answerable": False},
            {"query": "Q3", "retrieved_chunks": self.empty_chunks, "is_answerable": False}
        ]
        res = tune_guardrail_thresholds(dataset, candidate_thresholds=[0.50, 0.72, 0.90])
        self.assertIn("best_threshold", res)
        self.assertIn("best_f1_score", res)
        self.assertEqual(len(res["candidate_evaluations"]), 3)

    def test_rag_pipeline_guarded_query(self):
        pipeline = RAGPipeline(use_api=False)
        # Call guarded_query with empty query handling or custom mock vector retrieval
        res = pipeline.guarded_query("What is the submission requirement?", k=2, min_top_score=0.99)
        self.assertTrue(res["guardrail_triggered"])
        self.assertIn("refused", res["status"])


if __name__ == "__main__":
    unittest.main()
