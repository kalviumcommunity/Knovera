"""
test_grounded_generator.py

Comprehensive test suite for Assignment 3.39: Grounded Answer Generation.
Validates:
1. Task 1: Grounded generation from injected context with source citations.
2. Task 2: Grounding verification, source claim accuracy, and unsupported content detection.
3. Task 3: Missing-context fallback handling ("I don't have enough information in the provided context.").
4. Task 4: Comparative evaluation of With Retrieval vs Without Retrieval.
5. Task 5: Object-oriented GroundedGenerator engine integration.
"""

import unittest
from typing import List, Dict, Any

from src.grounded_generator import (
    generate_grounded_answer,
    generate_ungrounded_answer,
    verify_grounding,
    answer_query,
    compare_grounded_vs_ungrounded,
    GroundedGenerator,
    STANDARD_MISSING_CONTEXT_FALLBACK
)


class MockRetriever:
    """Mock Retriever for testing query answering and fallback behavior."""
    def __init__(self, chunks_to_return=None):
        self.chunks = chunks_to_return or []
        
    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return self.chunks[:k]


class TestGroundedAnswerGeneration(unittest.TestCase):
    """Unit and Integration tests for Grounded Answer Generation."""

    def setUp(self):
        """Prepares sample grounded test chunks."""
        self.sample_chunks = [
            {
                "id": "submission-rubric.md#chunk_0",
                "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough demonstrating all test executions, architectural flows, and key modular components.",
                "metadata": {
                    "source": "submission-rubric.md",
                    "chunk_index": 0,
                    "section": "Evidence Requirements",
                    "doc_title": "Project Submission Rubric",
                    "category": "Academics"
                },
                "score": 0.8921
            },
            {
                "id": "grading-policy.md#chunk_0",
                "text": "Assignment grading criteria: Submissions are evaluated on unit test coverage (40%), modular code structure (30%), and comprehensive documentation (30%). A score of 60% or higher is required to pass.",
                "metadata": {
                    "source": "grading-policy.md",
                    "chunk_index": 0,
                    "section": "Evaluation Rubrics",
                    "doc_title": "Academic Grading Policy",
                    "category": "Academics"
                },
                "score": 0.7415
            }
        ]

    # ------------------------------------------------------------------------
    # TASK 1 TESTS: GENERATE FROM INJECTED CONTEXT
    # ------------------------------------------------------------------------
    def test_task1_generate_grounded_answer_success(self):
        """Validates grounded answer generation from injected context."""
        question = "What evidence is required for project submission?"
        result = generate_grounded_answer(
            question=question,
            retrieved_chunks=self.sample_chunks,
            use_api=False
        )
        
        self.assertEqual(result["status"], "GROUNDED_SUCCESS")
        self.assertEqual(result["question"], question)
        self.assertIn("GitHub", result["answer"])
        self.assertIn("video walkthrough", result["answer"])
        self.assertTrue(result["grounding_verified"])
        self.assertGreaterEqual(result["grounding_score"], 0.5)
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["source"], "submission-rubric.md")

    def test_task1_generate_grounded_answer_empty_question_error(self):
        """Validates error raised on empty question."""
        with self.assertRaises(ValueError):
            generate_grounded_answer("", self.sample_chunks)

    # ------------------------------------------------------------------------
    # TASK 2 TESTS: CHECK SOURCE ACCURACY & VERIFY GROUNDING
    # ------------------------------------------------------------------------
    def test_task2_verify_grounding_positive(self):
        """Validates that a correctly grounded answer receives a high grounding score."""
        answer = "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough. [1]"
        audit = verify_grounding(answer, self.sample_chunks)
        
        self.assertTrue(audit["is_grounded"])
        self.assertGreaterEqual(audit["grounding_score"], 0.7)
        self.assertTrue(audit["has_citations"])
        self.assertIn("[1]", audit["cited_markers"])
        self.assertEqual(audit["reason"], "VERIFIED_GROUNDED")

    def test_task2_verify_grounding_negative_hallucination(self):
        """Validates that an answer with fabricated claims fails grounding verification."""
        hallucinated_answer = "Learners must build an extraterrestrial rocket ship and travel to Jupiter by midnight."
        audit = verify_grounding(hallucinated_answer, self.sample_chunks)
        
        self.assertFalse(audit["is_grounded"])
        self.assertLess(audit["grounding_score"], 0.35)
        self.assertEqual(audit["reason"], "POTENTIAL_UNGROUNDED_CONTENT")

    def test_task2_verify_grounding_fallback_phrase(self):
        """Validates that standard missing-context fallback is recognized as valid grounded response."""
        audit = verify_grounding(STANDARD_MISSING_CONTEXT_FALLBACK, self.sample_chunks)
        self.assertTrue(audit["is_grounded"])
        self.assertEqual(audit["grounding_score"], 1.0)
        self.assertEqual(audit["reason"], "VALID_MISSING_CONTEXT_FALLBACK")

    # ------------------------------------------------------------------------
    # TASK 3 TESTS: MISSING-CONTEXT FALLBACK HANDLING
    # ------------------------------------------------------------------------
    def test_task3_answer_query_empty_retrieval_fallback(self):
        """Validates answer_query triggers fallback when retriever returns no chunks."""
        empty_retriever = MockRetriever(chunks_to_return=[])
        question = "What is the warp drive maintenance schedule?"
        
        result = answer_query(
            question=question,
            retriever=empty_retriever,
            use_api=False
        )
        
        self.assertEqual(result["status"], "MISSING_CONTEXT_FALLBACK")
        self.assertEqual(result["answer"], STANDARD_MISSING_CONTEXT_FALLBACK)
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["grounding_verified"])

    def test_task3_answer_query_with_valid_chunks(self):
        """Validates answer_query produces grounded answer when chunks are present."""
        active_retriever = MockRetriever(chunks_to_return=self.sample_chunks)
        question = "What evidence is required for project submission?"
        
        result = answer_query(
            question=question,
            retriever=active_retriever,
            use_api=False
        )
        
        self.assertEqual(result["status"], "GROUNDED_SUCCESS")
        self.assertIn("GitHub", result["answer"])
        self.assertEqual(len(result["sources"]), 2)

    # ------------------------------------------------------------------------
    # TASK 4 TESTS: COMPARE WITH AND WITHOUT RETRIEVAL
    # ------------------------------------------------------------------------
    def test_task4_generate_ungrounded_answer(self):
        """Validates ungrounded answer generation has zero sources and ungrounded mode."""
        question = "What evidence is required for project submission?"
        result = generate_ungrounded_answer(question=question, use_api=False)
        
        self.assertEqual(result["mode"], "UNGROUNDED_NO_RETRIEVAL")
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["context_injected"])
        self.assertIsInstance(result["answer"], str)

    def test_task4_compare_grounded_vs_ungrounded(self):
        """Validates side-by-side comparison structure and metric contrast."""
        question = "What evidence is required for project submission?"
        comparison = compare_grounded_vs_ungrounded(
            question=question,
            retrieved_chunks=self.sample_chunks,
            use_api=False
        )
        
        self.assertIn("with_retrieval", comparison)
        self.assertIn("without_retrieval", comparison)
        self.assertTrue(comparison["with_retrieval"]["is_grounded"])
        self.assertFalse(comparison["without_retrieval"]["is_grounded"])
        self.assertEqual(comparison["with_retrieval"]["sources_count"], 2)
        self.assertEqual(comparison["without_retrieval"]["sources_count"], 0)
        self.assertTrue(comparison["with_retrieval"]["has_citations"])
        self.assertFalse(comparison["without_retrieval"]["has_citations"])

    # ------------------------------------------------------------------------
    # TASK 5 TESTS: GROUNDED GENERATOR OOP ENGINE
    # ------------------------------------------------------------------------
    def test_task5_grounded_generator_oop_engine(self):
        """Validates GroundedGenerator class wrapper methods."""
        retriever = MockRetriever(chunks_to_return=self.sample_chunks)
        generator = GroundedGenerator(retriever=retriever, use_api=False)
        
        # Grounded generation method
        g_res = generator.generate_grounded("What evidence is needed?", self.sample_chunks)
        self.assertEqual(g_res["status"], "GROUNDED_SUCCESS")
        
        # Ungrounded generation method
        u_res = generator.generate_ungrounded("What evidence is needed?")
        self.assertEqual(u_res["mode"], "UNGROUNDED_NO_RETRIEVAL")
        
        # Query answering method
        ans_res = generator.answer("What evidence is needed?")
        self.assertEqual(ans_res["status"], "GROUNDED_SUCCESS")
        
        # Verification method
        ver_res = generator.verify(g_res["answer"], self.sample_chunks)
        self.assertTrue(ver_res["is_grounded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
