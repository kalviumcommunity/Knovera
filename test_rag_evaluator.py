"""
test_rag_evaluator.py

Unit Test Suite for RAG Evaluation & Answer Quality Scoring Engine (Assignment 3.43).
Tests:
1. judge_expected_points: Exact, partial, and refusal point matching.
2. judge_grounding: Context support, safe refusal, and hallucination scoring.
3. check_citations: Ground-truth citation recall, precision, and F1 scoring.
4. diagnose_failure_cause: Root-cause identification for sub-optimal scores.
5. summarize_evaluations: Aggregate quality metric calculation and failure extraction.
6. RAGEvaluator: End-to-end evaluation pipeline over structured test sets.
"""

import unittest
from typing import List, Dict, Any

from src.rag_evaluator import (
    RAGEvaluator,
    judge_expected_points,
    judge_grounding,
    check_citations,
    diagnose_failure_cause,
    normalize_source_name
)


class TestRAGEvaluator(unittest.TestCase):
    """Test suite for RAGEvaluator metrics, scoring, and summary functions."""

    def test_normalize_source_name(self):
        self.assertEqual(normalize_source_name("submission-rubric.md"), "submission_rubric.md")
        self.assertEqual(normalize_source_name("docs/video_policy.md"), "video_policy.md")
        self.assertEqual(normalize_source_name("GUARDRAILS.MD"), "guardrails.md")

    def test_judge_expected_points_full_match(self):
        answer = "Project submission requires a PR link, a sample output JSON file, and a video explanation."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        self.assertEqual(score, 1.0)

    def test_judge_expected_points_partial_match(self):
        answer = "You must provide a PR link and a video explanation for your submission."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        self.assertAlmostEqual(score, 0.6667, places=3)

    def test_judge_expected_points_refusal(self):
        answer = "I don't have enough reliable information in the provided context to answer that."
        expected = ["refuse", "say not enough information"]
        score = judge_expected_points(answer, expected)
        self.assertEqual(score, 1.0)

    def test_judge_expected_points_zero_match(self):
        answer = "The weather today is sunny with light rain."
        expected = ["PR link", "sample output"]
        score = judge_expected_points(answer, expected)
        self.assertEqual(score, 0.0)

    def test_judge_grounding_supported_context(self):
        answer = "The video explanation must be recorded as a 3-5 minute screen-share walkthrough."
        context = "Video explanation policy: The video explanation must be recorded as a screen-share walkthrough (3-5 minutes)."
        score = judge_grounding(answer, context)
        self.assertEqual(score, 1.0)

    def test_judge_grounding_missing_context_safe_refusal(self):
        answer = "I don't have enough information in the provided context."
        context = ""
        score = judge_grounding(answer, context, status="MISSING_CONTEXT_FALLBACK")
        self.assertEqual(score, 1.0)

    def test_judge_grounding_missing_context_hallucination(self):
        answer = "The capital of Mars is Olympus City with a population of 5 million."
        context = ""
        score = judge_grounding(answer, context, status="NO_CONTEXT_FOUND")
        self.assertEqual(score, 0.0)

    def test_check_citations_exact_match(self):
        citations = ["submission-rubric.md"]
        expected = ["submission_rubric.md"]
        score = check_citations(citations, expected)
        self.assertEqual(score, 1.0)

    def test_check_citations_partial_match(self):
        citations = ["submission-rubric.md", "extra_policy.md"]
        expected = ["submission_rubric.md"]
        score = check_citations(citations, expected)
        # Recall = 1.0, Precision = 0.5 -> F1 = 2 * (1*0.5) / (1+0.5) = 0.6667
        self.assertAlmostEqual(score, 0.6667, places=3)

    def test_check_citations_no_sources_expected(self):
        # Out-of-scope query: no sources expected, none cited -> 1.0
        self.assertEqual(check_citations([], []), 1.0)
        # Out-of-scope query: no sources expected, but cited -> 0.0
        self.assertEqual(check_citations(["some_doc.md"], []), 0.0)

    def test_diagnose_failure_cause(self):
        self.assertIsNone(diagnose_failure_cause(1.0, 1.0, 1.0))
        self.assertEqual(
            diagnose_failure_cause(0.5, 1.0, 1.0, num_retrieved=0),
            "WEAK_RETRIEVAL_MISSING_CONTEXT"
        )
        self.assertEqual(
            diagnose_failure_cause(1.0, 0.4, 1.0),
            "UNSUPPORTED_DETAILS_OR_OVERGENERATION"
        )
        self.assertEqual(
            diagnose_failure_cause(1.0, 1.0, 0.5),
            "BAD_CITATIONS_OR_MISSING_METADATA"
        )

    def test_summarize_evaluations(self):
        rows = [
            {
                "question": "Q1",
                "answer": "A1",
                "correctness": 1.0,
                "grounding": 1.0,
                "citation_accuracy": 1.0
            },
            {
                "question": "Q2",
                "answer": "A2",
                "correctness": 0.5,
                "grounding": 1.0,
                "citation_accuracy": 0.5
            }
        ]
        summary = RAGEvaluator.summarize_evaluations(rows)
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["avg_correctness"], 0.75)
        self.assertEqual(summary["avg_grounding"], 1.0)
        self.assertEqual(summary["avg_citation_accuracy"], 0.75)
        self.assertEqual(summary["failures_count"], 1)

    def test_end_to_end_mock_evaluation(self):
        test_set = [
            {
                "question": "What evidence is required for project submission?",
                "expected_points": ["PR link", "sample output", "video explanation"],
                "expected_sources": {"submission-rubric.md"}
            },
            {
                "question": "What should the system do when context is missing?",
                "expected_points": ["refuse", "say not enough information"],
                "expected_sources": {"guardrails.md"}
            }
        ]

        mock_results = {
            "What evidence is required for project submission?": {
                "answer": "Project submission requires a PR link, sample output, and a video explanation.",
                "citations": {"submission-rubric.md"},
                "context": "Project submission evidence requirement: Final project submissions require a PR link, a sample output JSON file, and a video explanation.",
                "status": "SUCCESS",
                "num_retrieved": 1
            },
            "What should the system do when context is missing?": {
                "answer": "The system should refuse to answer and say not enough information.",
                "citations": {"guardrails.md"},
                "context": "Guardrails: When context is missing, the system must refuse and say not enough information.",
                "status": "SUCCESS",
                "num_retrieved": 1
            }
        }

        # Test score_answer with override
        evaluator = RAGEvaluator(use_api=False)
        scored_rows = []
        for example in test_set:
            res_override = mock_results[example["question"]]
            scored = evaluator.score_answer(example, result_override=res_override)
            scored_rows.append(scored)

        summary = evaluator.summarize_evaluations(scored_rows)
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["avg_correctness"], 1.0)
        self.assertEqual(summary["avg_grounding"], 1.0)
        self.assertEqual(summary["avg_citation_accuracy"], 1.0)
        self.assertEqual(summary["failures_count"], 0)


if __name__ == "__main__":
    unittest.main()
