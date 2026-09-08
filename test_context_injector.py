"""
test_context_injector.py

Comprehensive test suite for Assignment 3.38: Context Injection & Prompt Augmentation.
Validates:
1. Task 1: Retrieved Chunk Formatting (labeling, metadata binding, text extraction)
2. Task 2: Token Budget Enforcement (dynamic counting, budget cutoff, chunk dropping)
3. Task 3: Source Markers Inclusion (citation indexing, source filename, chunk ID markers)
4. Task 4: Grounding Instructions (strict context boundary, fallback phrases, prompt rendering)
5. Task 5: ContextInjector Class & Telemetry (end-to-end prompt payload generation)
"""

import unittest
from typing import List, Dict, Any

from src.context_injector import (
    format_chunk,
    assemble_context,
    build_prompt,
    count_tokens,
    ContextInjector,
    DEFAULT_MAX_CONTEXT_TOKENS,
    GROUNDED_RAG_PROMPT_TEMPLATE
)


class TestContextInjectionPromptAugmentation(unittest.TestCase):
    """Unit and Integration tests for Context Injection and Prompt Augmentation."""

    def setUp(self):
        """Prepares sample retrieved document chunks for testing."""
        self.sample_chunks = [
            {
                "id": "submission-rubric.md#chunk_0",
                "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough.",
                "metadata": {
                    "source": "submission-rubric.md",
                    "chunk_index": 0,
                    "section": "Evidence Requirements",
                    "category": "Academics"
                }
            },
            {
                "id": "account-guide.md#chunk_0",
                "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
                "metadata": {
                    "source": "account-guide.md",
                    "chunk_index": 0,
                    "section": "Authentication",
                    "category": "Authentication"
                }
            },
            {
                "id": "campus-guide.md#chunk_0",
                "text": "Campus cafeteria hours: The cafeteria operates from 11:30 AM to 2:30 PM offering hot wraps and fresh salads.",
                "metadata": {
                    "source": "campus-guide.md",
                    "chunk_index": 0,
                    "section": "Dining Facilities",
                    "category": "Campus Life"
                }
            },
            {
                "id": "grading-policy.md#chunk_0",
                "text": "Grading policy: Submissions are evaluated on unit test coverage (40%), modular code structure (30%), and documentation (30%).",
                "metadata": {
                    "source": "grading-policy.md",
                    "chunk_index": 0,
                    "section": "Evaluation Rubrics",
                    "category": "Academics"
                }
            }
        ]

    # ------------------------------------------------------------------------
    # TASK 1 & 3 TESTS: CHUNK FORMATTING & SOURCE MARKERS
    # ------------------------------------------------------------------------
    def test_format_chunk_standard(self):
        """Validates standard chunk formatting with [index] source#chunk_index marker."""
        formatted = format_chunk(1, self.sample_chunks[0], style="standard")
        self.assertTrue(formatted.startswith("[1] submission-rubric.md#0"))
        self.assertIn("Project submission evidence requirement:", formatted)

    def test_format_chunk_detailed(self):
        """Validates detailed chunk formatting including section and category metadata."""
        formatted = format_chunk(2, self.sample_chunks[1], style="detailed")
        self.assertTrue(formatted.startswith("[2] Source: account-guide.md#0"))
        self.assertIn("Section: 'Authentication'", formatted)
        self.assertIn("Category: Authentication", formatted)
        self.assertIn("Password reset instructions", formatted)

    def test_format_chunk_compact(self):
        """Validates compact marker styling."""
        formatted = format_chunk(3, self.sample_chunks[2], style="compact")
        self.assertTrue(formatted.startswith("[3] campus-guide.md"))

    # ------------------------------------------------------------------------
    # TASK 2 TESTS: TOKEN BUDGET ENFORCEMENT & CONTEXT ASSEMBLY
    # ------------------------------------------------------------------------
    def test_count_tokens_accuracy(self):
        """Validates tiktoken count computation."""
        sample_text = "Knovera RAG Assistant delivers enterprise context injection."
        tokens = count_tokens(sample_text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 5)
        self.assertEqual(count_tokens(""), 0)

    def test_assemble_context_within_budget(self):
        """Validates context assembly when all chunks fit well within the token budget."""
        context, tokens, included, dropped = assemble_context(
            chunks=self.sample_chunks,
            max_tokens=2000
        )
        self.assertEqual(len(included), 4)
        self.assertEqual(len(dropped), 0)
        self.assertGreater(tokens, 50)
        self.assertIn("[1] submission-rubric.md#0", context)
        self.assertIn("[4] grading-policy.md#0", context)
        self.assertIn("\n\n---\n\n", context)

    def test_assemble_context_strict_budget_overflow(self):
        """Validates that low-ranked chunks are dropped when token budget is constrained."""
        # Calculate tokens for 1st chunk
        c1_tokens = count_tokens(format_chunk(1, self.sample_chunks[0]))
        # Set budget to fit chunk 1 but not chunk 2
        tight_budget = c1_tokens + 10
        
        context, tokens, included, dropped = assemble_context(
            chunks=self.sample_chunks,
            max_tokens=tight_budget
        )
        
        self.assertEqual(len(included), 1)
        self.assertEqual(len(dropped), 3)
        self.assertIn("[1] submission-rubric.md#0", context)
        self.assertNotIn("account-guide.md", context)
        self.assertLessEqual(tokens, tight_budget)
        self.assertEqual(dropped[0]["reason"], "EXCEEDED_TOKEN_BUDGET")
        self.assertEqual(dropped[0]["index"], 2)

    def test_assemble_context_empty_input(self):
        """Validates assemble_context returns empty outputs on empty chunk list."""
        context, tokens, included, dropped = assemble_context([])
        self.assertEqual(context, "")
        self.assertEqual(tokens, 0)
        self.assertEqual(included, [])
        self.assertEqual(dropped, [])

    def test_assemble_context_invalid_budget(self):
        """Validates error on non-positive token budget."""
        with self.assertRaises(ValueError):
            assemble_context(self.sample_chunks, max_tokens=0)

    # ------------------------------------------------------------------------
    # TASK 4 TESTS: GROUNDED AUGMENTED PROMPT BUILDING
    # ------------------------------------------------------------------------
    def test_build_prompt_grounding_instructions(self):
        """Validates prompt text contains strict grounding instructions and citations."""
        question = "What evidence is required for project submission?"
        payload = build_prompt(
            question=question,
            retrieved_chunks=self.sample_chunks,
            max_context_tokens=1000
        )
        
        prompt = payload["prompt"]
        self.assertIn("Answer the question using ONLY the provided context below.", prompt)
        self.assertIn("I don't have enough information in the provided context.", prompt)
        self.assertIn("When possible, cite sources using the citation markers like [1] or [2].", prompt)
        self.assertIn("Context:", prompt)
        self.assertIn(f"Question:\n{question}", prompt)
        self.assertIn("[1] submission-rubric.md#0", prompt)

    def test_build_prompt_empty_question_error(self):
        """Validates error when building prompt with empty question."""
        with self.assertRaises(ValueError):
            build_prompt("", self.sample_chunks)
        with self.assertRaises(ValueError):
            build_prompt("   ", self.sample_chunks)

    def test_build_prompt_telemetry_structure(self):
        """Validates complete telemetry metrics returned in prompt payload."""
        question = "How can a learner reset their password?"
        payload = build_prompt(
            question=question,
            retrieved_chunks=self.sample_chunks[:2],
            max_context_tokens=500
        )
        
        self.assertIn("context_tokens", payload)
        self.assertIn("question_tokens", payload)
        self.assertIn("total_prompt_tokens", payload)
        self.assertIn("sources_used", payload)
        self.assertEqual(len(payload["sources_used"]), 2)
        self.assertEqual(payload["sources_used"][0]["marker"], "[1]")
        self.assertEqual(payload["sources_used"][0]["source"], "submission-rubric.md")
        self.assertEqual(payload["chunks_included_count"], 2)
        self.assertEqual(payload["chunks_dropped_count"], 0)

    # ------------------------------------------------------------------------
    # TASK 5 TESTS: CONTEXT INJECTOR OBJECT-ORIENTED ENGINE
    # ------------------------------------------------------------------------
    def test_context_injector_oop_engine(self):
        """Validates ContextInjector class methods and custom configuration."""
        injector = ContextInjector(
            max_context_tokens=1500,
            model_context_window=8000,
            reserved_answer_tokens=1500,
            chunk_style="detailed"
        )
        
        formatted = injector.format_chunk(1, self.sample_chunks[0])
        self.assertIn("Source: submission-rubric.md#0", formatted)
        
        ctx, tokens, inc, drop = injector.assemble(self.sample_chunks)
        self.assertEqual(len(inc), 4)
        
        prompt_payload = injector.augment_prompt(
            question="When does cafeteria serve lunch?",
            retrieved_chunks=self.sample_chunks
        )
        self.assertIn("When does cafeteria serve lunch?", prompt_payload["prompt"])
        self.assertGreater(prompt_payload["total_prompt_tokens"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
