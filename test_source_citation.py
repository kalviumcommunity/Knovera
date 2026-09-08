"""
test_source_citation.py

Comprehensive test suite for Assignment 3.40: Source Citation & Attribution.
Validates:
1. Task 1: Add source references ([1], [2]) in generated answers and prompts.
2. Task 2: Map citations to metadata (source document, chunk ID, chunk index, page, section, char_span, text).
3. Task 3: Verify cited sources against original raw document text and character spans.
4. Task 4: Avoid fabricated citations when supporting evidence is missing or when markers are hallucinated.
5. Task 5: End-to-end answer_with_citations orchestration and payload structure.
"""

import sys
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Ensure Knovera root is in sys.path
_root_dir = str(Path(__file__).resolve().parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.source_tracer import (
    build_citation_map,
    build_cited_prompt,
    answer_with_citations,
    verify_citation,
    detect_fabricated_citations,
    SourceTracer
)
from src.grounded_generator import STANDARD_MISSING_CONTEXT_FALLBACK


class MockRetriever:
    """Mock Retriever for testing citation pipeline execution."""
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return self.chunks[:k]


class TestSourceCitationAndAttribution(unittest.TestCase):
    """Unit tests for Source Citation & Attribution module."""

    def setUp(self):
        """Prepare sample raw text and retrieved chunks with rich metadata."""
        self.raw_document_text = (
            "Knovera Architecture Documentation v2.0\n"
            "Section 1: Security\n"
            "All API requests must pass JWT authentication with RS256 token verification.\n\n"
            "Section 2: Submission Requirements\n"
            "Learners must submit a public GitHub repository PR link and a 3-minute video walkthrough.\n"
            "A minimum score of 60% is required to pass the assignment evaluation."
        )

        self.sample_chunks = [
            {
                "id": "security-spec.md#chunk_0",
                "text": "All API requests must pass JWT authentication with RS256 token verification.",
                "source": "security-spec.md",
                "metadata": {
                    "source": "security-spec.md",
                    "chunk_id": "security-spec.md#chunk_0",
                    "chunk_index": 0,
                    "section": "Security",
                    "page_number": 1,
                    "char_start": 60,
                    "char_end": 136,
                    "doc_title": "Security Specification"
                },
                "score": 0.9120
            },
            {
                "id": "submission-guide.md#chunk_1",
                "text": "Learners must submit a public GitHub repository PR link and a 3-minute video walkthrough.",
                "source": "submission-guide.md",
                "metadata": {
                    "source": "submission-guide.md",
                    "chunk_id": "submission-guide.md#chunk_1",
                    "chunk_index": 1,
                    "section": "Submission Requirements",
                    "page_number": 2,
                    "char_start": 172,
                    "char_end": 260,
                    "doc_title": "Submission Guide"
                },
                "score": 0.8450
            }
        ]

    # ------------------------------------------------------------------------
    # TASK 1: ADD SOURCE REFERENCES
    # ------------------------------------------------------------------------
    def test_task1_build_cited_prompt_format(self):
        """Validates that build_cited_prompt formats context and instructions with [1], [2] markers."""
        prompt = build_cited_prompt("How are API requests authenticated?", self.sample_chunks)
        
        self.assertIn("Answer using only the context below.", prompt)
        self.assertIn("Cite every factual claim using source markers like [1] or [2].", prompt)
        self.assertIn("[1] security-spec.md#0", prompt)
        self.assertIn("[2] submission-guide.md#1", prompt)
        self.assertIn("Question:\nHow are API requests authenticated?", prompt)

    def test_task1_build_cited_prompt_empty_question_error(self):
        """Validates that build_cited_prompt raises ValueError for empty questions."""
        with self.assertRaises(ValueError):
            build_cited_prompt("", self.sample_chunks)

    # ------------------------------------------------------------------------
    # TASK 2: MAP CITATIONS TO METADATA
    # ------------------------------------------------------------------------
    def test_task2_build_citation_map_structure(self):
        """Validates build_citation_map maps markers to document metadata and location."""
        citation_map = build_citation_map(self.sample_chunks)

        self.assertIn("[1]", citation_map)
        self.assertIn("[2]", citation_map)
        
        entry_1 = citation_map["[1]"]
        self.assertEqual(entry_1["source"], "security-spec.md")
        self.assertEqual(entry_1["chunk_id"], "security-spec.md#chunk_0")
        self.assertEqual(entry_1["chunk_index"], 0)
        self.assertEqual(entry_1["section"], "Security")
        self.assertEqual(entry_1["page"], 1)
        self.assertEqual(entry_1["char_span"], "60-136")
        self.assertIn("JWT authentication", entry_1["text"])

        entry_2 = citation_map["[2]"]
        self.assertEqual(entry_2["source"], "submission-guide.md")
        self.assertEqual(entry_2["chunk_id"], "submission-guide.md#chunk_1")
        self.assertEqual(entry_2["section"], "Submission Requirements")

    def test_task2_build_citation_map_empty_chunks(self):
        """Validates build_citation_map returns empty dictionary for empty input."""
        citation_map = build_citation_map([])
        self.assertEqual(citation_map, {})

    # ------------------------------------------------------------------------
    # TASK 3: VERIFY CITED SOURCES
    # ------------------------------------------------------------------------
    def test_task3_verify_citation_valid_marker(self):
        """Validates verify_citation confirms cited marker details and exact character match."""
        citation_map = build_citation_map(self.sample_chunks)
        audit = verify_citation("[1]", citation_map, full_document_text=self.raw_document_text)

        self.assertTrue(audit["is_valid"])
        self.assertEqual(audit["marker"], "[1]")
        self.assertEqual(audit["source"], "security-spec.md")
        self.assertEqual(audit["chunk_id"], "security-spec.md#chunk_0")
        self.assertTrue(audit["verified_exact_match"])
        self.assertIn("JWT authentication", audit["text_snippet"])

    def test_task3_verify_citation_invalid_marker(self):
        """Validates verify_citation handles non-existent citation markers gracefully."""
        citation_map = build_citation_map(self.sample_chunks)
        audit = verify_citation("[99]", citation_map)

        self.assertFalse(audit["is_valid"])
        self.assertEqual(audit["reason"], "MARKER_NOT_IN_CITATION_MAP")
        self.assertFalse(audit["verified_exact_match"])

    def test_task3_trace_chunk_source_audit(self):
        """Validates SourceTracer.trace_chunk_source line range and offset traceback."""
        trace = SourceTracer.trace_chunk_source(self.sample_chunks[0], self.raw_document_text)

        self.assertEqual(trace["source"], "security-spec.md")
        self.assertTrue(trace["verified_exact_match"])
        self.assertEqual(trace["char_span"], "60-136")
        self.assertIn("L3", trace["line_range"])

    # ------------------------------------------------------------------------
    # TASK 4: AVOID FABRICATED CITATIONS
    # ------------------------------------------------------------------------
    def test_task4_answer_with_citations_no_source_fallback(self):
        """Validates that when no chunks are retrieved, a fallback with empty citations is returned."""
        result = answer_with_citations(
            question="What is the quantum encryption key?",
            chunks=[],
            use_api=False
        )

        self.assertEqual(result["status"], "MISSING_CONTEXT_FALLBACK")
        self.assertEqual(result["answer"], STANDARD_MISSING_CONTEXT_FALLBACK)
        self.assertEqual(result["citations"], {})
        self.assertEqual(result["cited_markers"], [])

    def test_task4_detect_fabricated_citations(self):
        """Validates detect_fabricated_citations flags citation markers absent from retrieved context."""
        valid_markers = ["[1]", "[2]"]
        answer_with_fake = "API requests use JWT [1], but secret ops use quantum locks [5]."
        
        audit = detect_fabricated_citations(answer_with_fake, valid_markers)

        self.assertTrue(audit["has_fabricated_citations"])
        self.assertIn("[5]", audit["fabricated_citations"])
        self.assertIn("[1]", audit["valid_citations"])

    def test_task4_detect_fabricated_citations_clean(self):
        """Validates detect_fabricated_citations passes when all citations are valid."""
        valid_markers = ["[1]", "[2]"]
        clean_answer = "API requests use JWT [1] and submissions require video [2]."

        audit = detect_fabricated_citations(clean_answer, valid_markers)

        self.assertFalse(audit["has_fabricated_citations"])
        self.assertEqual(audit["fabricated_citations"], [])

    # ------------------------------------------------------------------------
    # TASK 5: END-TO-END ANSWER WITH CITATIONS
    # ------------------------------------------------------------------------
    def test_task5_answer_with_citations_success(self):
        """Validates end-to-end answer_with_citations generates grounded response with citations."""
        question = "What evidence is required for submission?"
        retriever = MockRetriever(chunks=self.sample_chunks)

        result = answer_with_citations(
            question=question,
            retriever=retriever,
            use_api=False
        )

        self.assertEqual(result["status"], "GROUNDED_SUCCESS")
        self.assertIn("GitHub", result["answer"])
        self.assertIn("[1]", result["citations"])
        self.assertIn("[2]", result["citations"])
        self.assertFalse(result.get("fabricated_citations_detected", False))

    def test_task5_sourcetracer_class_static_methods(self):
        """Validates class-level static methods on SourceTracer."""
        cit_map = SourceTracer.build_citation_map(self.sample_chunks)
        self.assertEqual(len(cit_map), 2)

        prompt = SourceTracer.build_cited_prompt("Test question?", self.sample_chunks)
        self.assertIn("Test question?", prompt)

        ans = SourceTracer.answer_with_citations("What evidence?", chunks=self.sample_chunks, use_api=False)
        self.assertEqual(ans["status"], "GROUNDED_SUCCESS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
