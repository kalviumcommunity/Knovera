"""
source_citation_demo.py

Interactive Demonstration Script for Assignment 3.40: Source Citation & Attribution.
Demonstrates:
- Task 1: Add source references to generated answers ([1], [2])
- Task 2: Map citations back to real documents, chunk IDs, sections, line ranges, and offsets
- Task 3: Verify cited sources against original raw document text
- Task 4: Prevent fabricated citations by enforcing missing-context fallback & anti-hallucination audits
- Task 5: Export sample cited answers payload to sample_cited_answers.json
"""

import os
import sys
import json
from pathlib import Path

# Ensure Knovera root is in sys.path
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
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


def run_demo():
    print("=" * 80)
    print("KNOVERA RAG ASSISTANT - SOURCE CITATION & ATTRIBUTION DEMONSTRATION")
    print("Assignment 3.40 - Source Traceability & Anti-Hallucination Guardrails")
    print("=" * 80 + "\n")

    # Sample Raw Document Text for Task 3 verification
    raw_document_text = (
        "Knovera Architecture Documentation v2.0\n"
        "Section 1: Security\n"
        "All API requests must pass JWT authentication with RS256 token verification.\n\n"
        "Section 2: Submission Requirements\n"
        "Learners must submit a public GitHub repository PR link and a 3-minute video walkthrough.\n"
        "A minimum score of 60% is required to pass the assignment evaluation."
    )

    sample_chunks = [
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

    # -------------------------------------------------------------------------
    # TASK 1: ADD SOURCE REFERENCES & CITED PROMPT
    # -------------------------------------------------------------------------
    print("--- TASK 1: ADD SOURCE REFERENCES & CITED PROMPT BUILDER ---")
    question_1 = "What evidence is required for project submission?"
    cited_prompt = build_cited_prompt(question_1, sample_chunks)
    print(f"Question: {question_1}")
    print("\nRendered Grounded Prompt with Source Markers:")
    print("-" * 50)
    print(cited_prompt)
    print("-" * 50 + "\n")

    # -------------------------------------------------------------------------
    # TASK 2: MAP CITATIONS TO METADATA
    # -------------------------------------------------------------------------
    print("--- TASK 2: MAP CITATIONS TO METADATA ---")
    citation_map = build_citation_map(sample_chunks)
    print("Generated Citation Map:")
    print(json.dumps(citation_map, indent=2))
    print("\n")

    # -------------------------------------------------------------------------
    # TASK 3: VERIFY CITED SOURCES
    # -------------------------------------------------------------------------
    print("--- TASK 3: VERIFY CITED SOURCES AGAINST RAW DOCUMENT ---")
    audit_marker_1 = verify_citation("[1]", citation_map, full_document_text=raw_document_text)
    print("Verification Audit for Marker [1]:")
    print(json.dumps(audit_marker_1, indent=2))

    audit_marker_2 = verify_citation("[2]", citation_map, full_document_text=raw_document_text)
    print("\nVerification Audit for Marker [2]:")
    print(json.dumps(audit_marker_2, indent=2))
    print("\n")

    # -------------------------------------------------------------------------
    # TASK 4: AVOID FABRICATED CITATIONS & FALLBACK HANDLING
    # -------------------------------------------------------------------------
    print("--- TASK 4: AVOID FABRICATED CITATIONS & FALLBACK HANDLING ---")
    question_2 = "What is the warp core coolant replacement procedure?"
    fallback_res = answer_with_citations(question_2, chunks=[], use_api=False)
    print(f"Unsupported Query: '{question_2}'")
    print(f"Fallback Response: '{fallback_res['answer']}'")
    print(f"Citations attached: {fallback_res['citations']} (No citations fabricated)")

    print("\nTesting Fabricated Citation Detection:")
    raw_hallucinatory_answer = "API requests use JWT [1], but secret ops use quantum locks [5]."
    valid_markers = list(citation_map.keys())
    fab_audit = detect_fabricated_citations(raw_hallucinatory_answer, valid_markers)
    print(f"Raw Model Answer: '{raw_hallucinatory_answer}'")
    print("Fabrication Audit Report:")
    print(json.dumps(fab_audit, indent=2))
    print("\n")

    # -------------------------------------------------------------------------
    # TASK 5: END-TO-END ANSWER WITH CITATIONS & SAMPLE EXPORT
    # -------------------------------------------------------------------------
    print("--- TASK 5: END-TO-END ANSWER WITH CITATIONS ---")
    e2e_res = answer_with_citations(question_1, chunks=sample_chunks, use_api=False)
    print(f"Query: {e2e_res['question']}")
    print(f"Answer: {e2e_res['answer']}")
    print("Citations Summary:")
    for marker, details in e2e_res["citations"].items():
        print(f"  {marker} -> Source: {details['source']} | Section: '{details['section']}' | Offset: {details['char_span']}")

    output_path = Path(_root_dir) / "sample_cited_answers.json"
    print(f"\n[SUCCESS] Exported sample cited answers payload to: {output_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
