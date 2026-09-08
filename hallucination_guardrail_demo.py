"""
hallucination_guardrail_demo.py

Interactive Demonstration Script for Assignment 3.41: Hallucination Guardrails & Refusal Handling.
Demonstrates:
- Task 1: Detect weak retrieval using signals (empty results, low similarity score < 0.72, insufficient supporting chunks)
- Task 2: Return safe refusal response ("I don't have enough reliable context to answer that.") when context is weak
- Task 3: Apply relevance thresholds (MIN_TOP_SCORE = 0.72, MIN_SUPPORTING_CHUNKS = 1) to control refusal behavior
- Task 4: Produce confident grounded answers with citations when strong supporting context exists
- Task 5: Export sample refusal and answer cases payload to sample_guardrail_outputs.json
"""

import os
import sys
import json
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


def run_demo():
    print("=" * 80)
    print("KNOVERA RAG ASSISTANT - HALLUCINATION GUARDRAILS & REFUSAL HANDLER DEMO")
    print("Assignment 3.41 - Hallucination Prevention & Pre-Generation Context Auditing")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # TEST DATASETS & RETRIEVAL SCENARIOS
    # -------------------------------------------------------------------------
    
    # Strong Context: High similarity scores (above 0.72 threshold)
    strong_chunks = [
        {
            "id": "submission_guide.md#chunk_1",
            "text": "Learners must submit a public GitHub repository PR link and a 3-minute video walkthrough.",
            "source": "submission_guide.md",
            "metadata": {
                "source": "submission_guide.md",
                "chunk_id": "submission_guide.md#chunk_1",
                "section": "Submission Requirements",
                "page_number": 2,
                "doc_title": "Knovera Submission Manual"
            },
            "score": 0.8950
        },
        {
            "id": "submission_guide.md#chunk_2",
            "text": "The GitHub PR must be public, open at submission time, and pass automated unit tests with a score >= 60%.",
            "source": "submission_guide.md",
            "metadata": {
                "source": "submission_guide.md",
                "chunk_id": "submission_guide.md#chunk_2",
                "section": "Passing Criteria",
                "page_number": 2,
                "doc_title": "Knovera Submission Manual"
            },
            "score": 0.8120
        }
    ]

    # Weak Context (Low Similarity): Top score is 0.4500 (below 0.72 threshold)
    weak_low_score_chunks = [
        {
            "id": "refund_policy.md#chunk_0",
            "text": "Products purchased online may be eligible for standard store credit within 14 calendar days.",
            "source": "refund_policy.md",
            "metadata": {
                "source": "refund_policy.md",
                "chunk_id": "refund_policy.md#chunk_0",
                "section": "General Terms"
            },
            "score": 0.4500
        },
        {
            "id": "shipping_policy.md#chunk_1",
            "text": "Standard ground shipping takes 3-5 business days for regional deliveries.",
            "source": "shipping_policy.md",
            "metadata": {
                "source": "shipping_policy.md",
                "chunk_id": "shipping_policy.md#chunk_1",
                "section": "Shipping Rates"
            },
            "score": 0.3800
        }
    ]

    # Weak Context (Empty Results)
    empty_chunks = []

    # -------------------------------------------------------------------------
    # TASK 1 & 3: DETECT WEAK RETRIEVAL USING RELEVANCE THRESHOLDS
    # -------------------------------------------------------------------------
    print("--- TASK 1 & 3: RETRIEVAL STRENGTH DIAGNOSTICS & RELEVANCE THRESHOLDS ---")
    print(f"Active Thresholds: MIN_TOP_SCORE = {MIN_TOP_SCORE}, MIN_SUPPORTING_CHUNKS = {MIN_SUPPORTING_CHUNKS}\n")

    eval_strong = evaluate_retrieval_quality(strong_chunks)
    print("1. Strong Context Retrieval Diagnostics:")
    print(f"   - Is Strong Context: {eval_strong['is_strong']}")
    print(f"   - Top Similarity Score: {eval_strong['top_score']}")
    print(f"   - Supporting Chunks Count: {eval_strong['strong_chunks_count']}")
    print(f"   - Status Code: {eval_strong['status_code']}")
    print(f"   - Decision Reason: {eval_strong['refusal_reason']}\n")

    eval_weak = evaluate_retrieval_quality(weak_low_score_chunks)
    print("2. Weak Context (Low Similarity) Retrieval Diagnostics:")
    print(f"   - Is Strong Context: {eval_weak['is_strong']}")
    print(f"   - Top Similarity Score: {eval_weak['top_score']}")
    print(f"   - Supporting Chunks Count: {eval_weak['strong_chunks_count']}")
    print(f"   - Status Code: {eval_weak['status_code']}")
    print(f"   - Decision Reason: {eval_weak['refusal_reason']}\n")

    eval_empty = evaluate_retrieval_quality(empty_chunks)
    print("3. Empty Context Retrieval Diagnostics:")
    print(f"   - Is Strong Context: {eval_empty['is_strong']}")
    print(f"   - Status Code: {eval_empty['status_code']}")
    print(f"   - Decision Reason: {eval_empty['refusal_reason']}\n")

    # -------------------------------------------------------------------------
    # TASK 2 & 4: GUARDED ANSWER GENERATION (REFUSAL VS CONFIDENT ANSWER)
    # -------------------------------------------------------------------------
    print("--- TASK 2 & 4: GUARDED ANSWER SYNTHESIS & SAFE REFUSALS ---")

    # Confident Answer Case
    q_answerable = "What evidence is required for project submission?"
    answer_case = guarded_answer(q_answerable, strong_chunks, use_api=False)
    print(f"Scenario A (Answerable Query): '{q_answerable}'")
    print(f"  Status: {answer_case['status']}")
    print(f"  Guardrail Triggered: {answer_case['guardrail_triggered']}")
    print(f"  Answer Output: {answer_case['answer']}")
    print(f"  Citations Count: {len(answer_case.get('citations', {}))}")
    print(f"  Sources List: {[s.get('source') for s in answer_case.get('sources', [])]}\n")

    # Refusal Case (Low Score Query)
    q_unsupported = "What is the refund policy for a product not in this corpus?"
    refusal_case_1 = guarded_answer(q_unsupported, weak_low_score_chunks, use_api=False)
    print(f"Scenario B (Low Similarity Query): '{q_unsupported}'")
    print(f"  Status: {refusal_case_1['status']}")
    print(f"  Guardrail Triggered: {refusal_case_1['guardrail_triggered']}")
    print(f"  Answer Output: {refusal_case_1['answer']}")
    print(f"  Reason: {refusal_case_1['retrieval_metrics']['refusal_reason']}")
    print(f"  Citations: {refusal_case_1['citations']}")
    print(f"  Sources: {refusal_case_1['sources']}\n")

    # Refusal Case (Empty Context Query)
    q_empty = "How do I calibrate quantum flux sensors?"
    refusal_case_2 = guarded_answer(q_empty, empty_chunks, use_api=False)
    print(f"Scenario C (Empty Retrieval Query): '{q_empty}'")
    print(f"  Status: {refusal_case_2['status']}")
    print(f"  Guardrail Triggered: {refusal_case_2['guardrail_triggered']}")
    print(f"  Answer Output: {refusal_case_2['answer']}")
    print(f"  Reason: {refusal_case_2['retrieval_metrics']['refusal_reason']}\n")

    # -------------------------------------------------------------------------
    # CLASS-BASED GUARDRAIL MANAGEMENT & TELEMETRY
    # -------------------------------------------------------------------------
    print("--- GUARDRAIL TELEMETRY & THRESHOLD EVALUATION ---")
    guardrail_mgr = HallucinationGuardrail(min_top_score=0.72)
    guardrail_mgr.generate(q_answerable, strong_chunks)
    guardrail_mgr.generate(q_unsupported, weak_low_score_chunks)
    guardrail_mgr.generate(q_empty, empty_chunks)

    telemetry = guardrail_mgr.get_telemetry()
    print(f"Guardrail Telemetry Summary: {json.dumps(telemetry, indent=2)}\n")

    # -------------------------------------------------------------------------
    # TASK 5: EXPORT SAMPLE PAYLOAD TO sample_guardrail_outputs.json
    # -------------------------------------------------------------------------
    print("--- TASK 5: EXPORTING SAMPLE PAYLOADS TO sample_guardrail_outputs.json ---")
    export_payload = {
        "metadata": {
            "assignment": "3.41 Hallucination Guardrails & Refusal Handling",
            "min_top_score_threshold": MIN_TOP_SCORE,
            "min_supporting_chunks_threshold": MIN_SUPPORTING_CHUNKS,
            "standard_refusal_message": STANDARD_SAFE_REFUSAL
        },
        "answer_case": answer_case,
        "refusal_cases": {
            "low_similarity_refusal": refusal_case_1,
            "empty_context_refusal": refusal_case_2
        },
        "telemetry_snapshot": telemetry
    }

    output_path = os.path.join(_root_dir, "sample_guardrail_outputs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"SUCCESS: Sample guardrail outputs exported to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
