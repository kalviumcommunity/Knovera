"""
context_injection_demo.py

Demonstration script for Assignment 3.38: Context Injection & Prompt Augmentation.
Demonstrates:
1. Formatting retrieved chunks with explicit indexed source markers ([1] source#chunk_index).
2. Assembling context within strict token budgets using tiktoken BPE token counting.
3. Managing token overflow: Dropping/trimming low-ranked chunks when exceeding context limits.
4. Building fully augmented prompts with strict grounding instructions and citation rules.
5. Exporting sample augmented prompts to 'sample_augmented_prompt.json' and 'sample_augmented_prompt.txt'.
"""

import os
import sys
import json
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

# Representative multi-domain retrieved chunks
RETRIEVED_CHUNKS_SAMPLE = [
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
        "score": 0.8845,
        "rank": 1
    },
    {
        "id": "account-guide.md#chunk_0",
        "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal. A secure verification link is sent to the registered email address immediately.",
        "metadata": {
            "source": "account-guide.md",
            "chunk_index": 0,
            "section": "Authentication",
            "doc_title": "Account & Security Guide",
            "category": "Authentication"
        },
        "score": 0.7632,
        "rank": 2
    },
    {
        "id": "campus-guide.md#chunk_0",
        "text": "Campus cafeteria hours and lunch menu: The cafeteria operates from 11:30 AM to 2:30 PM offering hot artisan wraps, organic salads, and rotating daily chef specials every weekday.",
        "metadata": {
            "source": "campus-guide.md",
            "chunk_index": 0,
            "section": "Dining Facilities",
            "doc_title": "Campus Amenities Guide",
            "category": "Campus Life"
        },
        "score": 0.6514,
        "rank": 3
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
        "score": 0.5891,
        "rank": 4
    }
]


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def demonstrate_chunk_formatting():
    """Demonstrates Task 1 & Task 3: Chunk formatting styles and source markers."""
    print_header("Task 1 & 3: Chunk Formatting & Source Markers")
    chunk = RETRIEVED_CHUNKS_SAMPLE[0]
    
    print("\n--- Standard Style ([index] source#chunk_index) ---")
    standard_fmt = format_chunk(1, chunk, style="standard")
    print(standard_fmt)
    print(f"Tokens: {count_tokens(standard_fmt)}")
    
    print("\n--- Detailed Style ([index] Source: source#chunk_index | Section | Category) ---")
    detailed_fmt = format_chunk(1, chunk, style="detailed")
    print(detailed_fmt)
    print(f"Tokens: {count_tokens(detailed_fmt)}")
    
    print("\n--- Compact Style ([index] source) ---")
    compact_fmt = format_chunk(1, chunk, style="compact")
    print(compact_fmt)
    print(f"Tokens: {count_tokens(compact_fmt)}")


def demonstrate_token_budget_enforcement():
    """Demonstrates Task 2: Token budget calculation and overflow handling."""
    print_header("Task 2: Token Budget Allocation & Overflow Enforcement")
    
    print("Total Model Context Window (e.g., 8,000 tokens):")
    print("  * Max Injected Context Budget : 5,000 tokens (62.5%)")
    print("  * Reserved Model Generation   : 1,500 tokens (18.75%)")
    print("  * System Instructions/Question:   800 tokens (10.0%)")
    print("  * Safety Buffer Margin        :   700 tokens (8.75%)\n")
    
    # Scenario A: Generous budget (all chunks fit)
    print("[SCENARIO A] Generous Budget (max_tokens = 2000):")
    ctx_a, tokens_a, inc_a, drop_a = assemble_context(RETRIEVED_CHUNKS_SAMPLE, max_tokens=2000)
    print(f"  -> Total Context Tokens: {tokens_a}")
    print(f"  -> Chunks Included     : {len(inc_a)} / {len(RETRIEVED_CHUNKS_SAMPLE)}")
    print(f"  -> Chunks Dropped      : {len(drop_a)}")
    
    # Scenario B: Strict tight budget (triggers overflow drop)
    c1_tok = count_tokens(format_chunk(1, RETRIEVED_CHUNKS_SAMPLE[0]))
    c2_tok = count_tokens(format_chunk(2, RETRIEVED_CHUNKS_SAMPLE[1]))
    strict_limit = c1_tok + c2_tok + 15  # Fits chunks 1 and 2, but drops 3 and 4
    
    print(f"\n[SCENARIO B] Strict Constrained Budget (max_tokens = {strict_limit}):")
    ctx_b, tokens_b, inc_b, drop_b = assemble_context(RETRIEVED_CHUNKS_SAMPLE, max_tokens=strict_limit)
    print(f"  -> Total Context Tokens: {tokens_b} (Budget: {strict_limit})")
    print(f"  -> Chunks Included     : {len(inc_b)} (Ranks: {[c['metadata']['source'] for c in inc_b]})")
    print(f"  -> Chunks Dropped      : {len(drop_b)}")
    for d in drop_b:
        print(f"     * Dropped Chunk #{d['index']} ({d['chunk']['metadata']['source']}) - Reason: {d['reason']} ({d['token_count']} tokens)")


def demonstrate_prompt_augmentation():
    """Demonstrates Task 4 & Task 5: Building grounded augmented prompt."""
    print_header("Task 4 & 5: Grounded Prompt Augmentation & Sample Export")
    
    sample_question = "What evidence is required for project submission and what are the grading criteria?"
    
    prompt_payload = build_prompt(
        question=sample_question,
        retrieved_chunks=RETRIEVED_CHUNKS_SAMPLE,
        max_context_tokens=1500,
        chunk_style="detailed"
    )
    
    print("--- Rendered Grounded Prompt ---")
    print(prompt_payload["prompt"])
    print("-" * 50)
    print("\n--- Prompt Telemetry & Budget Breakdown ---")
    print(f"  * Question Tokens     : {prompt_payload['question_tokens']}")
    print(f"  * Context Tokens      : {prompt_payload['context_tokens']}")
    print(f"  * Total Prompt Tokens : {prompt_payload['total_prompt_tokens']}")
    print(f"  * Chunks Included     : {prompt_payload['chunks_included_count']}")
    print(f"  * Chunks Dropped      : {prompt_payload['chunks_dropped_count']}")
    print(f"  * Sources Mapped      : {len(prompt_payload['sources_used'])}")
    for s in prompt_payload["sources_used"]:
        print(f"     - {s['marker']} Source: {s['source']} (Chunk #{s.get('chunk_index', 0)})")
        
    return prompt_payload


def export_sample_artifacts(payload: Dict[str, Any]):
    """Exports structured JSON and formatted text artifact files."""
    json_path = "sample_augmented_prompt.json"
    txt_path = "sample_augmented_prompt.txt"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[EXPORT] Successfully generated '{json_path}'")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KNOVERA RAG ASSISTANT - SAMPLE AUGMENTED PROMPT\n")
        f.write("=" * 80 + "\n\n")
        f.write(payload["prompt"] + "\n\n")
        f.write("-" * 80 + "\n")
        f.write("TELEMETRY METRICS:\n")
        f.write(f"Question Tokens     : {payload['question_tokens']}\n")
        f.write(f"Context Tokens      : {payload['context_tokens']}\n")
        f.write(f"Total Prompt Tokens : {payload['total_prompt_tokens']}\n")
        f.write(f"Max Context Budget  : {payload['max_context_tokens']}\n")
        f.write(f"Chunks Included     : {payload['chunks_included_count']}\n")
        f.write(f"Chunks Dropped      : {payload['chunks_dropped_count']}\n")
        f.write("-" * 80 + "\n")
    print(f"[EXPORT] Successfully generated '{txt_path}'")


def main():
    print_header("Assignment 3.38: Context Injection & Prompt Augmentation")
    
    # 1. Chunk formatting styles & source markers
    demonstrate_chunk_formatting()
    
    # 2. Token budget enforcement & overflow handling
    demonstrate_token_budget_enforcement()
    
    # 3. Grounded prompt augmentation
    payload = demonstrate_prompt_augmentation()
    
    # 4. Export sample files
    export_sample_artifacts(payload)
    
    print("\n" + "=" * 80)
    print("  CONTEXT INJECTION & PROMPT AUGMENTATION DEMO COMPLETE - VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
