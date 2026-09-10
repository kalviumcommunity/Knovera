"""
grounded_generation_demo.py

Demonstration script for Assignment 3.39: Grounded Answer Generation.
Demonstrates:
1. Generating grounded answers strictly from injected retrieved context.
2. Verifying answer grounding and source accuracy (claim-to-chunk word overlap & citations).
3. Handling missing context gracefully with explicit fallback ("I don't have enough information in the provided context.").
4. Comparing With Retrieval (Grounded RAG) vs Without Retrieval (Ungrounded baseline model).
5. Exporting full structured demonstration payload to 'sample_grounded_answers.json'.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.grounded_generator import (
    generate_grounded_answer,
    generate_ungrounded_answer,
    verify_grounding,
    answer_query,
    compare_grounded_vs_ungrounded,
    GroundedGenerator,
    STANDARD_MISSING_CONTEXT_FALLBACK
)

# Benchmark document corpus for Knovera
CORPUS_KNOWLEDGE_BASE = [
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
        "score": 0.8924,
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
        "score": 0.7812,
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
        "score": 0.6945,
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
        "score": 0.6120,
        "rank": 4
    }
]


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def demonstrate_task1_and_2():
    """Demonstrates Task 1 (Grounded Generation) and Task 2 (Source Accuracy Verification)."""
    print_header("Task 1 & 2: Grounded Generation from Injected Context & Accuracy Check")
    
    question = "What evidence is required for project submission?"
    chunks = [CORPUS_KNOWLEDGE_BASE[0], CORPUS_KNOWLEDGE_BASE[3]]  # submission rubric & grading policy
    
    print(f"Question: \"{question}\"")
    print(f"Injected Chunks Count: {len(chunks)}")
    for c in chunks:
        print(f"  * [{c['metadata']['source']}] {c['text'][:90]}...")
        
    print("\n--- Generating Grounded Answer ---")
    result = generate_grounded_answer(
        question=question,
        retrieved_chunks=chunks,
        use_api=False
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Sources Used: {[s['source'] for s in result['sources']]}")
    
    print("\n--- Grounding Verification Audit ---")
    verification = result["verification_details"]
    print(f"  * Grounded Status         : {verification['is_grounded']}")
    print(f"  * Grounding Alignment Score: {verification['grounding_score'] * 100:.1f}%")
    print(f"  * Citation Markers Found   : {verification['cited_markers']}")
    print(f"  * Supported Claim Tokens   : {verification['supported_claim_tokens']} / {verification['total_claim_tokens']}")
    print(f"  * Audit Reason             : {verification['reason']}")
    
    return result


def demonstrate_task3_fallback():
    """Demonstrates Task 3: Missing-Context Fallback."""
    print_header("Task 3: Missing-Context Fallback Handling")
    
    out_of_domain_question = "What is the warp drive maintenance schedule for the starship Enterprise?"
    print(f"Question: \"{out_of_domain_question}\"")
    print("Simulating Empty Context Retrieval (chunks = [])...")
    
    fallback_result = generate_grounded_answer(
        question=out_of_domain_question,
        retrieved_chunks=[],
        use_api=False
    )
    
    print(f"\nAnswer: \"{fallback_result['answer']}\"")
    print(f"Sources: {fallback_result['sources']}")
    print(f"Status: {fallback_result['status']}")
    print(f"Anti-Hallucination Verified: {fallback_result['grounding_verified']}")
    
    return fallback_result


def demonstrate_task4_comparison():
    """Demonstrates Task 4: Comparative Evaluation (With Retrieval vs Without Retrieval)."""
    print_header("Task 4: Side-by-Side Comparison (With vs Without Retrieval)")
    
    comparison_queries = [
        {
            "question": "What evidence is required for project submission?",
            "chunks": [CORPUS_KNOWLEDGE_BASE[0], CORPUS_KNOWLEDGE_BASE[3]]
        },
        {
            "question": "How can a learner reset their password?",
            "chunks": [CORPUS_KNOWLEDGE_BASE[1]]
        },
        {
            "question": "When does the cafeteria menu change?",
            "chunks": [CORPUS_KNOWLEDGE_BASE[2]]
        }
    ]
    
    comparisons = []
    
    for idx, item in enumerate(comparison_queries, start=1):
        q = item["question"]
        chk = item["chunks"]
        
        print(f"\n[QUERY #{idx}] \"{q}\"")
        comp = compare_grounded_vs_ungrounded(
            question=q,
            retrieved_chunks=chk,
            use_api=False
        )
        
        print(f"  WITH RETRIEVAL (Grounded RAG):")
        print(f"    -> Answer         : {comp['with_retrieval']['answer']}")
        print(f"    -> Sources        : {comp['with_retrieval']['sources']}")
        print(f"    -> Grounding Score: {comp['with_retrieval']['grounding_score'] * 100:.1f}%")
        print(f"    -> Has Citations  : {comp['with_retrieval']['has_citations']}")
        
        print(f"  WITHOUT RETRIEVAL (Ungrounded Baseline):")
        print(f"    -> Answer         : {comp['without_retrieval']['answer']}")
        print(f"    -> Sources        : {comp['without_retrieval']['sources']}")
        print(f"    -> Grounding Score: {comp['without_retrieval']['grounding_score'] * 100:.1f}%")
        print(f"    -> Has Citations  : {comp['without_retrieval']['has_citations']}")
        
        comparisons.append(comp)
        
    return comparisons


def main():
    print_header("Assignment 3.39: Grounded Answer Generation Demonstration")
    
    # 1. Grounded Generation and Accuracy Verification
    task1_res = demonstrate_task1_and_2()
    
    # 2. Missing-Context Fallback
    fallback_res = demonstrate_task3_fallback()
    
    # 3. With vs Without Retrieval Comparison
    comparisons_res = demonstrate_task4_comparison()
    
    # 4. Task 5: Export structured JSON payload
    export_payload = {
        "assignment": "3.39 Grounded Answer Generation",
        "sample_grounded_answer": task1_res,
        "sample_missing_context_fallback": fallback_res,
        "comparisons_with_vs_without_retrieval": comparisons_res,
        "supporting_chunks_corpus": CORPUS_KNOWLEDGE_BASE
    }
    
    json_path = "sample_grounded_answers.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n[EXPORT] Successfully generated '{json_path}'")
    
    print("\n" + "=" * 80)
    print("  GROUNDED ANSWER GENERATION DEMO COMPLETE - VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
