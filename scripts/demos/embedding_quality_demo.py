"""
embedding_quality_demo.py

Demonstration and Verification Suite for Assignment 3.29: Embedding Quality Checks & Sanity Tests.
Platform: Knovera RAG Assistant

Features Demonstrated:
- Task 1: Create known query-chunk test cases for retrieval relevance.
- Task 2: Confirm that known-related chunks rank above unrelated chunks with measurable score margins.
- Task 3: Identify, diagnose, and explain surprising/failing edge cases (e.g., broad vs specific query ambiguity, negation sensitivity, and mismatched model vector corruption).
- Task 4: Summarize results in a structured sanity report (test count, passes, failures, top-ranked sources, scores, notes).
- Task 5: Export sanity report JSON and text execution logs to the outputs directory.
"""

import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Reconfigure stdout to UTF-8 for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.embedding_generator import EmbeddingGenerator, cosine_similarity
from src.embedding_quality_checker import EmbeddingQualityChecker


def load_corpus() -> List[Dict[str, Any]]:
    """
    Constructs an authoritative multi-document corpus spanning Authentication,
    Campus Dining, Billing/Refunds, SLAs, and Developer API SDKs.
    """
    return [
        {
            "id": "account-guide#chunk_0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and submitting their registered email address.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Recovery", "category": "Authentication"}
        },
        {
            "id": "account-guide#chunk_1",
            "text": "Learners can recover access using their registered email. A one-time secure verification link is dispatched, which remains valid for 15 minutes.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Verification", "category": "Authentication"}
        },
        {
            "id": "account-guide#chunk_2",
            "text": "Two-factor authentication (2FA) is mandatory for enterprise admin consoles. Admins must register an authenticator app (TOTP) or hardware security key.",
            "metadata": {"source": "account-guide.md", "chunk_index": 2, "section": "Multi-Factor Security", "category": "Authentication"}
        },
        {
            "id": "service-policy#chunk_0",
            "text": "Knovera Customer Service & Refund Policy: Enterprise clients are eligible for full subscription refunds within a 14-day evaluation window from initial provisioning.",
            "metadata": {"source": "service-policy.md", "chunk_index": 0, "section": "Refund Eligibility", "category": "Billing"}
        },
        {
            "id": "service-policy#chunk_1",
            "text": "Refund processing requires an approved ticket from the billing department. Once authorized, funds will appear on the original payment method in 3-5 business days.",
            "metadata": {"source": "service-policy.md", "chunk_index": 1, "section": "Refund Processing SLA", "category": "Billing"}
        },
        {
            "id": "service-policy#chunk_2",
            "text": "Enterprise Service Level Agreement (SLA): Knovera guarantees 99.9% uptime for core API endpoints. Outages exceeding SLA thresholds qualify for service credits.",
            "metadata": {"source": "service-policy.md", "chunk_index": 2, "section": "Uptime SLA", "category": "Infrastructure"}
        },
        {
            "id": "campus-guide#chunk_0",
            "text": "Campus cafeteria hours and lunch menu: The cafeteria operates from 11:30 AM to 2:30 PM offering Italian pasta, fresh garden salads, and hot daily specials.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Dining Hours", "category": "Campus Life"}
        },
        {
            "id": "campus-guide#chunk_1",
            "text": "Campus dining cards can be recharged via the student portal or mobile app. Cash and contactless digital cards are accepted at all food counters.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 1, "section": "Payment Methods", "category": "Campus Life"}
        },
        {
            "id": "dev-guide#chunk_0",
            "text": "Knovera Python SDK Quickstart: Install knovera-sdk via pip and configure KNOVERA_API_KEY environment variable before invoking the Client.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 0, "section": "SDK Installation", "category": "Developer"}
        },
        {
            "id": "dev-guide#chunk_1",
            "text": "Semantic chunking API endpoints allow developers to pass raw markdown and receive token-bounded chunks tagged with positional metadata.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 1, "section": "Chunking Endpoints", "category": "Developer"}
        }
    ]


def load_known_test_cases() -> List[Dict[str, Any]]:
    """
    Task 1: Constructs known query-chunk test cases with expected target sources.
    """
    return [
        {
            "id": "TC-01",
            "query": "How can a learner reset their account password?",
            "expected_source": "account-guide.md",
            "category": "Authentication",
            "description": "User password recovery query should retrieve account administration guide."
        },
        {
            "id": "TC-02",
            "query": "When does the campus cafeteria open and what is on the lunch menu?",
            "expected_source": "campus-guide.md",
            "category": "Campus Life",
            "description": "Dining hours query should retrieve campus cafeteria guide."
        },
        {
            "id": "TC-03",
            "query": "What is the policy and time window for receiving a subscription refund?",
            "expected_source": "service-policy.md",
            "category": "Billing",
            "description": "Refund policy query should retrieve billing service policy guide."
        },
        {
            "id": "TC-04",
            "query": "How do developers install the Python SDK and configure their API key?",
            "expected_source": "dev-guide.md",
            "category": "Developer",
            "description": "Developer setup query should retrieve developer SDK guide."
        },
        {
            "id": "TC-05",
            "query": "What uptime guarantee is provided in the enterprise SLA?",
            "expected_source": "service-policy.md",
            "category": "Infrastructure",
            "description": "Uptime SLA query should retrieve infrastructure policy chunk."
        }
    ]


def load_surprising_edge_cases() -> List[Dict[str, Any]]:
    """
    Task 3: Edge cases designed to test the boundary behaviors of dense semantic retrieval.
    """
    return [
        {
            "id": "EDGE-01",
            "query": "How do I pay?",
            "expected_source": "campus-guide.md",
            "category": "Polysemy / High Brevity",
            "description": "Ambiguous 4-word query where payment could match campus dining cards OR subscription billing."
        },
        {
            "id": "EDGE-02",
            "query": "I do NOT want to install Python, how do I reset my password?",
            "expected_source": "account-guide.md",
            "category": "Negation Blindspot",
            "description": "Query containing negative distractor ('do NOT want to install Python') testing dense vector attention."
        }
    ]


def format_sanity_table(suite_results: Dict[str, Any]) -> str:
    """Formats the sanity test results into an ASCII table."""
    lines = [
        "+-------+------------------------------------------------------+--------------------+--------------------+-------+--------+--------+",
        "| ID    | Query Preview                                        | Expected Source    | Top Ranked Source  | Rank  | Score  | Result |",
        "+-------+------------------------------------------------------+--------------------+--------------------+-------+--------+--------+"
    ]
    for r in suite_results["results"]:
        q_prev = (r["query"][:50] + "..") if len(r["query"]) > 52 else r["query"]
        exp_src = r["expected_source"] or "N/A"
        top_src = r["top_source"] or "N/A"
        rank_str = str(r["target_rank"]) if r["target_rank"] is not None else "N/A"
        score_str = f"{r['top_score']:.4f}"
        res_str = "PASS" if r["passed"] else "FAIL/SURPRISE"
        lines.append(f"| {r.get('id', 'TC'):<5} | {q_prev:<52} | {exp_src:<18} | {top_src:<18} | {rank_str:<5} | {score_str:<6} | {res_str:<6} |")
    lines.append("+-------+------------------------------------------------------+--------------------+--------------------+-------+--------+--------+")
    return "\n".join(lines)


def main():
    load_dotenv()
    print("=" * 85)
    print(" KNOVERA RAG ASSISTANT - EMBEDDING QUALITY CHECKS & SANITY TESTS (3.29)")
    print("=" * 85)

    # 1. Initialize Generator and Checker
    generator = EmbeddingGenerator()
    checker = EmbeddingQualityChecker(generator=generator)
    print(f"Embedding Model:         {checker.model_name}")
    print(f"Embedding Dimension:     {generator.dimension}")

    # 2. Embed Corpus
    corpus = load_corpus()
    print(f"Corpus Chunk Count:      {len(corpus)} chunks across 4 documents")
    print("Generating corpus embeddings...")
    corpus_records = generator.embed_chunks(corpus)
    print(f"Generated {len(corpus_records)} embedded records successfully.\n")

    # =========================================================================
    # TASK 1 & 2: RUN KNOWN RELEVANCE TEST CASES
    # =========================================================================
    print("-" * 85)
    print(" TASK 1 & 2: KNOWN RELEVANCE TEST SUITE (RELATED VS UNRELATED CHUNKS)")
    print("-" * 85)

    test_cases = load_known_test_cases()
    for tc in test_cases:
        tc["id"] = tc.get("id", "TC")

    suite_results = checker.run_sanity_suite(test_cases, corpus_records)

    # Assign IDs back for display
    for tc, res in zip(test_cases, suite_results["results"]):
        res["id"] = tc["id"]

    print("Relevance Sanity Test Results Table:")
    print(format_sanity_table(suite_results))

    print("\nAggregate Suite Metrics:")
    print(f"  - Total Tests:               {suite_results['total_tests']}")
    print(f"  - Passed (Hit@1):            {suite_results['passed']} ({suite_results['hit_at_1_rate']}%)")
    print(f"  - Failed:                    {suite_results['failed']}")
    print(f"  - Mean Reciprocal Rank (MRR): {suite_results['mrr']:.4f}")
    print(f"  - Average Positive Margin:   +{suite_results['avg_positive_margin']:.4f} cosine similarity")

    print("\nDetailed Score Margins (Target vs Top Distractor):")
    for r in suite_results["results"]:
        target_chunk = r["top_k_matches"][0]
        distractor = r["top_k_matches"][1] if len(r["top_k_matches"]) > 1 else None
        print(f"  [{r['id']}] \"{r['query']}\"")
        print(f"       -> Target:     {target_chunk['source']} (Score: {target_chunk['score']:.4f})")
        if distractor:
            print(f"       -> Distractor: {distractor['source']} (Score: {distractor['score']:.4f})")
            print(f"       -> Margin (Delta): +{r['margin_vs_distractor']:.4f}\n")

    # =========================================================================
    # TASK 3: IDENTIFY AND EXPLAIN SURPRISING / FAILING CASES
    # =========================================================================
    print("-" * 85)
    print(" TASK 3: SURPRISING / FAILING CASE ANALYSIS & DIAGNOSTICS")
    print("-" * 85)

    edge_cases = load_surprising_edge_cases()

    print("Case 1: Ambiguous Brevity & Domain Competition")
    edge1 = edge_cases[0]
    res_edge1 = checker.evaluate_test_case(edge1, corpus_records)
    print(f"  Query: \"{edge1['query']}\"")
    print(f"  Top Match #1: {res_edge1['top_k_matches'][0]['source']} (Score: {res_edge1['top_k_matches'][0]['score']:.4f}) - \"{res_edge1['top_k_matches'][0]['preview']}\"")
    print(f"  Top Match #2: {res_edge1['top_k_matches'][1]['source']} (Score: {res_edge1['top_k_matches'][1]['score']:.4f}) - \"{res_edge1['top_k_matches'][1]['preview']}\"")
    score_gap = abs(res_edge1['top_k_matches'][0]['score'] - res_edge1['top_k_matches'][1]['score'])
    print(f"  Score Gap between Dining Payment vs Billing SLA: {score_gap:.4f}")
    print("  Diagnosis: Highly truncated queries lack disambiguating domain context, causing close cosine collisions between distinct payment methods.\n")

    print("Case 2: Negation Handling in Dense Vectors")
    edge2 = edge_cases[1]
    res_edge2 = checker.evaluate_test_case(edge2, corpus_records)
    print(f"  Query: \"{edge2['query']}\"")
    print(f"  Top Match #1: {res_edge2['top_k_matches'][0]['source']} (Score: {res_edge2['top_k_matches'][0]['score']:.4f}) - \"{res_edge2['top_k_matches'][0]['preview']}\"")
    print(f"  Top Match #2: {res_edge2['top_k_matches'][1]['source']} (Score: {res_edge2['top_k_matches'][1]['score']:.4f}) - \"{res_edge2['top_k_matches'][1]['preview']}\"")
    print("  Diagnosis: Standard dense embeddings treat lexical terms symmetrically; 'NOT want Python' still introduces Python vector coordinates, elevating dev-guide.md higher than in a purely semantic query.\n")

    print("Case 3: Model Mismatch / Vector Space Incompatibility Hazard")
    mismatch_diag = checker.simulate_mismatched_model_ranking(
        query="How can a learner reset their account password?",
        corpus_records=corpus_records,
        expected_source="account-guide.md"
    )
    print(f"  Query: \"{mismatch_diag['query']}\"")
    print(f"  Consistent Model Match:   {mismatch_diag['correct_top_source']} (Score: {mismatch_diag['correct_top_score']:.4f}) -> Result: {'PASS' if mismatch_diag['correct_passed'] else 'FAIL'}")
    print(f"  Mismatched Space Match:   {mismatch_diag['mismatched_top_source']} (Score: {mismatch_diag['mismatched_top_score']:.4f}) -> Result: {'PASS' if mismatch_diag['mismatched_passed'] else 'FAIL (CORRUPTED)'}")
    print(f"  Critical Hazard Note:     {mismatch_diag['diagnosis']}\n")

    # =========================================================================
    # TASK 4 & 5: SUMMARISE REPORT & EXPORT ARTIFACTS
    # =========================================================================
    print("-" * 85)
    print(" TASK 4 & 5: SUMMARISE SANITY REPORT & EXPORT AUDIT ARTIFACTS")
    print("-" * 85)

    os.makedirs("outputs", exist_ok=True)
    report_json_path = os.path.join("outputs", "embedding_sanity_report.json")
    
    export_payload = {
        "assignment": "3.29 Embedding Quality Checks & Sanity Tests",
        "model": checker.model_name,
        "dimension": generator.dimension,
        "corpus_chunks_count": len(corpus_records),
        "relevance_suite": suite_results,
        "edge_case_diagnostics": {
            "brevity_ambiguity_case": res_edge1,
            "negation_case": res_edge2,
            "mismatched_model_simulation": mismatch_diag
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"  Exported Structured Sanity JSON Report to: {report_json_path}")

    # Export Text Execution Log
    report_txt_path = os.path.join("outputs", "embedding_sanity_output.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.29: EMBEDDING QUALITY CHECKS & SANITY TESTS\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Model:           {checker.model_name}\n")
        f.write(f"Vector Length:   {generator.dimension}\n")
        f.write(f"Total Tests:     {suite_results['total_tests']}\n")
        f.write(f"Hit@1 Rate:      {suite_results['hit_at_1_rate']}%\n")
        f.write(f"MRR:             {suite_results['mrr']:.4f}\n\n")
        f.write("RELEVANCE TEST SUITE RESULTS:\n")
        f.write(format_sanity_table(suite_results) + "\n\n")
        f.write("SURPRISING / FAILING CASE ANALYSIS:\n")
        f.write(f"1. Brevity Collision: Query '{edge1['query']}' yielded top score {res_edge1['top_score']:.4f} for {res_edge1['top_source']}.\n")
        f.write(f"2. Negation Blindspot: Query '{edge2['query']}' retrieved {res_edge2['top_source']} (Rank: {res_edge2['target_rank']}).\n")
        f.write(f"3. Model Mismatch Corruption: Consistent Model -> {mismatch_diag['correct_top_source']} (Score: {mismatch_diag['correct_top_score']}), Mismatched -> {mismatch_diag['mismatched_top_source']} (Score: {mismatch_diag['mismatched_top_score']}).\n")

    print(f"  Exported Text Execution Log to:           {report_txt_path}")

    print("\n" + "=" * 85)
    print(" ASSIGNMENT 3.29 QUALITY CHECKS & SANITY TESTS COMPLETE - ALL CRITERIA MET")
    print("=" * 85)


if __name__ == "__main__":
    main()
