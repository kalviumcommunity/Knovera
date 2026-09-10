"""
retrieval_evaluation_demo.py

Enterprise Demonstration & Experimental Script for Assignment 3.36:
Retrieval Evaluation & Recall Testing.

Demonstrates:
1. Ground-truth benchmark corpus indexing with ChromaDB vector store.
2. Labelled query loading and mapping to expected chunk IDs.
3. Recall@k and Precision@k evaluation across single and batch queries.
4. Multi-K trade-off analysis (k=1, 3, 5, 10).
5. Automated failure inspection and root cause diagnosis.
6. Report generation and audit log export to JSON and TXT artifacts.
"""

import os
import json
import logging
from typing import List, Dict, Any

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.corpus_indexer import CorpusIndexer
from src.retrieval_evaluator import (
    LabelledQuery,
    QueryEvaluationResult,
    AggregateMetrics,
    RetrievalEvaluator
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_demo_benchmark_corpus() -> List[Dict[str, Any]]:
    """Creates a rich, domain-specific benchmark chunk corpus for retrieval evaluation."""
    return [
        {
            "id": "account-guide.md:0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and submitting their registered email address.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "category": "Authentication", "doc_type": "guide"}
        },
        {
            "id": "account-guide.md:1",
            "text": "Password recovery email dispatch: Learners receive a single-use secure link valid for 15 minutes to configure a new password. Two-factor authentication (2FA) codes can also be requested.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "category": "Authentication", "doc_type": "guide"}
        },
        {
            "id": "submission-rubric.md:0",
            "text": "Project submission evidence requirement: Learners must upload repository GitHub PR links and a 3-to-5 minute video screen recording demonstrating working functionality.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 0, "category": "Academics", "doc_type": "rubric"}
        },
        {
            "id": "campus-guide.md:0",
            "text": "Campus cafeteria operating hours and menu rotation: Lunch service runs from 11:30 AM to 2:30 PM offering fresh salads, soups, and rotating weekly hot entrees.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "category": "Campus Life", "doc_type": "guide"}
        },
        {
            "id": "api-policy.md:0",
            "text": "API rate limits and request throttling: Standard developer keys are restricted to 60 requests per minute and 10,000 embedding tokens per batch query.",
            "metadata": {"source": "api-policy.md", "chunk_index": 0, "category": "Operations", "doc_type": "policy"}
        },
        {
            "id": "api-policy.md:1",
            "text": "Vector indexing token quota guidelines: Enterprise tier applications receive up to 1,000,000 monthly vector embedding tokens with dedicated concurrency channels.",
            "metadata": {"source": "api-policy.md", "chunk_index": 1, "category": "Operations", "doc_type": "policy"}
        },
        {
            "id": "billing-faq.md:0",
            "text": "Enterprise billing tier refund policy: Cancelation within 14 calendar days of subscription activation guarantees a 100% full refund with zero cancellation penalty.",
            "metadata": {"source": "billing-faq.md", "chunk_index": 0, "category": "Billing", "doc_type": "faq"}
        },
        {
            "id": "vector-db-guide.md:0",
            "text": "HNSW vector indexing and distance metrics: ChromaDB supports Cosine Similarity, L2 Euclidean Distance, and Inner Product dot metrics for high-dimensional vector search.",
            "metadata": {"source": "vector-db-guide.md", "chunk_index": 0, "category": "Infrastructure", "doc_type": "guide"}
        },
        # Distractor chunks to simulate realistic corpus density
        {
            "id": "distractor-guide.md:0",
            "text": "General software engineering guidelines: Clean code principles, modular function definitions, and unit testing protocols for Python applications.",
            "metadata": {"source": "distractor-guide.md", "chunk_index": 0, "category": "General", "doc_type": "guide"}
        },
        {
            "id": "distractor-policy.md:0",
            "text": "Campus library quiet hours policy: Study rooms must be reserved online 24 hours in advance.",
            "metadata": {"source": "distractor-policy.md", "chunk_index": 0, "category": "Campus Life", "doc_type": "policy"}
        }
    ]


def main():
    print("================================================================================")
    print(" KNOVERA RAG ASSISTANT — RETRIEVAL EVALUATION & RECALL TESTING DEMO")
    print("================================================================================\n")

    # Step 1: Initialize Embedding Generator and Vector Store
    logger.info("Initializing Embedding Generator and ChromaDB Vector Store...")
    generator = EmbeddingGenerator()
    vdb = VectorDatabase(in_memory=True)
    indexer = CorpusIndexer(vector_db=vdb, default_collection="knovera_eval_demo_chunks")

    # Embed and Index Benchmark Corpus
    logger.info("Embedding benchmark document chunks...")
    corpus = create_demo_benchmark_corpus()
    embedded_records = generator.embed_chunks(corpus)
    indexer.index_corpus(embedded_records, reset_collection=True)

    count = vdb.count("knovera_eval_demo_chunks")
    print(f"[OK] Successfully indexed {count} document chunks into collection 'knovera_eval_demo_chunks'.\n")

    # Step 2: Load Labelled Queries
    queries_file = "sample_labelled_queries.json"
    print(f"[*] Loading labelled queries from '{queries_file}'...")
    with open(queries_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    labelled_queries = dataset["labelled_queries"]
    print(f"Loaded {len(labelled_queries)} labelled test queries for benchmark testing.\n")

    # Step 3: Initialize Retrieval Evaluator
    evaluator = RetrievalEvaluator(
        vector_db=vdb,
        embedding_generator=generator,
        default_collection="knovera_eval_demo_chunks"
    )

    # Step 4: Evaluate Single Query Demonstration
    print("--------------------------------------------------------------------------------")
    print(" TASK 1 & 2: SINGLE QUERY RECALL & PRECISION EVALUATION DEMONSTRATION")
    print("--------------------------------------------------------------------------------")
    single_q = labelled_queries[0]
    single_res = evaluator.evaluate_query(query_item=single_q, k=5)

    print(f"Query:               '{single_res.query}'")
    print(f"Category:            {single_res.category}")
    print(f"Expected Chunks:     {single_res.relevant_chunk_ids}")
    print(f"Retrieved Chunks:    {single_res.retrieved_ids}")
    print(f"Retrieved Hits:      {single_res.hits}")
    print(f"Recall@5:            {single_res.recall * 100:.1f}%")
    print(f"Precision@5:         {single_res.precision * 100:.1f}%")
    print(f"F1-Score:            {single_res.f1_score:.4f}")
    print(f"Reciprocal Rank:     {single_res.reciprocal_rank:.4f}\n")

    # Step 5: Full Benchmark Dataset Evaluation (top-k=5)
    print("--------------------------------------------------------------------------------")
    print(" TASK 3: BATCH BENCHMARK RETRIEVAL EVALUATION (top-k=5)")
    print("--------------------------------------------------------------------------------")
    metrics_k5 = evaluator.evaluate_queries(labelled_queries=labelled_queries, k=5)

    summary_report = evaluator.generate_summary_report(metrics_k5)
    print(summary_report)
    print()

    # Step 6: Multi-K Metric Comparison (k=1, 3, 5, 10)
    print("--------------------------------------------------------------------------------")
    print(" TASK 2 & 3: MULTI-K RECALL vs PRECISION TRADE-OFF ANALYSIS")
    print("--------------------------------------------------------------------------------")
    k_comparison = evaluator.evaluate_across_k(labelled_queries=labelled_queries, k_values=[1, 3, 5, 10])

    print("| Top-K (k) | Avg Recall@k | Avg Precision@k | Avg F1-Score | MRR | Hit Rate | Passed Queries |")
    print("|---|---|---|---|---|---|---|")
    for k_val, m in k_comparison.items():
        print(
            f"| **k={k_val}** | **`{m.avg_recall * 100:.1f}%`** | `{m.avg_precision * 100:.1f}%` | "
            f"`{m.avg_f1:.4f}` | `{m.mrr:.4f}` | `{m.hit_rate * 100:.1f}%` | `{m.passed_queries_count}/{m.total_queries}` |"
        )
    print()

    # Step 7: Failure Inspection & Root Cause Diagnosis
    print("--------------------------------------------------------------------------------")
    print(" TASK 4: FAILURE INSPECTION & ROOT CAUSE DIAGNOSIS")
    print("--------------------------------------------------------------------------------")
    # Evaluate at k=1 to demonstrate failure inspection when k is small relative to relevant set
    metrics_k1 = evaluator.evaluate_queries(labelled_queries=labelled_queries, k=1)
    failures_k1 = metrics_k1.failure_analysis

    print(f"Evaluated top-k=1 cutoff. Detected {len(failures_k1)} queries with Recall < 1.0:\n")
    for idx, f in enumerate(failures_k1, start=1):
        print(f"Failure #{idx}:")
        print(f"  Query:            '{f['query']}'")
        print(f"  Recall@1:         {f['recall'] * 100:.1f}%")
        print(f"  Precision@1:      {f['precision'] * 100:.1f}%")
        print(f"  Expected Chunks:  {f['expected_chunk_ids']}")
        print(f"  Retrieved Chunks: {f['retrieved_ids']}")
        print(f"  Hits:             {f['hits']}")
        print(f"  Missing Chunks:   {f['missing_chunk_ids']}")
        print(f"  Diagnosed Cause:  {f['diagnosed_cause']}")
        print(f"  Recommendation:   {f['actionable_recommendation']}\n")

    # Step 8: Export Results & Audit Logs
    print("--------------------------------------------------------------------------------")
    print(" TASK 5: EXPORT RESULTS & AUDIT ARTIFACTS")
    print("--------------------------------------------------------------------------------")
    json_path = os.path.join("outputs", "retrieval_evaluation_results.json")
    txt_path = os.path.join("outputs", "retrieval_evaluation_audit.log")

    evaluator.export_results(
        aggregate_metrics=metrics_k5,
        json_path=json_path,
        txt_path=txt_path
    )

    print(f"[OK] Exported structured JSON results: '{json_path}'")
    print(f"[OK] Exported plain text audit log:    '{txt_path}'")

    print("\n================================================================================")
    print(" DEMONSTRATION COMPLETE — RETRIEVAL EVALUATION & RECALL TESTING SUCCESSFUL")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
