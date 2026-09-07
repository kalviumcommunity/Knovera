"""
retrieval_relevance_tuning_demo.py

Comprehensive Demonstration Script for Assignment 3.34: Retrieval Relevance Tuning.
Executes multi-setting retrieval experiments against a multi-domain corpus in ChromaDB,
calculates Hit Rate, Top-1 Hit Rate, and MRR metrics, automatically selects the optimal setting,
and exports audit logs to outputs/retrieval_relevance_tuning_results.json and
outputs/retrieval_relevance_tuning_output.txt.
"""

import os
import json
import logging
from typing import List

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.retrieval_tuner import (
    TestQuery,
    RetrievalSetting,
    RetrievalTuner
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_tuning_demo_corpus() -> List[dict]:
    """Builds a rich, multi-domain grounded document corpus for relevance tuning experiments."""
    return [
        {
            "id": "account-guide.md#chunk_0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and entering their registered email address.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Authentication", "doc_title": "Learner Account Administration"}
        },
        {
            "id": "account-guide.md#chunk_1",
            "text": "Email verification process: Learners receive a secure single-use verification link valid for 15 minutes to complete account recovery.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "doc_type": "guide", "category": "Authentication", "doc_title": "Learner Account Administration"}
        },
        {
            "id": "campus-guide.md#chunk_0",
            "text": "Cafeteria menu updates: The dining menu changes every Monday morning. Weekly specials and dietary options are posted on the campus portal.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Campus Life", "doc_title": "Campus Facility Guide"}
        },
        {
            "id": "campus-guide.md#chunk_1",
            "text": "Library study room reservations: Learners can reserve private study rooms up to 7 days in advance via the student mobile application.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 1, "doc_type": "guide", "category": "Campus Life", "doc_title": "Campus Facility Guide"}
        },
        {
            "id": "submission-rubric.md#chunk_0",
            "text": "Project submission evidence requirements: Submissions must include a verified GitHub Pull Request link and a 3-5 minute video explanation link.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 0, "doc_type": "rubric", "category": "Academics", "doc_title": "Project Evaluation Rubric"}
        },
        {
            "id": "submission-rubric.md#chunk_1",
            "text": "Grading rubric distribution: Automated unit tests account for 40%, code architecture 30%, and video walkthrough defense accounts for 30%.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 1, "doc_type": "rubric", "category": "Academics", "doc_title": "Project Evaluation Rubric"}
        },
        {
            "id": "service-policy.md#chunk_0",
            "text": "Refund processing SLA: Full refund requests for enterprise subscriptions are processed within 5 to 7 business days following ticket approval.",
            "metadata": {"source": "service-policy.md", "chunk_index": 0, "doc_type": "policy", "category": "Billing", "doc_title": "Service SLA & Refund Policy"}
        },
        {
            "id": "dev-guide.md#chunk_0",
            "text": "Developer environment variable configuration: Store sensitive API keys in .env files and load them using dotenv. Never commit secrets to git.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Development", "doc_title": "Developer Setup Manual"}
        },
        {
            "id": "sec-ops.md#chunk_0",
            "text": "Security incident response escalation: Report security anomalies immediately to sec-ops@knovera.internal. Escalations trigger immediate key rotation.",
            "metadata": {"source": "sec-ops.md", "chunk_index": 0, "doc_type": "policy", "category": "Security", "doc_title": "Security Operations SOP"}
        }
    ]


def run_tuning_demo():
    """Executes the full retrieval relevance tuning pipeline."""
    print("=" * 80)
    print("KNOVERA RAG ASSISTANT — RETRIEVAL RELEVANCE TUNING DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize Generator and Vector DB
    generator = EmbeddingGenerator()
    vector_db = VectorDatabase(in_memory=True)
    collection_name = "knovera_relevance_tuning_demo"

    # 2. Index Document Corpus
    print("\n[Step 1] Embedding and Indexing Multi-Domain Corpus Into ChromaDB...")
    raw_chunks = build_tuning_demo_corpus()
    embedded_records = generator.embed_chunks(raw_chunks)
    indexer = CorpusIndexer(vector_db=vector_db, default_collection=collection_name)
    indexing_results = indexer.index_corpus(embedded_records, reset_collection=True)
    print(f"-> Successfully indexed {indexing_results['inserted_this_run']} document chunks into ChromaDB.")

    # 3. Define Ground-Truth Test Queries
    print("\n[Step 2] Defining Ground-Truth Test Queries & Target Sources...")
    test_queries = [
        TestQuery(
            query="How can a learner reset their password?",
            expected_source="account-guide.md",
            category="Authentication",
            description="Account recovery instructions query"
        ),
        TestQuery(
            query="When does the cafeteria menu change?",
            expected_source="campus-guide.md",
            category="Campus Life",
            description="Dining menu scheduling query"
        ),
        TestQuery(
            query="What evidence is required for project submission?",
            expected_source="submission-rubric.md",
            category="Academics",
            description="Submission artifact requirements query"
        ),
        TestQuery(
            query="What is the timeline for subscription refund processing?",
            expected_source="service-policy.md",
            category="Billing",
            description="Refund processing SLA query"
        ),
        TestQuery(
            query="How should developers manage API key environment variables?",
            expected_source="dev-guide.md",
            category="Development",
            description="Dev environment security configuration query"
        )
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"  Query {idx}: '{q.query}' -> Expected: '{q.expected_source}' ({q.category})")

    # 4. Define Retrieval Settings to Compare
    print("\n[Step 3] Defining Candidate Retrieval Configurations...")
    settings = [
        RetrievalSetting(
            name="baseline_k3",
            k=3,
            metadata_filter=None,
            min_score=0.0,
            use_hybrid=False,
            description="Standard top-3 vector search without filters or thresholding"
        ),
        RetrievalSetting(
            name="filtered_k3",
            k=3,
            metadata_filter={"doc_type": "guide"},
            min_score=0.0,
            use_hybrid=False,
            description="Top-3 vector search filtered strictly to doc_type='guide'"
        ),
        RetrievalSetting(
            name="strict_k5",
            k=5,
            metadata_filter=None,
            min_score=0.72,
            use_hybrid=False,
            description="Top-5 vector search with strict similarity score threshold (min_score >= 0.72)"
        ),
        RetrievalSetting(
            name="hybrid_k3",
            k=3,
            metadata_filter=None,
            min_score=0.0,
            use_hybrid=True,
            vector_weight=0.7,
            keyword_weight=0.3,
            description="Hybrid vector + lexical keyword search (70% vector / 30% lexical)"
        ),
        RetrievalSetting(
            name="hybrid_filtered_strict_k3",
            k=3,
            metadata_filter={"doc_type": "guide"},
            min_score=0.50,
            use_hybrid=True,
            vector_weight=0.7,
            keyword_weight=0.3,
            description="Filtered hybrid search with moderate score threshold (min_score >= 0.50)"
        )
    ]

    # 5. Initialize Tuner and Run Evaluation
    print("\n[Step 4] Running Comparative Retrieval Evaluation Across All Settings...")
    tuner = RetrievalTuner(
        vector_db=vector_db,
        embedding_generator=generator,
        default_collection=collection_name
    )

    summaries = tuner.evaluate_all_settings(
        settings=settings,
        test_queries=test_queries,
        collection_name=collection_name
    )

    # 6. Display Summary Comparison Table
    print("\n" + "=" * 80)
    print("EMPIRICAL RETRIEVAL RELEVANCE COMPARISON TABLE")
    print("=" * 80)
    table_markdown = tuner.format_comparison_table(summaries)
    print(table_markdown)

    # 7. Select Best Setting with Evidence
    winning_summary, justification = tuner.select_best_setting(summaries)

    print("\n" + "=" * 80)
    print("QUANTITATIVE OPTIMAL SETTING SELECTION")
    print("=" * 80)
    print(f"CHOSEN SETTING:     {winning_summary.setting_name}")
    print(f"HIT RATE:           {winning_summary.hit_rate * 100:.1f}% ({winning_summary.hits}/{winning_summary.total_queries})")
    print(f"TOP-1 HIT RATE:     {winning_summary.top_1_hit_rate * 100:.1f}%")
    print(f"MEAN RECIPROCAL RANK: {winning_summary.mrr:.4f}")
    print(f"AVG TOP SCORE:      {winning_summary.avg_top_score:.4f}")
    print(f"AVG RETAINED CHUNKS:{winning_summary.avg_retained_chunks:.1f}")
    print(f"\nJUSTIFICATION:\n{justification}")

    # 8. Export Audit Logs
    json_output_path = os.path.join("outputs", "retrieval_relevance_tuning_results.json")
    txt_output_path = os.path.join("outputs", "retrieval_relevance_tuning_output.txt")
    
    tuner.export_results(
        summaries=summaries,
        winning_summary=winning_summary,
        justification=justification,
        json_path=json_output_path,
        txt_path=txt_output_path
    )

    print("\n" + "=" * 80)
    print("AUDIT ARTIFACTS EXPORTED SUCCESSFULLY")
    print(f"-> JSON Audit Log: {json_output_path}")
    print(f"-> Text Audit Log: {txt_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_tuning_demo()
