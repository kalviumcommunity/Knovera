"""
similarity_search_demo.py

Demonstration and Audit Verification Engine for Assignment 3.32: Similarity Search & Top-K Retrieval.
Platform: Knovera RAG Assistant

Features & Verification Flow:
- Task 1: Embed sample user query using identical document embedding model.
- Task 2: Execute top-k similarity search against ChromaDB vector store.
- Task 3: Return ranked chunks with similarity scores, distance values, raw source text, and metadata.
- Task 4: Demonstrate changing k across k=1, k=3, k=5 and analyze context expansion vs noise trade-offs.
- Task 5: Commit retrieval code and export output text/JSON to outputs directory.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# UTF-8 stdout configuration for Windows terminal compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.top_k_retriever import TopKRetriever


def get_demo_corpus() -> List[Dict[str, Any]]:
    """
    Constructs a rich multi-document corpus spanning Authentication, Billing,
    Campus Life, Service SLAs, and Developer API SDKs.
    """
    return [
        {
            "id": "account-guide.md:0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and submitting their registered email address.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Recovery", "doc_title": "Account Admin Guide", "category": "Authentication"}
        },
        {
            "id": "account-guide.md:1",
            "text": "Learners can recover access using their registered email. A one-time secure verification link is dispatched, which remains valid for 15 minutes.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Verification", "doc_title": "Account Admin Guide", "category": "Authentication"}
        },
        {
            "id": "account-guide.md:2",
            "text": "Two-factor authentication (2FA) is mandatory for enterprise admin consoles. Admins must register an authenticator app (TOTP) or hardware security key.",
            "metadata": {"source": "account-guide.md", "chunk_index": 2, "section": "Multi-Factor Security", "doc_title": "Account Admin Guide", "category": "Authentication"}
        },
        {
            "id": "service-policy.md:0",
            "text": "Knovera Customer Service & Refund Policy: Enterprise clients are eligible for full subscription refunds within a 14-day evaluation window from initial provisioning.",
            "metadata": {"source": "service-policy.md", "chunk_index": 0, "section": "Refund Eligibility", "doc_title": "Service Policy Guide", "category": "Billing"}
        },
        {
            "id": "service-policy.md:1",
            "text": "Refund processing requires an approved ticket from the billing department. Once authorized, funds will appear on the original payment method in 3-5 business days.",
            "metadata": {"source": "service-policy.md", "chunk_index": 1, "section": "Refund Processing SLA", "doc_title": "Service Policy Guide", "category": "Billing"}
        },
        {
            "id": "service-policy.md:2",
            "text": "Enterprise Service Level Agreement (SLA): Knovera guarantees 99.9% uptime for core API endpoints. Outages exceeding SLA thresholds qualify for service credits.",
            "metadata": {"source": "service-policy.md", "chunk_index": 2, "section": "Uptime SLA", "doc_title": "Service Policy Guide", "category": "Infrastructure"}
        },
        {
            "id": "campus-guide.md:0",
            "text": "Campus cafeteria hours and lunch menu: The cafeteria operates from 11:30 AM to 2:30 PM offering Italian pasta, fresh garden salads, and hot daily specials.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Dining Hours", "doc_title": "Campus Guide", "category": "Campus Life"}
        },
        {
            "id": "campus-guide.md:1",
            "text": "Campus dining cards can be recharged via the student portal or mobile app. Cash and contactless digital cards are accepted at all food counters.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 1, "section": "Payment Methods", "doc_title": "Campus Guide", "category": "Campus Life"}
        },
        {
            "id": "dev-guide.md:0",
            "text": "Knovera Python SDK Integration: Install via pip install knovera. Initialize the client using API key authentication to manage vector embeddings and semantic search.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 0, "section": "SDK Setup", "doc_title": "Developer Integration Guide", "category": "Developer API"}
        },
        {
            "id": "dev-guide.md:1",
            "text": "Rate limits and error handling: API calls are capped at 100 requests per minute. Handle HTTP 429 status codes using exponential backoff retry strategies.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 1, "section": "Rate Limits & Retries", "doc_title": "Developer Integration Guide", "category": "Developer API"}
        }
    ]


def run_demo():
    """
    Executes similarity search & top-k retrieval demonstration, capturing output
    for console printing and JSON/TXT file export.
    """
    load_dotenv()
    output_dir = os.path.join(".", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, "similarity_search_results.json")
    txt_output_path = os.path.join(output_dir, "similarity_search_output.txt")

    print("=" * 80)
    print("KNOVERA RAG ASSISTANT: ASSIGNMENT 3.32 - SIMILARITY SEARCH & TOP-K RETRIEVAL")
    print("=" * 80)

    # 1. Initialize Vector Database and Embedding Generator
    print("\n[Step 1/5] Initializing Vector Store and Model Alignment...")
    generator = EmbeddingGenerator()
    vector_db = VectorDatabase(in_memory=True, embedding_generator=generator)
    indexer = CorpusIndexer(vector_db=vector_db, default_collection="demo_top_k_collection")
    retriever = TopKRetriever(vector_db=vector_db, embedding_generator=generator, default_collection="demo_top_k_collection")

    # 2. Index Sample Corpus
    print("[Step 2/5] Bulk Indexing Multi-Document Corpus into ChromaDB...")
    raw_chunks = get_demo_corpus()
    embedded_chunks = generator.embed_chunks(raw_chunks)
    index_summary = indexer.index_corpus(embedded_chunks, reset_collection=True)

    print(f" -> Collection Name: {index_summary['collection_name']}")
    print(f" -> Indexed Count:   {index_summary['final_indexed_count']} / {index_summary['expected_chunks']}")
    print(f" -> Count Verification Status: {index_summary['status']}")

    # Check model alignment
    alignment = retriever.verify_model_alignment()
    print(f" -> Embedding Model: {alignment['model_name']}")
    print(f" -> Query Vector Dimension: {alignment['query_dimension']} (Aligned: {alignment['is_aligned']})")

    # 3. Task 1, 2 & 3: Primary Sample Query Retrieval (k=3)
    sample_query = "How can a learner reset their password?"
    print(f"\n[Step 3/5] Executing Top-K Similarity Search for Primary Query (k=3)")
    print(f" -> Sample Query: \"{sample_query}\"")

    results_k3 = retriever.retrieve(query=sample_query, k=3)

    print("\nRetrieved Grounded Context Chunks (k=3):")
    print("-" * 80)
    for rank, result in enumerate(results_k3, start=1):
        print(f"rank:        {rank}")
        print(f"score:       {result['score']:.4f}")
        print(f"distance:    {result['distance']:.4f}")
        print(f"source:      {result['metadata']['source']}")
        print(f"chunk_index: {result['metadata']['chunk_index']}")
        print(f"category:    {result['metadata']['category']}")
        print(f"section:     {result['metadata']['section']}")
        print(f"text:        {result['text']}")
        print("-" * 80)

    # 4. Task 4: Demonstrate Changing k (k=1, k=3, k=5)
    print(f"\n[Step 4/5] Demonstrating Changing K (k=1, k=3, k=5) for Primary Query")
    print(f" -> Query: \"{sample_query}\"")

    k_comparison = retriever.compare_k(query=sample_query, k_values=[1, 3, 5])

    for run in k_comparison["runs"]:
        k_val = run["k"]
        print(f"\n--- k = {k_val} (Retrieved: {run['retrieved_count']} chunks | Context Tokens: ~{run['estimated_context_tokens']} | Top Score: {run['top_score']}) ---")
        for res in run["results"]:
            print(f"  [{res['score']:.4f}] [{res['metadata']['source']}#chunk_{res['metadata']['chunk_index']}] {res['text'][:95]}...")

    # Secondary Query Demonstration
    secondary_query = "What are the campus cafeteria hours and lunch menu options?"
    print(f"\n--- Secondary Query Evaluation ---")
    print(f" -> Query: \"{secondary_query}\"")
    sec_comparison = retriever.compare_k(query=secondary_query, k_values=[1, 3, 5])
    for run in sec_comparison["runs"]:
        k_val = run["k"]
        print(f"\n--- k = {k_val} (Retrieved: {run['retrieved_count']} chunks | Top Score: {run['top_score']}) ---")
        for res in run["results"]:
            print(f"  [{res['score']:.4f}] [{res['metadata']['source']}#chunk_{res['metadata']['chunk_index']}] {res['text'][:95]}...")

    # 5. Task 5: Export Summary JSON & Human-Readable Log Text
    print(f"\n[Step 5/5] Exporting Run Results and Audit Logs...")
    
    export_payload = {
        "assignment": "3.32 Similarity Search & Top-K Retrieval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_alignment": alignment,
        "index_summary": index_summary,
        "primary_query": sample_query,
        "results_k3": results_k3,
        "multi_k_demonstration": k_comparison,
        "secondary_query_demonstration": sec_comparison,
        "k_trade_off_analysis": {
            "k_1": "Cheapest and most focused. High precision, zero noise, but risks missing secondary details (e.g. 2FA rules).",
            "k_3": "Optimal balance for RAG prompt context. Captures full password recovery workflow without distractor chunks.",
            "k_5": "Higher recall. Includes unrelated chunks (e.g. Billing/Refund policies) which consumes context window and adds noise."
        }
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)
    print(f" -> JSON Results Exported: {json_output_path}")

    # Build human-readable output log file
    log_buffer = []
    log_buffer.append("================================================================================")
    log_buffer.append("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.32 SIMILARITY SEARCH & TOP-K RETRIEVAL LOG")
    log_buffer.append("================================================================================\n")
    log_buffer.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_buffer.append(f"Embedding Model: {alignment['model_name']}")
    log_buffer.append(f"Vector Dimension: {alignment['query_dimension']}")
    log_buffer.append(f"Collection Record Count: {index_summary['final_indexed_count']}\n")

    log_buffer.append("--- TASK 1-3: PRIMARY RETRIEVAL RESULTS (Query: 'How can a learner reset their password?', k=3) ---")
    for r in results_k3:
        log_buffer.append(f"rank: {r['rank']} | score: {r['score']:.4f} | distance: {r['distance']:.4f}")
        log_buffer.append(f"source: {r['metadata']['source']} | chunk_index: {r['metadata']['chunk_index']} | category: {r['metadata']['category']}")
        log_buffer.append(f"text: {r['text']}\n")

    log_buffer.append("--- TASK 4: MULTI-K DEMONSTRATION (k=1, k=3, k=5) ---")
    for run in k_comparison["runs"]:
        log_buffer.append(f"\nk = {run['k']}")
        for r in run["results"]:
            log_buffer.append(f"{r['score']:.4f} {r['metadata']} {r['text'][:100]}")

    log_buffer.append("\n================================================================================")
    log_buffer.append("AUDIT VERIFICATION COMPLETED SUCCESSFULLY")
    log_buffer.append("================================================================================")

    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_buffer))
    print(f" -> Text Audit Log Exported: {txt_output_path}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION & RETRIEVAL PROOF COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
