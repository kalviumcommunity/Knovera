"""
corpus_indexing_demo.py

Demonstration and Verification Suite for Assignment 3.31: Indexing Embeddings & Metadata Storage.
Platform: Knovera RAG Assistant

Features Demonstrated:
- Task 1: Bulk insert all corpus embeddings into a ChromaDB vector database collection.
- Task 2: Store embedding vectors with source text and rich hierarchical metadata.
- Task 3: Confirm indexed record count matches the input chunk count.
- Task 4: Spot-check stored record integrity (ID, vector length, text, metadata parity).
- Task 5: Export indexing summary JSON and human-readable audit logs to outputs directory.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Reconfigure stdout to UTF-8 for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator


def load_corpus() -> List[Dict[str, Any]]:
    """
    Constructs a rich multi-document corpus spanning Authentication, Billing,
    Campus Life, SLAs, and Developer API SDKs.
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
            "text": "Knovera Python SDK Quickstart: Install knovera-sdk via pip and configure KNOVERA_API_KEY environment variable before invoking the Client.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 0, "section": "SDK Installation", "doc_title": "Developer SDK Guide", "category": "Developer"}
        },
        {
            "id": "dev-guide.md:1",
            "text": "Semantic chunking API endpoints allow developers to pass raw markdown and receive token-bounded chunks tagged with positional metadata.",
            "metadata": {"source": "dev-guide.md", "chunk_index": 1, "section": "Chunking Endpoints", "doc_title": "Developer SDK Guide", "category": "Developer"}
        }
    ]


def main():
    load_dotenv()
    print("=" * 85)
    print(" KNOVERA RAG ASSISTANT - INDEXING EMBEDDINGS & METADATA STORAGE (3.31)")
    print("=" * 85)

    # 1. Initialize Vector Database & Indexer
    persist_dir = os.path.join(".", "data", "chroma_db")
    collection_name = "knovera_knowledge_base"
    
    vector_db = VectorDatabase(persist_dir=persist_dir)
    indexer = CorpusIndexer(vector_db=vector_db, default_collection=collection_name)
    generator = EmbeddingGenerator()

    print(f"Vector Store Engine:     ChromaDB (v0.5.23)")
    print(f"Storage Persistence:     {os.path.abspath(persist_dir)}")
    print(f"Target Collection:       '{collection_name}'")
    print(f"Embedding Model:         {generator.model_name}")

    # =========================================================================
    # PREPARATION: EMBED RAW CORPUS CHUNKS
    # =========================================================================
    raw_chunks = load_corpus()
    print(f"\nEmbedding {len(raw_chunks)} corpus chunks via OpenRouter API...")
    embedded_chunks = generator.embed_chunks(raw_chunks, batch_size=4)
    print(f"Produced {len(embedded_chunks)} embedded chunks with vector dimension {generator.dimension}.")

    # =========================================================================
    # TASK 1 & 2: BULK INDEX ALL CORPUS EMBEDDINGS WITH METADATA
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 1 & 2: BULK INDEX ALL CORPUS EMBEDDINGS WITH TEXT AND METADATA")
    print("-" * 85)
    print(f"Indexing {len(embedded_chunks)} records in batches of 4...")

    index_summary = indexer.index_corpus(
        embedded_chunks=embedded_chunks,
        collection_name=collection_name,
        batch_size=4,
        reset_collection=True
    )

    print("\nIndexing Execution Summary:")
    print(f"  - Target Collection:         {index_summary['collection_name']}")
    print(f"  - Expected Input Chunks:     {index_summary['expected_chunks']}")
    print(f"  - Inserted Records (Run):    {index_summary['inserted_this_run']}")
    print(f"  - Total Batches:             {index_summary['total_batches']}")
    print(f"  - Elapsed Duration:          {index_summary['duration_seconds']}s")
    print(f"  - Ingestion Throughput:      {index_summary['throughput_records_per_sec']} records/sec")
    print(f"  - Failures Encountered:      {len(index_summary['failures'])}")

    # =========================================================================
    # TASK 3: CONFIRM INDEXED RECORD COUNT
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 3: CONFIRM INDEXED RECORD COUNT AGAINST CORPUS")
    print("-" * 85)

    expected_count = len(embedded_chunks)
    final_indexed_count = index_summary["final_indexed_count"]
    count_matches = index_summary["count_matches"]

    print(f"Expected Chunks Produced:     {expected_count}")
    print(f"Final Count in Vector DB:     {final_indexed_count}")
    print(f"Count Verification:           {'MATCH CONFIRMED (100% Data Integrity)' if count_matches else 'COUNT MISMATCH DETECTED'}")

    assert count_matches, f"Error: Indexed count ({final_indexed_count}) != expected ({expected_count})"

    # =========================================================================
    # TASK 4: SPOT-CHECK STORED INTEGRITY
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 4: SPOT-CHECK STORED RECORD INTEGRITY (FIRST, MIDDLE, LAST)")
    print("-" * 85)

    # Select representative sample chunks: first (0), middle (4), last (9)
    sample_indices = [0, 4, len(embedded_chunks) - 1]
    samples_to_check = [embedded_chunks[i] for i in sample_indices]

    spot_check_results = indexer.spot_check_integrity(
        sample_chunks=samples_to_check,
        collection_name=collection_name
    )

    for idx, sc in enumerate(spot_check_results, start=1):
        print(f"Spot Check #{idx} - ID: '{sc['id']}'")
        print(f"  - Found in Database:    {sc['found']}")
        print(f"  - Vector Length:        {sc['vector_length']} (Expected: {sc['expected_vector_length']})")
        print(f"  - Source Document:      {sc['source']} (Chunk Index: {sc['chunk_index']})")
        print(f"  - Section Heading:      {sc['section']}")
        print(f"  - Text Preview:         \"{sc['text_preview']}\"")
        print(f"  - All Checks Passed:    {'YES (PASSED)' if sc['passed'] else 'NO (FAILED)'}\n")
        assert sc["passed"], f"Spot check failed for chunk {sc['id']}"

    # Demonstrate Metadata Filtering Retrieval
    print("-" * 85)
    print(" ADVANCED VALIDATION: METADATA FILTERED SEARCH (WHY METADATA STORAGE MATTERS)")
    print("-" * 85)
    query_str = "How do I recover access or reset credentials?"
    query_vector = generator.embed([query_str])[0]

    # Unfiltered Search
    unfiltered_res = vector_db.query_similar(query_vector, top_k=2, collection_name=collection_name)
    print(f"Query: \"{query_str}\"")
    print("Unfiltered Top-2 Results:")
    for r in unfiltered_res:
        print(f"  - Rank #{r['rank']} [Sim: {r['similarity']:.4f}] - ID: {r['id']} ({r['metadata'].get('source')})")

    # Filtered Search by Category='Authentication'
    filtered_res = vector_db.query_similar(
        query_vector,
        top_k=2,
        where={"category": "Authentication"},
        collection_name=collection_name
    )
    print("\nFiltered by category='Authentication' Top-2 Results:")
    for r in filtered_res:
        print(f"  - Rank #{r['rank']} [Sim: {r['similarity']:.4f}] - ID: {r['id']} (Category: {r['metadata'].get('category')})")

    # Demonstrate Incremental Document Re-indexing
    print("\n" + "-" * 85)
    print(" FOLLOW-UP VALIDATION: INCREMENTAL DOCUMENT RE-INDEXING (WHEN DOCS CHANGE)")
    print("-" * 85)
    updated_campus_chunk = {
        "id": "campus-guide.md:0",
        "text": "Campus cafeteria hours updated: Cafeteria now operates from 10:30 AM to 3:00 PM offering artisan pizza, vegan bowls, and gourmet pasta.",
        "embedding": generator.embed(["Campus cafeteria hours updated: Cafeteria now operates from 10:30 AM to 3:00 PM offering artisan pizza, vegan bowls, and gourmet pasta."])[0],
        "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Updated Dining Hours", "doc_title": "Campus Guide", "category": "Campus Life"}
    }
    reindex_result = indexer.reindex_document(
        doc_source="campus-guide.md",
        new_chunks=[updated_campus_chunk],
        collection_name=collection_name
    )
    print(f"Incremental Re-indexing of 'campus-guide.md':")
    print(f"  - Evicted Stale Chunks:      {reindex_result['evicted_stale_chunks']}")
    print(f"  - Inserted New Chunks:       {reindex_result['inserted_new_chunks']}")
    print(f"  - Post-Update Total Count:   {reindex_result['collection_total_count']}")
    print(f"  - Status:                    {reindex_result['status']}")

    # =========================================================================
    # TASK 5: EXPORT INDEXING SUMMARY & AUDIT LOGS
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 5: EXPORT INDEXING SUMMARY ARTIFACTS")
    print("-" * 85)

    os.makedirs("outputs", exist_ok=True)
    
    # 1. Export JSON Summary
    summary_json_path = os.path.join("outputs", "indexing_run_summary.json")
    export_payload = {
        "assignment": "3.31 Indexing Embeddings & Metadata Storage",
        "collection_name": collection_name,
        "database": "ChromaDB (v0.5.23)",
        "model": generator.model_name,
        "vector_dimension": generator.dimension,
        "metrics": index_summary,
        "spot_checks": spot_check_results,
        "incremental_reindex_demonstration": reindex_result
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"  Exported Indexing Summary JSON to: {summary_json_path}")

    # 2. Export Text Audit Log
    summary_txt_path = os.path.join("outputs", "indexing_output.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.31: INDEXING EMBEDDINGS & METADATA STORAGE\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Collection Name:     {collection_name}\n")
        f.write(f"Embedding Model:     {generator.model_name}\n")
        f.write(f"Vector Dimension:    {generator.dimension}\n")
        f.write(f"Expected Chunks:     {index_summary['expected_chunks']}\n")
        f.write(f"Inserted Records:    {index_summary['inserted_this_run']}\n")
        f.write(f"Final Count in DB:   {index_summary['final_indexed_count']}\n")
        f.write(f"Count Matches:       {index_summary['count_matches']}\n")
        f.write(f"Duration:            {index_summary['duration_seconds']}s\n")
        f.write(f"Throughput:          {index_summary['throughput_records_per_sec']} records/sec\n\n")
        f.write("SPOT CHECK INTEGRITY AUDIT:\n")
        for sc in spot_check_results:
            f.write(f"  ID: {sc['id']} | Dim: {sc['vector_length']} | Passed: {sc['passed']}\n")
            f.write(f"    Source: {sc['source']} | Section: {sc['section']}\n")
            f.write(f"    Text: {sc['text_preview']}\n\n")
    print(f"  Exported Text Audit Log to:       {summary_txt_path}")

    print("\n" + "=" * 85)
    print(" ASSIGNMENT 3.31 CORPUS INDEXING & VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    main()
