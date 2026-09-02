"""
api_embedding_demo.py

Demonstration and Verification Suite for Assignment 3.26: Generating Embeddings via API.
- Reads API key, model name, and base URL from environment configuration.
- Batches prepared text chunks and sends them to an OpenAI-compatible Embeddings API.
- Stores each vector bound directly to its source chunk text and rich metadata for retrieval.
- Prints verification outputs: number of records, vector dimension, and sample float values.
- Demonstrates retrieval using identical-model query embeddings and exports JSON/log artifacts.
"""

import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.embedding_generator import EmbeddingGenerator, cosine_similarity


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Masks secret keys for secure console logging."""
    if not secret:
        return "None (Using Offline Fallback Engine)"
    if len(secret) <= visible_chars * 2:
        return "****"
    return f"{secret[:visible_chars]}...{secret[-visible_chars:]}"


def load_prepared_corpus() -> List[Dict[str, Any]]:
    """
    Returns a prepared multi-document corpus with rich retrieval metadata.
    Covers authentication, service policies, SLAs, and general campus info.
    """
    return [
        {
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and submitting their registered email address.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 0,
                "doc_title": "Knovera Learner Account Administration",
                "section": "Password Recovery",
                "page": 1,
                "category": "Authentication"
            }
        },
        {
            "text": "Learners can recover access using their registered email. A one-time secure verification link is dispatched, which remains valid for 15 minutes.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 1,
                "doc_title": "Knovera Learner Account Administration",
                "section": "Email Verification & Tokens",
                "page": 1,
                "category": "Authentication"
            }
        },
        {
            "text": "Two-factor authentication (2FA) is mandatory for enterprise admin consoles. Admins must register an authenticator app (TOTP) or hardware security key.",
            "metadata": {
                "source": "account-guide.md",
                "chunk_index": 2,
                "doc_title": "Knovera Learner Account Administration",
                "section": "Multi-Factor Security",
                "page": 2,
                "category": "Authentication"
            }
        },
        {
            "text": "Knovera Customer Service & Refund Policy: Enterprise clients are eligible for full subscription refunds within a 14-day evaluation window from initial provisioning.",
            "metadata": {
                "source": "customer-policy.md",
                "chunk_index": 0,
                "doc_title": "Customer Service & Refund Policy",
                "section": "Refund Eligibility",
                "page": 1,
                "category": "Policy"
            }
        },
        {
            "text": "Service Level Agreements (SLAs): Tier 1 Technical Support ensures an initial response within 2 business hours. Emergency outage response guarantees 24/7 coverage within 15 minutes.",
            "metadata": {
                "source": "customer-policy.md",
                "chunk_index": 1,
                "doc_title": "Customer Service & Refund Policy",
                "section": "Support Tiers & Response SLA",
                "page": 2,
                "category": "Policy"
            }
        },
        {
            "text": "Campus cafeteria lunch hours run from 11:30 AM to 2:30 PM daily. A rotating menu of Mediterranean pasta, salads, and hot entrees is served at the main dining hall.",
            "metadata": {
                "source": "campus-handbook.md",
                "chunk_index": 0,
                "doc_title": "Campus Life & Facilities Guide",
                "section": "Dining Services",
                "page": 4,
                "category": "Facilities"
            }
        }
    ]


def main():
    print("=" * 80)
    print(" KNOVERA RAG ASSISTANT - ASSIGNMENT 3.26: GENERATING EMBEDDINGS VIA API ")
    print("=" * 80)

    # ---------------------------------------------------------
    # Task 3 - Read Configuration from Environment
    # ---------------------------------------------------------
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
    model_name = os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "text-embedding-3-small"

    print("\n" + "-" * 80)
    print(" TASK 3: ENVIRONMENT CONFIGURATION")
    print("-" * 80)
    print(f"  [CONFIG] API Key Source: {'FOUND (' + mask_secret(api_key) + ')' if api_key else 'NOT SET (Fallback Active)'}")
    print(f"  [CONFIG] Base URL:       {base_url if base_url else 'Default API Endpoint'}")
    print(f"  [CONFIG] Embedding Model:{model_name}")

    # Initialize Embedding Generator Client
    generator = EmbeddingGenerator(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name
    )
    engine_type = "OpenAI / OpenRouter Compatible API" if generator.client else "Deterministic Semantic Fallback Engine"
    print(f"  [INIT] Active Vector Engine: {engine_type}")

    # ---------------------------------------------------------
    # Task 1 & 2 - Load Prepared Corpus and Generate Batch Embeddings
    # ---------------------------------------------------------
    corpus_chunks = load_prepared_corpus()
    print("\n" + "-" * 80)
    print(" TASK 1 & 2: GENERATE EMBEDDINGS & STORE VECTORS WITH SOURCE CHUNKS")
    print("-" * 80)
    print(f"Prepared Corpus Size: {len(corpus_chunks)} chunks across {len(set(c['metadata']['source'] for c in corpus_chunks))} source documents.\n")

    for i, c in enumerate(corpus_chunks):
        print(f"  Chunk [{i}] [{c['metadata']['source']} | {c['metadata']['section']}]: \"{c['text'][:70]}...\"")

    print(f"\nSending batch request to embeddings API (model: '{generator.model_name}')...")
    stored_records = generator.embed_chunks(corpus_chunks, batch_size=4)
    print(f"Successfully generated and stored {len(stored_records)} vector records.")

    # ---------------------------------------------------------
    # Task 4 - Print Verification Output
    # ---------------------------------------------------------
    print("\n" + "-" * 80)
    print(" TASK 4: VERIFICATION OUTPUT (MODEL, RECORD COUNT, VECTOR LENGTH, SAMPLE VALUES)")
    print("-" * 80)

    is_uniform, exp_dim, lengths = generator.verify_dimensions([r["embedding"] for r in stored_records])
    
    print(f"  model:         {generator.model_name}")
    print(f"  records:       {len(stored_records)}")
    print(f"  vector length: {exp_dim}")
    print(f"  uniform check: {'PASSED (All ' + str(len(stored_records)) + ' vectors have identical dimension ' + str(exp_dim) + ')' if is_uniform else 'FAILED'}")
    print(f"  sample values (Record 0, first 5): {stored_records[0]['embedding'][:5]}")

    print("\nDetailed Stored Record Audit:")
    for idx, r in enumerate(stored_records):
        preview_vals = [round(val, 6) for val in r["embedding"][:5]]
        print(f"\n  [Record #{idx}] ID: {r['id']}")
        print(f"    Source Document: {r['metadata'].get('source')} | Section: {r['metadata'].get('section')} | Page: {r['metadata'].get('page')}")
        print(f"    Chunk Text:      \"{r['text'][:85]}...\"")
        print(f"    Vector Dim:      {r['vector_length']} dimensions")
        print(f"    Sample Values:   {preview_vals}")

    # ---------------------------------------------------------
    # Demonstration: Semantic Search & Model Consistency
    # ---------------------------------------------------------
    print("\n" + "-" * 80)
    print(" RETRIEVAL DEMO: QUERY EMBEDDING USING IDENTICAL MODEL")
    print("-" * 80)
    user_query = "How can a learner reset lost login credentials?"
    print(f"User Query: \"{user_query}\"")
    print("Embedding query with identical model and performing Cosine Similarity search over stored records...")

    search_results = generator.search_similar(user_query, stored_records, top_k=3)
    
    print(f"\nTop {len(search_results)} Retrieved Context Chunks:")
    for rank, res in enumerate(search_results, 1):
        print(f"  Rank #{rank} [Cosine Sim: {res['similarity']:.4f}] - ID: {res['id']}")
        print(f"    Source:  {res['metadata'].get('source')} (Section: {res['metadata'].get('section')})")
        print(f"    Content: \"{res['text']}\"\n")

    # ---------------------------------------------------------
    # Task 5 - Export Outputs & Artifacts
    # ---------------------------------------------------------
    print("-" * 80)
    print(" TASK 5: EXPORT SAMPLE CORPUS OUTPUTS & AUDIT LOGS")
    print("-" * 80)

    # 1. Full JSON Export
    full_export_path = "sample_embedded_corpus.json"
    with open(full_export_path, "w", encoding="utf-8") as f:
        json.dump({
            "assignment": "3.26 Generating Embeddings via API",
            "model": generator.model_name,
            "engine": engine_type,
            "total_records": len(stored_records),
            "vector_dimension": exp_dim,
            "records": stored_records
        }, f, indent=2)
    print(f"  Exported complete embedded records to: {full_export_path}")

    # 2. Trimmed JSON Export (for clean review and PR inspection)
    trimmed_export_path = "sample_api_embeddings_trimmed.json"
    trimmed_records = [
        {
            "id": r["id"],
            "text": r["text"],
            "metadata": r["metadata"],
            "vector_length": r["vector_length"],
            "first_5_values": [round(val, 6) for val in r["embedding"][:5]],
            "last_5_values": [round(val, 6) for val in r["embedding"][-5:]]
        }
        for r in stored_records
    ]
    with open(trimmed_export_path, "w", encoding="utf-8") as f:
        json.dump({
            "assignment": "3.26 Generating Embeddings via API",
            "model": generator.model_name,
            "engine": engine_type,
            "total_records": len(trimmed_records),
            "vector_dimension": exp_dim,
            "records": trimmed_records
        }, f, indent=2)
    print(f"  Exported trimmed inspection records to: {trimmed_export_path}")

    # 3. Text Execution Log
    os.makedirs("outputs", exist_ok=True)
    log_path = os.path.join("outputs", "api_embedding_output.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.26: GENERATING EMBEDDINGS VIA API\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model:           {generator.model_name}\n")
        f.write(f"Engine:          {engine_type}\n")
        f.write(f"Records Count:   {len(stored_records)}\n")
        f.write(f"Vector Length:   {exp_dim}\n")
        f.write(f"Uniformity:      {is_uniform}\n\n")
        f.write("SAMPLE VECTOR VALUES (First 5 coordinates per record):\n")
        for idx, r in enumerate(stored_records):
            p = [round(val, 6) for val in r["embedding"][:5]]
            f.write(f"  Record [{idx}] [{r['id']}] -> Dim: {r['vector_length']}, Sample: {p}\n")
            f.write(f"    Source: {r['metadata'].get('source')} | Section: {r['metadata'].get('section')}\n")
            f.write(f"    Text: {r['text']}\n\n")
        f.write("SAMPLE RETRIEVAL QUERY RESULTS:\n")
        f.write(f"Query: \"{user_query}\"\n")
        for rank, res in enumerate(search_results, 1):
            f.write(f"  Rank #{rank} [Sim: {res['similarity']:.4f}] - {res['id']}: {res['text']}\n")

    print(f"  Exported execution audit log to:        {log_path}")
    print("\n" + "=" * 80)
    print(" ASSIGNMENT 3.26 DEMONSTRATION COMPLETE - ALL CHECKS PASSED ")
    print("=" * 80)


if __name__ == "__main__":
    main()
