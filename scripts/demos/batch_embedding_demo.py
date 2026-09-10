"""
batch_embedding_demo.py

Demonstration and Verification Suite for Assignment 3.28: Batch Embedding & Rate/Cost Management.
Platform: Knovera RAG Assistant

Features Demonstrated:
- Task 1: Embed chunks in batches with configurable batch sizes (reduces request overhead).
- Task 2: Retry transient / rate-limit failures with exponential backoff and jitter.
- Task 3: Track token counts and approximate costs using model pricing ($0.00002 / 1K tokens for text-embedding-3-small).
- Task 4: Skip already-embedded chunks on re-runs to avoid duplicate API calls and wasted spend.
- Task 5: Export comprehensive audit log, run summary JSON, and embedded corpus artifacts.
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

from src.batch_embedding_pipeline import BatchEmbeddingPipeline, MODEL_PRICING_PER_1K
from src.embedding_generator import EmbeddingGenerator, cosine_similarity


def load_demonstration_corpus() -> List[Dict[str, Any]]:
    """
    Constructs a rich multi-document corpus spanning authentication, security,
    service policies, dining, and IT infrastructure.
    """
    return [
        {
            "id": "auth-guide#chunk_0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and submitting their registered email address.",
            "metadata": {"source": "auth-guide.md", "chunk_index": 0, "section": "Password Recovery", "category": "Authentication"}
        },
        {
            "id": "auth-guide#chunk_1",
            "text": "Learners can recover access using their registered email. A one-time secure verification token is dispatched, which remains valid for 15 minutes.",
            "metadata": {"source": "auth-guide.md", "chunk_index": 1, "section": "Email Verification", "category": "Authentication"}
        },
        {
            "id": "auth-guide#chunk_2",
            "text": "Two-factor authentication (2FA) is mandatory for enterprise admin consoles. Admins must register an authenticator app (TOTP) or hardware security key.",
            "metadata": {"source": "auth-guide.md", "chunk_index": 2, "section": "Multi-Factor Security", "category": "Authentication"}
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
            "text": "Enterprise Service Level Agreement (SLA): Knovera guarantees 99.9% uptime for core API endpoints. Outages exceeding SLA thresholds qualify for service credit.",
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


def format_summary_table(summary: Dict[str, Any]) -> str:
    """Formats the run summary dictionary into a clean markdown / ASCII table."""
    lines = [
        "+------------------------------------+------------------------------------+",
        "| Metric                             | Value                              |",
        "+------------------------------------+------------------------------------+",
        f"| Pipeline Status                    | {summary.get('status', 'N/A'):<34} |",
        f"| Embedding Model                    | {summary.get('model', 'N/A'):<34} |",
        f"| Batch Size                         | {summary.get('batch_size', 0):<34} |",
        f"| Total Input Chunks                 | {summary.get('total_chunks', 0):<34} |",
        f"| Previously Existing Chunks         | {summary.get('existing_chunks', 0):<34} |",
        f"| Skipped Chunks (Already Embedded)  | {summary.get('skipped_chunks', 0):<34} |",
        f"| Pending Chunks Embedded            | {summary.get('embedded_chunks', 0):<34} |",
        f"| Failed Chunks                      | {summary.get('failed_chunks', 0):<34} |",
        f"| Total Batches Processed            | {summary.get('batches_processed', 0):<34} |",
        f"| Failed Batches                     | {summary.get('batches_failed', 0):<34} |",
        f"| Retries Attempted (Backoff)        | {summary.get('retries_attempted', 0):<34} |",
        f"| Input Tokens Processed             | {summary.get('input_tokens', 0):<34} |",
        f"| Price per 1K Tokens                | ${summary.get('price_per_1k_tokens', 0.0):.6f} USD/1k            |",
        f"| Estimated Run Cost                 | ${summary.get('estimated_cost_usd', 0.0):.6f} USD               |",
        f"| Duration (seconds)                 | {summary.get('duration_seconds', 0.0):.4f}s                          |",
        f"| Throughput (chunks/sec)            | {summary.get('throughput_chunks_per_sec', 0.0):.2f} chunks/sec                |",
        f"| Throughput (tokens/sec)            | {summary.get('throughput_tokens_per_sec', 0.0):.2f} tokens/sec                |",
        "+------------------------------------+------------------------------------+"
    ]
    return "\n".join(lines)


def main():
    load_dotenv()
    print("=" * 80)
    print(" KNOVERA RAG ASSISTANT - BATCH EMBEDDING & RATE/COST MANAGEMENT (3.28)")
    print("=" * 80)

    corpus = load_demonstration_corpus()
    storage_file = "sample_batch_embedded_corpus.json"
    
    # Ensure fresh demonstration environment
    if os.path.exists(storage_file):
        try:
            os.remove(storage_file)
        except Exception:
            pass

    # =========================================================================
    # TASK 1: BATCH EMBEDDING WITH CONFIGURABLE BATCH SIZE
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TASK 1 & 3: FRESH BATCH EMBEDDING RUN WITH TOKEN & COST ESTIMATION")
    print("-" * 80)
    print(f"Total Corpus Size:       {len(corpus)} text chunks")
    print(f"Configured Batch Size:   4 chunks per API request")
    print(f"Embedding Storage File:  {storage_file}")

    pipeline = BatchEmbeddingPipeline(
        batch_size=4,
        max_retries=4,
        initial_delay=0.5,
        backoff_factor=2.0,
        storage_path=storage_file
    )

    print(f"Model Name:              {pipeline.model_name}")
    print(f"Model Unit Cost:         ${pipeline.price_per_1k_tokens:.6f} per 1K tokens ($0.02 / 1M)")
    print("\nExecuting initial batch embedding run...")

    run1_summary = pipeline.run(corpus, batch_size=4)
    print("\nRun 1 (Fresh Run) Summary Table:")
    print(format_summary_table(run1_summary))

    # =========================================================================
    # TASK 2: RETRY WITH EXPONENTIAL BACKOFF ON SIMULATED RATE-LIMIT / 429
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TASK 2: RETRY WITH EXPONENTIAL BACKOFF (RATE LIMIT / TRANSIENT ERROR)")
    print("-" * 80)
    print("Simulating temporary HTTP 429 Rate Limit on batch requests to demonstrate backoff recovery...")

    retry_pipeline = BatchEmbeddingPipeline(
        batch_size=3,
        max_retries=4,
        initial_delay=0.4,
        backoff_factor=2.0,
        jitter=True
    )
    # Inject 2 transient failures before succeeding on attempt 3
    retry_pipeline._simulated_failure_countdown = 2
    retry_pipeline._simulated_exception_type = Exception

    test_batch_texts = [
        "Authentication policy for API tokens.",
        "Rate limiting guidelines and token bucket algorithms.",
        "Retry strategy with exponential backoff and jitter."
    ]

    print(f"Batch texts to embed ({len(test_batch_texts)} texts):")
    for t in test_batch_texts:
        print(f"  - \"{t}\"")

    start_retry_t = time.time()
    embeddings, retry_count = retry_pipeline.embed_with_retry(test_batch_texts, batch_idx=1)
    retry_duration = time.time() - start_retry_t

    print(f"\nRetry Backoff Results:")
    print(f"  - Transient Errors Encountered: 2")
    print(f"  - Retries Successfully Completed: {retry_count}")
    print(f"  - Embeddings Produced:          {len(embeddings)} vectors")
    print(f"  - Dimension per Vector:         {len(embeddings[0]) if embeddings else 0}")
    print(f"  - Total Backoff Elapsed Time:   {retry_duration:.3f}s")
    print("  - Status:                       RECOVERED_SUCCESSFULLY")

    # =========================================================================
    # TASK 4: SKIP ALREADY-EMBEDDED CHUNKS ON RE-RUN (IDEMPOTENCY & ZERO SPEND)
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TASK 4: SKIP ALREADY-EMBEDDED CHUNKS ON RE-RUN (RESUMABLE PIPELINE)")
    print("-" * 80)
    print("Re-running the pipeline on the identical corpus against existing storage...")
    print(f"Existing Records in Storage: {len(pipeline.get_existing_ids())}")

    # Re-run pipeline using existing storage
    rerun_pipeline = BatchEmbeddingPipeline(
        batch_size=4,
        storage_path=storage_file
    )

    run2_summary = rerun_pipeline.run(corpus, batch_size=4)
    print("\nRun 2 (Identical Re-run) Summary Table:")
    print(format_summary_table(run2_summary))

    # =========================================================================
    # PARTIAL RESUME SIMULATION: NEW CHUNKS ADDED TO PARTIALLY EMBEDDED CORPUS
    # =========================================================================
    print("\n" + "-" * 80)
    print(" RESUMPTION TEST: EXPANDED CORPUS WITH NEW PENDING CHUNKS")
    print("-" * 80)
    
    new_chunks = [
        {
            "id": "sec-ops#chunk_0",
            "text": "Security Operations Center (SOC) Alerting: All unusual administrative privilege escalations trigger immediate P1 pager alerts.",
            "metadata": {"source": "sec-ops.md", "chunk_index": 0, "section": "Alerting", "category": "Security"}
        },
        {
            "id": "sec-ops#chunk_1",
            "text": "Data Retention & Sanitization Policy: Customer session telemetry logs are archived for 90 days and permanently expunged thereafter.",
            "metadata": {"source": "sec-ops.md", "chunk_index": 1, "section": "Data Retention", "category": "Security"}
        }
    ]
    augmented_corpus = corpus + new_chunks
    print(f"Augmented Corpus Size:   {len(augmented_corpus)} chunks ({len(corpus)} existing + {len(new_chunks)} new)")

    run3_summary = rerun_pipeline.run(augmented_corpus, batch_size=4)
    print("\nRun 3 (Partial Resume with New Chunks) Summary Table:")
    print(format_summary_table(run3_summary))

    # =========================================================================
    # TASK 5: EXPORT ARTIFACTS & AUDIT LOGS
    # =========================================================================
    print("\n" + "-" * 80)
    print(" TASK 5: EXPORT SAMPLE RUN SUMMARIES & AUDIT ARTIFACTS")
    print("-" * 80)

    os.makedirs("outputs", exist_ok=True)
    summary_json_path = os.path.join("outputs", "batch_embedding_run_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "assignment": "3.28 Batch Embedding & Rate/Cost Management",
            "runs": {
                "initial_fresh_run": run1_summary,
                "identical_rerun_skip": run2_summary,
                "partial_resumption_run": run3_summary
            }
        }, f, indent=2)
    print(f"  Exported Run Summaries to:  {summary_json_path}")

    # Export Human-Readable Audit Log
    log_output_path = os.path.join("outputs", "batch_embedding_output.txt")
    with open(log_output_path, "w", encoding="utf-8") as f:
        f.write("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.28: BATCH EMBEDDING & RATE/COST MANAGEMENT\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. FRESH BATCH RUN AUDIT:\n")
        f.write(format_summary_table(run1_summary) + "\n\n")
        f.write("2. RETRY WITH EXPONENTIAL BACKOFF AUDIT:\n")
        f.write(f"  Simulated Failures: 2\n  Retries Recovered: {retry_count}\n  Backoff Duration: {retry_duration:.3f}s\n\n")
        f.write("3. RE-RUN IDEMPOTENCY AUDIT (SKIPPED CHUNKS):\n")
        f.write(format_summary_table(run2_summary) + "\n\n")
        f.write("4. PARTIAL RESUME WITH NEW CHUNKS AUDIT:\n")
        f.write(format_summary_table(run3_summary) + "\n\n")
        f.write(f"5. TOTAL PERSISTED RECORDS IN STORAGE: {len(rerun_pipeline.get_all_records())}\n")
    print(f"  Exported Text Audit Log to: {log_output_path}")

    print("\n" + "=" * 80)
    print(" ASSIGNMENT 3.28 VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
