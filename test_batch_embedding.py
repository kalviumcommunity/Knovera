"""
test_batch_embedding.py

Comprehensive Unit Test Suite for Assignment 3.28: Batch Embedding & Rate/Cost Management.
Tests:
- Configurable batch partitioning
- Token counting and cost estimation formulas
- Retry with exponential backoff and transient error recovery
- Recording permanent failures in run summary
- Skipping previously embedded chunks (idempotent re-runs)
- Checkpoint persistence and durable job resumption
- Output integrity and dimension consistency
"""

import os
import json
import unittest
from typing import List, Dict, Any

from src.batch_embedding_pipeline import BatchEmbeddingPipeline, MODEL_PRICING_PER_1K
from src.embedding_generator import EmbeddingGenerator


class TestBatchEmbeddingPipeline(unittest.TestCase):

    def setUp(self):
        self.test_storage = "test_temp_embeddings.json"
        if os.path.exists(self.test_storage):
            try:
                os.remove(self.test_storage)
            except Exception:
                pass

        self.sample_chunks = [
            {
                "id": "doc1#chunk_0",
                "text": "How to reset your user account password in the portal.",
                "metadata": {"source": "doc1.md", "chunk_index": 0}
            },
            {
                "id": "doc1#chunk_1",
                "text": "Email verification links expire after 15 minutes.",
                "metadata": {"source": "doc1.md", "chunk_index": 1}
            },
            {
                "id": "doc2#chunk_0",
                "text": "Customer service refund policies and processing windows.",
                "metadata": {"source": "doc2.md", "chunk_index": 0}
            },
            {
                "id": "doc2#chunk_1",
                "text": "Enterprise service level agreements guarantee 99.9% uptime.",
                "metadata": {"source": "doc2.md", "chunk_index": 1}
            },
            {
                "id": "doc3#chunk_0",
                "text": "Campus cafeteria dining menu offers pizza and pasta daily.",
                "metadata": {"source": "doc3.md", "chunk_index": 0}
            }
        ]

    def tearDown(self):
        if os.path.exists(self.test_storage):
            try:
                os.remove(self.test_storage)
            except Exception:
                pass

    def test_batch_creation_partitioning(self):
        """Verify that items are sliced into exact batch sizes."""
        items = list(range(10))
        batches = list(BatchEmbeddingPipeline.create_batches(items, size=3))
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0], [0, 1, 2])
        self.assertEqual(batches[1], [3, 4, 5])
        self.assertEqual(batches[2], [6, 7, 8])
        self.assertEqual(batches[3], [9])

    def test_token_counting_and_cost_estimation(self):
        """Verify token calculation and mathematical pricing accuracy."""
        pipeline = BatchEmbeddingPipeline(price_per_1k_tokens=0.00002)
        text = "This is a sample test prompt for token estimation."
        tokens = pipeline.count_tokens(text)
        self.assertGreater(tokens, 0)
        
        # 10,000 tokens at $0.00002/1k should equal $0.000200
        cost = pipeline.calculate_cost(10000)
        self.assertAlmostEqual(cost, 0.000200, places=6)

    def test_fresh_batch_embedding_run(self):
        """Verify standard batch embedding run creates embeddings and summary metrics."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            storage_path=self.test_storage
        )
        summary = pipeline.run(self.sample_chunks)

        self.assertEqual(summary["status"], "SUCCESS")
        self.assertEqual(summary["total_chunks"], 5)
        self.assertEqual(summary["embedded_chunks"], 5)
        self.assertEqual(summary["skipped_chunks"], 0)
        self.assertEqual(summary["batches_processed"], 3) # 5 items with batch_size 2 => 3 batches
        self.assertEqual(summary["failed_chunks"], 0)
        self.assertGreater(summary["input_tokens"], 0)
        self.assertGreater(summary["estimated_cost_usd"], 0.0)

        records = pipeline.get_all_records()
        self.assertEqual(len(records), 5)
        for r in records:
            self.assertIn("id", r)
            self.assertIn("embedding", r)
            self.assertEqual(len(r["embedding"]), 1536)

    def test_skip_already_embedded_chunks_on_rerun(self):
        """Verify that re-running over existing corpus skips all chunks and incurs 0 cost."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            storage_path=self.test_storage
        )
        # First Run
        run1 = pipeline.run(self.sample_chunks)
        self.assertEqual(run1["embedded_chunks"], 5)

        # Second Run on same storage
        rerun_pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            storage_path=self.test_storage
        )
        run2 = rerun_pipeline.run(self.sample_chunks)

        self.assertEqual(run2["total_chunks"], 5)
        self.assertEqual(run2["skipped_chunks"], 5)
        self.assertEqual(run2["pending_chunks"], 0)
        self.assertEqual(run2["embedded_chunks"], 0)
        self.assertEqual(run2["input_tokens"], 0)
        self.assertEqual(run2["estimated_cost_usd"], 0.0)
        self.assertEqual(run2["status"], "ALL_CHUNKS_ALREADY_EMBEDDED")

    def test_partial_corpus_resumption(self):
        """Verify adding new chunks embeds ONLY the new chunks while preserving previous."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            storage_path=self.test_storage
        )
        # Run first 3 chunks
        run1 = pipeline.run(self.sample_chunks[:3])
        self.assertEqual(run1["embedded_chunks"], 3)

        # Run all 5 chunks
        run2 = pipeline.run(self.sample_chunks)
        self.assertEqual(run2["total_chunks"], 5)
        self.assertEqual(run2["skipped_chunks"], 3)
        self.assertEqual(run2["pending_chunks"], 2)
        self.assertEqual(run2["embedded_chunks"], 2)
        self.assertEqual(len(pipeline.get_all_records()), 5)

    def test_retry_with_exponential_backoff_recovery(self):
        """Verify pipeline retries upon transient errors and succeeds when errors clear."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            max_retries=4,
            initial_delay=0.1,
            backoff_factor=1.5,
            jitter=False
        )
        # Simulate 2 temporary failures
        pipeline._simulated_failure_countdown = 2
        
        texts = ["Text item 1", "Text item 2"]
        embeddings, retries = pipeline.embed_with_retry(texts, batch_idx=1)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(retries, 2)

    def test_permanent_failure_recorded_in_summary(self):
        """Verify that exhausting retries records failure in summary without silent swallowing."""
        pipeline = BatchEmbeddingPipeline(
            batch_size=2,
            max_retries=2,
            initial_delay=0.05,
            backoff_factor=1.5,
            jitter=False
        )
        # Simulate 5 failures (exceeds max_retries of 2)
        pipeline._simulated_failure_countdown = 5

        summary = pipeline.run(self.sample_chunks[:2])
        self.assertIn(summary["status"], ["PARTIAL_FAILURE", "FAILED"])
        self.assertEqual(summary["batches_failed"], 1)
        self.assertEqual(summary["failed_chunks"], 2)
        self.assertEqual(len(summary["failures"]), 1)


if __name__ == "__main__":
    unittest.main()
