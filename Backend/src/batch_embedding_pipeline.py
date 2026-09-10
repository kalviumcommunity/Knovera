"""
src/batch_embedding_pipeline.py

Enterprise Batch Embedding, Rate-Limit Resilience, and Token Cost Management Pipeline.
Designed for Knovera RAG Assistant.

Key Capabilities:
1. Configurable Batching: Groups chunks into multi-item requests to minimize network round-trips.
2. Exponential Backoff & Jitter: Automatically catches transient errors / HTTP 429 rate limits and retries with progressive delays.
3. Cost & Token Tracking: Accurately counts tokens via tiktoken and computes approximate financial costs per run.
4. Resumable Processing (Durable Jobs): Detects previously embedded chunks from persistent storage to skip redundant API calls.
5. Rich Run Summary: Produces comprehensive audit analytics on throughput, tokens, costs, skipped chunks, and error logs.
"""

import os
import sys
import time
import math
import random
import logging
from typing import List, Dict, Any, Set, Optional, Tuple, Iterator
from dotenv import load_dotenv

# Try importing tiktoken for exact token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Try importing OpenAI client
try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    RateLimitError = Exception
    APIConnectionError = Exception
    APITimeoutError = Exception
    InternalServerError = Exception
    APIError = Exception

from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)

# Standard Model Embedding Pricing (USD per 1,000 input tokens)
MODEL_PRICING_PER_1K: Dict[str, float] = {
    "text-embedding-3-small": 0.00002,     # $0.02 per 1M tokens
    "openai/text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,     # $0.13 per 1M tokens
    "openai/text-embedding-3-large": 0.00013,
    "text-embedding-ada-002": 0.00010,     # $0.10 per 1M tokens
    "openai/text-embedding-ada-002": 0.00010,
    "default": 0.00002
}


class BatchEmbeddingPipeline:
    """
    Production-grade batch embedding pipeline with automatic retries,
    rate-limit backoff, token budgeting, cost estimation, and skip-on-rerun caching.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        model_name: Optional[str] = None,
        price_per_1k_tokens: Optional[float] = None,
        storage_path: Optional[str] = None,
        generator: Optional[EmbeddingGenerator] = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the BatchEmbeddingPipeline.

        Args:
            batch_size: Number of text chunks sent per single embedding request (default: 32).
            max_retries: Maximum number of retry attempts per batch upon transient failure (default: 5).
            initial_delay: Initial sleep duration in seconds before first retry (default: 1.0s).
            backoff_factor: Multiplier for exponential backoff delay calculation (default: 2.0).
            max_delay: Cap on maximum retry sleep duration in seconds (default: 60.0s).
            jitter: Whether to add random jitter (0-25%) to prevent thundering herd collisions.
            model_name: Embedding model identifier (defaults to env or text-embedding-3-small).
            price_per_1k_tokens: Custom token cost override in USD.
            storage_path: Path to persistent storage JSON file for caching and durable resumption.
            generator: Optional pre-configured EmbeddingGenerator instance.
            encoding_name: Tiktoken tokenizer encoding name (default: cl100k_base).
        """
        load_dotenv()
        self.batch_size = max(1, batch_size)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.storage_path = storage_path

        # Initialize or reuse underlying embedding generator
        self.generator = generator or EmbeddingGenerator(model_name=model_name)
        self.model_name = self.generator.model_name

        # Resolve cost per 1k tokens
        if price_per_1k_tokens is not None:
            self.price_per_1k_tokens = price_per_1k_tokens
        else:
            model_key = self.model_name.lower()
            self.price_per_1k_tokens = MODEL_PRICING_PER_1K.get(model_key, MODEL_PRICING_PER_1K["default"])

        # Tokenizer initialization
        self.encoding_name = encoding_name
        self.tokenizer = None
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception:
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.tokenizer = None

        # Existing embeddings cache: id -> record dictionary
        self.existing_records: Dict[str, Dict[str, Any]] = {}
        if self.storage_path and os.path.exists(self.storage_path):
            self.load_existing_storage(self.storage_path)

        # Simulation hooks for testing rate limit & failure scenarios
        self._simulated_failure_countdown: int = 0
        self._simulated_exception_type: type = Exception

    # -------------------------------------------------------------------------
    # Token Estimation & Cost Accounting
    # -------------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """
        Calculates exact token count using tiktoken (or robust 4-char heuristic fallback).
        """
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Heuristic fallback: ~4 characters per token in standard English corpora
        return max(1, math.ceil(len(text) / 4.0))

    def estimate_batch_tokens(self, texts: List[str]) -> int:
        """Calculates the total token sum for a batch of text strings."""
        return sum(self.count_tokens(t) for t in texts)

    def calculate_cost(self, token_count: int) -> float:
        """Computes approximate USD cost for the provided token count."""
        return (token_count / 1000.0) * self.price_per_1k_tokens

    # -------------------------------------------------------------------------
    # Batch Partitioning & ID Resolution
    # -------------------------------------------------------------------------

    @staticmethod
    def get_chunk_id(chunk: Dict[str, Any], fallback_idx: int = 0) -> str:
        """
        Derives a deterministic unique ID for a chunk using id property or metadata.
        """
        if chunk.get("id"):
            return str(chunk["id"])
        meta = chunk.get("metadata", {})
        source = meta.get("source", meta.get("doc_title", "doc"))
        idx = meta.get("chunk_index", fallback_idx)
        return f"{source}#chunk_{idx}"

    @staticmethod
    def create_batches(items: List[Any], size: int) -> Iterator[List[Any]]:
        """
        Splits a list of items into contiguous slices of size `size`.
        """
        for start in range(0, len(items), size):
            yield items[start:start + size]

    # -------------------------------------------------------------------------
    # Storage & Persistence (Resumable Pipeline / Durable Jobs)
    # -------------------------------------------------------------------------

    def load_existing_storage(self, filepath: str) -> int:
        """
        Loads pre-existing embedded records from disk into memory cache.
        Returns the number of loaded existing records.
        """
        import json
        if not os.path.exists(filepath):
            return 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data.get("records", data) if isinstance(data, dict) else data
                if isinstance(records, list):
                    for r in records:
                        if isinstance(r, dict) and "id" in r:
                            self.existing_records[r["id"]] = r
            logger.info(f"Loaded {len(self.existing_records)} existing embedded records from {filepath}")
            return len(self.existing_records)
        except Exception as e:
            logger.warning(f"Failed to load existing storage file ({e}). Starting with empty cache.")
            return 0

    def save_to_storage(self, filepath: Optional[str] = None) -> bool:
        """
        Flushes the current in-memory records cache to disk as formatted JSON.
        """
        import json
        target_path = filepath or self.storage_path
        if not target_path:
            return False
        try:
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            export_data = {
                "pipeline": "Knovera BatchEmbeddingPipeline",
                "model": self.model_name,
                "total_records": len(self.existing_records),
                "records": list(self.existing_records.values())
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error persisting embeddings to {target_path}: {e}")
            return False

    def get_existing_ids(self) -> Set[str]:
        """Returns the set of all unique record IDs currently in the persistent store."""
        return set(self.existing_records.keys())

    # -------------------------------------------------------------------------
    # Retry with Exponential Backoff & Jitter
    # -------------------------------------------------------------------------

    def embed_with_retry(
        self,
        texts: List[str],
        batch_idx: int = 0
    ) -> Tuple[List[List[float]], int]:
        """
        Generates vector embeddings for a list of texts with exponential backoff retry.
        
        Formula:
            Wait Time = min(initial_delay * (backoff_factor ** attempt), max_delay) + jitter
            
        Args:
            texts: List of text strings to embed in a single request.
            batch_idx: Index of current batch for structured logging.
            
        Returns:
            Tuple[List[List[float]], int]: (embeddings, retry_count)
            
        Raises:
            Exception: If all max_retries attempts are exhausted.
        """
        if not texts:
            return [], 0

        retry_count = 0

        for attempt in range(self.max_retries):
            try:
                # Handle simulated error injection for test/demo environments
                if self._simulated_failure_countdown > 0:
                    self._simulated_failure_countdown -= 1
                    raise self._simulated_exception_type(
                        f"Simulated rate limit / 429 Too Many Requests on batch {batch_idx}"
                    )

                # Execute embedding generation via client or fallback
                if self.generator.client:
                    response = self.generator.client.embeddings.create(
                        input=texts,
                        model=self.model_name
                    )
                    embeddings = [item.embedding for item in response.data]
                    return embeddings, retry_count
                else:
                    # Offline deterministic fallback engine
                    embeddings = self.generator._generate_fallback_embeddings(texts)
                    return embeddings, retry_count

            except Exception as error:
                retry_count += 1
                if attempt == self.max_retries - 1:
                    logger.error(f"Batch {batch_idx}: All {self.max_retries} attempts exhausted. Fatal error: {error}")
                    raise

                # Calculate Exponential Backoff with Jitter
                base_delay = self.initial_delay * (self.backoff_factor ** attempt)
                capped_delay = min(base_delay, self.max_delay)
                jitter_val = random.uniform(0.0, 0.25 * capped_delay) if self.jitter else 0.0
                wait_seconds = round(capped_delay + jitter_val, 3)

                logger.warning(
                    f"Batch {batch_idx} [Attempt {attempt + 1}/{self.max_retries}] failed: {error}. "
                    f"Retrying in {wait_seconds}s with exponential backoff..."
                )
                time.sleep(wait_seconds)

        return [], retry_count

    # -------------------------------------------------------------------------
    # Main Batch Pipeline Execution
    # -------------------------------------------------------------------------

    def run(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        auto_persist: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the batch embedding pipeline over the supplied corpus of chunks.
        
        Features:
        - Identifies and skips chunks that have already been embedded (skip-on-rerun).
        - Divides pending chunks into configurable batches.
        - Calculates token usage and estimated cost.
        - Executes requests with exponential backoff retry on transient/rate-limit errors.
        - Durable state saving after each successful batch.
        - Produces a comprehensive run summary audit.
        
        Args:
            chunks: List of chunk dictionaries containing 'text' and optional 'id'/'metadata'.
            batch_size: Optional override for batch size.
            auto_persist: Whether to write checkpoints to storage_path after each batch.
            
        Returns:
            Dict[str, Any]: Detailed execution summary containing metrics, cost, and results.
        """
        start_time = time.time()
        effective_batch_size = batch_size or self.batch_size
        total_chunks = len(chunks)

        # Step 1: Detect existing chunk IDs & filter pending work
        existing_ids = self.get_existing_ids()
        
        pending_chunks: List[Tuple[int, Dict[str, Any], str]] = []
        skipped_chunks: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            cid = self.get_chunk_id(chunk, fallback_idx=idx)
            if cid in existing_ids:
                skipped_chunks.append({
                    "id": cid,
                    "text_preview": chunk.get("text", "")[:60] + "...",
                    "status": "skipped_existing"
                })
            else:
                pending_chunks.append((idx, chunk, cid))

        # Initialize Run Summary Metrics
        summary: Dict[str, Any] = {
            "pipeline": "BatchEmbeddingPipeline",
            "model": self.model_name,
            "batch_size": effective_batch_size,
            "total_chunks": total_chunks,
            "existing_chunks": len(existing_ids),
            "skipped_chunks": len(skipped_chunks),
            "pending_chunks": len(pending_chunks),
            "embedded_chunks": 0,
            "failed_chunks": 0,
            "total_batches": math.ceil(len(pending_chunks) / effective_batch_size) if pending_chunks else 0,
            "batches_processed": 0,
            "batches_failed": 0,
            "retries_attempted": 0,
            "input_tokens": 0,
            "estimated_cost_usd": 0.0,
            "price_per_1k_tokens": self.price_per_1k_tokens,
            "duration_seconds": 0.0,
            "throughput_chunks_per_sec": 0.0,
            "throughput_tokens_per_sec": 0.0,
            "failures": [],
            "status": "SUCCESS"
        }

        if not pending_chunks:
            # Everything is already embedded
            duration = time.time() - start_time
            summary["duration_seconds"] = round(duration, 4)
            summary["status"] = "ALL_CHUNKS_ALREADY_EMBEDDED"
            return summary

        # Step 2: Process Pending Chunks in Configurable Batches
        batch_generator = self.create_batches(pending_chunks, effective_batch_size)

        for b_idx, batch in enumerate(batch_generator, start=1):
            batch_texts = [item[1].get("text", "") for item in batch]
            batch_cids = [item[2] for item in batch]
            batch_chunks = [item[1] for item in batch]
            batch_original_indices = [item[0] for item in batch]

            # Token Counting & Cost Accumulation
            batch_tokens = self.estimate_batch_tokens(batch_texts)
            summary["input_tokens"] += batch_tokens

            try:
                # Attempt Embedding with Exponential Retry & Backoff
                embeddings, retries = self.embed_with_retry(batch_texts, batch_idx=b_idx)
                summary["retries_attempted"] += retries
                summary["batches_processed"] += 1
                summary["embedded_chunks"] += len(embeddings)

                # Bind Embeddings with Metadata and Persist to Cache
                for orig_idx, c_dict, cid, emb in zip(batch_original_indices, batch_chunks, batch_cids, embeddings):
                    metadata = dict(c_dict.get("metadata", {}))
                    source = metadata.get("source", metadata.get("doc_title", "corpus_doc"))
                    chunk_idx = metadata.get("chunk_index", orig_idx)

                    record = {
                        "id": cid,
                        "text": c_dict.get("text", ""),
                        "metadata": metadata,
                        "embedding": emb,
                        "vector_length": len(emb),
                        "model": self.model_name
                    }
                    self.existing_records[cid] = record

                # Durable checkpoint save after each successful batch
                if auto_persist and self.storage_path:
                    self.save_to_storage(self.storage_path)

            except Exception as batch_error:
                summary["batches_failed"] += 1
                summary["failed_chunks"] += len(batch)
                summary["status"] = "PARTIAL_FAILURE"
                summary["failures"].append({
                    "batch_index": b_idx,
                    "chunk_ids": batch_cids,
                    "error_message": str(batch_error)
                })
                logger.error(f"Batch {b_idx} failed permanently: {batch_error}")

        # Finalize Metrics
        end_time = time.time()
        duration = max(0.001, end_time - start_time)
        summary["duration_seconds"] = round(duration, 4)
        summary["estimated_cost_usd"] = round(self.calculate_cost(summary["input_tokens"]), 6)
        summary["throughput_chunks_per_sec"] = round(summary["embedded_chunks"] / duration, 2)
        summary["throughput_tokens_per_sec"] = round(summary["input_tokens"] / duration, 2)

        if summary["failed_chunks"] > 0 and summary["embedded_chunks"] == 0:
            summary["status"] = "FAILED"

        return summary

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Returns all embedded records stored in the cache."""
        return list(self.existing_records.values())

    def reset_storage(self):
        """Clears in-memory storage cache and deletes storage file if configured."""
        self.existing_records.clear()
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
            except Exception as e:
                logger.warning(f"Could not remove storage file {self.storage_path}: {e}")
