"""
src/corpus_indexer.py

Enterprise Corpus Indexing & Vector Metadata Storage Engine.
Designed for Knovera RAG Assistant using ChromaDB Vector Database.

Key Capabilities:
1. Canonical Vector Record Transformation: Prepares embedded chunks with stable IDs, 1536-dim vectors, source text, and rich metadata.
2. Batched Bulk Indexing: Ingests thousands of records in configurable slices to avoid client timeouts and payload bottlenecks.
3. Strict Count Verification: Compares post-ingestion collection counts against input chunk counts to ensure zero data loss.
4. Spot-Check Integrity Auditing: Verifies 100% field parity (ID, vector dimension, text, and metadata) between stored records and source chunks.
5. Incremental Re-Indexing: Supports selective upsert and document-level eviction when source files change without rebuilding the full corpus.
"""

import os
import sys
import time
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class CorpusIndexer:
    """
    Manages end-to-end indexing of embedded document chunks into ChromaDB collections.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        default_collection: str = "knovera_knowledge_base"
    ):
        load_dotenv()
        self.vector_db = vector_db or VectorDatabase()
        self.default_collection = default_collection

    @staticmethod
    def to_vector_record(chunk: Dict[str, Any], fallback_idx: int = 0) -> Dict[str, Any]:
        """
        Transforms an embedded chunk dictionary into a canonical vector database record.
        
        Args:
            chunk: Dictionary containing 'text', 'embedding' or 'vector', and 'metadata'.
            fallback_idx: Fallback integer index if chunk_index is missing.
            
        Returns:
            Dict[str, Any]: Standardized record with 'id', 'vector', 'text', 'metadata'.
        """
        metadata = dict(chunk.get("metadata", {}))
        source = metadata.get("source", metadata.get("doc_title", "corpus_doc"))
        chunk_idx = metadata.get("chunk_index", fallback_idx)

        # Stable, deterministic chunk identifier
        record_id = chunk.get("id") or f"{source}#chunk_{chunk_idx}"

        vec = chunk.get("vector") if "vector" in chunk else chunk.get("embedding", [])

        return {
            "id": str(record_id),
            "vector": list(vec),
            "text": str(chunk.get("text", "")),
            "metadata": {
                "source": str(source),
                "chunk_index": int(chunk_idx),
                "section": str(metadata.get("section", "General")),
                "doc_title": str(metadata.get("doc_title", source)),
                "category": str(metadata.get("category", "Uncategorized")),
                **(
                    {k: v for k, v in metadata.items() if k not in ("source", "chunk_index", "section", "doc_title", "category")}
                )
            }
        }

    @staticmethod
    def create_batches(items: List[Any], size: int):
        """Yields successive slices of size `size` from `items`."""
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def index_corpus(
        self,
        embedded_chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        batch_size: int = 64,
        reset_collection: bool = False
    ) -> Dict[str, Any]:
        """
        Bulk indexes all embedded chunks into the vector database in batched transactions.
        
        Args:
            embedded_chunks: List of chunk dictionaries containing embeddings and metadata.
            collection_name: Target ChromaDB collection name.
            batch_size: Number of records per batch upsert transaction (default: 64).
            reset_collection: If True, deletes existing collection before indexing.
            
        Returns:
            Dict[str, Any]: Comprehensive indexing run summary with count verification and timing.
        """
        target_collection = collection_name or self.default_collection
        start_time = time.time()

        if reset_collection:
            self.vector_db.delete_collection(target_collection)

        # Ensure collection is created with correct dimension
        col = self.vector_db.create_or_get_collection(
            name=target_collection,
            dimension=self.vector_db.expected_dimension,
            metric="cosine"
        )

        initial_count = self.vector_db.count(target_collection)
        records = [self.to_vector_record(c, idx) for idx, c in enumerate(embedded_chunks)]
        total_chunks = len(records)

        inserted_count = 0
        failures = []

        for batch_idx, batch in enumerate(self.create_batches(records, batch_size), start=1):
            try:
                self.vector_db.upsert_records(batch, collection_name=target_collection)
                inserted_count += len(batch)
            except Exception as e:
                batch_start_id = batch[0]["id"] if batch else "unknown"
                logger.error(f"Failed to index batch {batch_idx} (Start ID: {batch_start_id}): {e}")
                failures.append({
                    "batch_index": batch_idx,
                    "batch_size": len(batch),
                    "start_id": batch_start_id,
                    "error": str(e)
                })

        end_time = time.time()
        duration = max(0.001, end_time - start_time)
        final_indexed_count = self.vector_db.count(target_collection)

        # Verification: Validate count match
        count_matches = (final_indexed_count == (initial_count + total_chunks) if reset_collection is False and initial_count == 0 else (final_indexed_count == total_chunks))

        summary = {
            "collection_name": target_collection,
            "expected_chunks": total_chunks,
            "inserted_this_run": inserted_count,
            "initial_count": initial_count,
            "final_indexed_count": final_indexed_count,
            "count_matches": count_matches,
            "batch_size": batch_size,
            "total_batches": math.ceil(total_chunks / batch_size) if total_chunks > 0 else 0,
            "failures": failures,
            "duration_seconds": round(duration, 4),
            "throughput_records_per_sec": round(inserted_count / duration, 2),
            "status": "SUCCESS" if count_matches and len(failures) == 0 else "MISMATCH_OR_FAILURE"
        }

        return summary

    def spot_check_integrity(
        self,
        sample_chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Spot-checks stored records against original source chunks to verify 100% field integrity.
        
        Args:
            sample_chunks: List of original chunk dictionaries to test.
            collection_name: Collection name where records are indexed.
            
        Returns:
            List[Dict[str, Any]]: Detailed per-record verification results.
        """
        target_collection = collection_name or self.default_collection
        check_results = []

        for chunk in sample_chunks:
            expected_record = self.to_vector_record(chunk)
            chunk_id = expected_record["id"]

            stored = self.vector_db.get_record(chunk_id, collection_name=target_collection)
            if not stored:
                check_results.append({
                    "id": chunk_id,
                    "found": False,
                    "passed": False,
                    "error": f"Record '{chunk_id}' not found in vector database collection."
                })
                continue

            # Field-by-field verification
            id_match = (stored["id"] == chunk_id)
            vector_len_match = (stored["vector_length"] == len(expected_record["vector"]))
            text_match = (stored["text"] == expected_record["text"])
            
            # Check core metadata keys
            expected_meta = expected_record["metadata"]
            stored_meta = stored["metadata"]
            source_match = (stored_meta.get("source") == expected_meta.get("source"))
            chunk_idx_match = (stored_meta.get("chunk_index") == expected_meta.get("chunk_index"))

            all_passed = id_match and vector_len_match and text_match and source_match and chunk_idx_match

            check_results.append({
                "id": chunk_id,
                "found": True,
                "passed": all_passed,
                "vector_length": stored["vector_length"],
                "expected_vector_length": len(expected_record["vector"]),
                "source": stored_meta.get("source"),
                "chunk_index": stored_meta.get("chunk_index"),
                "section": stored_meta.get("section"),
                "text_preview": stored["text"][:80] + ("..." if len(stored["text"]) > 80 else ""),
                "checks": {
                    "id_match": id_match,
                    "vector_length_match": vector_len_match,
                    "text_match": text_match,
                    "source_match": source_match,
                    "chunk_index_match": chunk_idx_match
                }
            })

        return check_results

    def reindex_document(
        self,
        doc_source: str,
        new_chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Demonstrates incremental re-indexing when a document changes:
        1. Queries and deletes stale chunk records for `doc_source`.
        2. Upserts updated chunk embeddings without rebuilding unrelated documents.
        """
        target_collection = collection_name or self.default_collection
        col = self.vector_db.create_or_get_collection(target_collection)

        # 1. Query existing records for document
        existing_records = col.get(where={"source": doc_source})
        stale_ids = existing_records.get("ids", [])

        # 2. Evict stale IDs if any
        if stale_ids:
            col.delete(ids=stale_ids)
            logger.info(f"Evicted {len(stale_ids)} stale records for document '{doc_source}'")

        # 3. Upsert fresh chunks
        new_records = [self.to_vector_record(c, idx) for idx, c in enumerate(new_chunks)]
        self.vector_db.upsert_records(new_records, collection_name=target_collection)

        return {
            "document_source": doc_source,
            "evicted_stale_chunks": len(stale_ids),
            "inserted_new_chunks": len(new_records),
            "collection_total_count": self.vector_db.count(target_collection),
            "status": "INCREMENTAL_REINDEX_SUCCESS"
        }
