"""
test_corpus_indexer.py

Comprehensive Unit Test Suite for Assignment 3.31: Indexing Embeddings & Metadata Storage.
Tests:
- Vector record transformation schema
- Batched bulk corpus indexing
- Count validation matching chunk count
- Spot-check integrity audits (ID, vector length, text, metadata parity)
- Metadata filtering over indexed records
- Incremental document re-indexing (stale eviction and fresh upsert)
"""

import os
import unittest
import shutil
from typing import List, Dict, Any

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator


class TestCorpusIndexer(unittest.TestCase):

    def setUp(self):
        self.test_persist_dir = os.path.join(".", "data", "test_indexer_chroma_db")
        self.vector_db = VectorDatabase(
            persist_dir=self.test_persist_dir,
            in_memory=True
        )
        self.collection_name = "test_index_collection"
        self.indexer = CorpusIndexer(
            vector_db=self.vector_db,
            default_collection=self.collection_name
        )

        self.sample_chunks = [
            {
                "id": "doc1.md:0",
                "text": "How to reset user passwords.",
                "embedding": [0.01] * 1536,
                "metadata": {"source": "doc1.md", "chunk_index": 0, "section": "Reset", "category": "Auth"}
            },
            {
                "id": "doc1.md:1",
                "text": "Email recovery verification link details.",
                "embedding": [0.02] * 1536,
                "metadata": {"source": "doc1.md", "chunk_index": 1, "section": "Recovery", "category": "Auth"}
            },
            {
                "id": "doc2.md:0",
                "text": "Campus cafeteria lunch hours.",
                "embedding": [0.03] * 1536,
                "metadata": {"source": "doc2.md", "chunk_index": 0, "section": "Dining", "category": "Campus"}
            }
        ]

    def tearDown(self):
        try:
            self.vector_db.delete_collection(self.collection_name)
        except Exception:
            pass
        if os.path.exists(self.test_persist_dir):
            try:
                shutil.rmtree(self.test_persist_dir)
            except Exception:
                pass

    def test_to_vector_record_transformation(self):
        """Verify transforming raw chunks into canonical vector records."""
        chunk = self.sample_chunks[0]
        rec = self.indexer.to_vector_record(chunk)

        self.assertEqual(rec["id"], "doc1.md:0")
        self.assertEqual(len(rec["vector"]), 1536)
        self.assertEqual(rec["text"], chunk["text"])
        self.assertEqual(rec["metadata"]["source"], "doc1.md")
        self.assertEqual(rec["metadata"]["chunk_index"], 0)
        self.assertEqual(rec["metadata"]["category"], "Auth")

    def test_index_corpus_and_count_verification(self):
        """Verify bulk indexing inserts all records and indexed count matches input count."""
        summary = self.indexer.index_corpus(
            embedded_chunks=self.sample_chunks,
            collection_name=self.collection_name,
            batch_size=2,
            reset_collection=True
        )

        self.assertEqual(summary["status"], "SUCCESS")
        self.assertEqual(summary["expected_chunks"], 3)
        self.assertEqual(summary["inserted_this_run"], 3)
        self.assertEqual(summary["final_indexed_count"], 3)
        self.assertTrue(summary["count_matches"])
        self.assertEqual(len(summary["failures"]), 0)

    def test_spot_check_integrity_passes_on_valid_records(self):
        """Verify spot check audits confirm 100% field parity."""
        self.indexer.index_corpus(
            embedded_chunks=self.sample_chunks,
            collection_name=self.collection_name,
            reset_collection=True
        )

        spot_results = self.indexer.spot_check_integrity(
            sample_chunks=self.sample_chunks,
            collection_name=self.collection_name
        )

        self.assertEqual(len(spot_results), 3)
        for sc in spot_results:
            self.assertTrue(sc["found"])
            self.assertTrue(sc["passed"])
            self.assertTrue(sc["checks"]["id_match"])
            self.assertTrue(sc["checks"]["vector_length_match"])
            self.assertTrue(sc["checks"]["text_match"])
            self.assertTrue(sc["checks"]["source_match"])
            self.assertTrue(sc["checks"]["chunk_index_match"])

    def test_metadata_filtered_search(self):
        """Verify indexed records can be queried with metadata filters."""
        self.indexer.index_corpus(
            embedded_chunks=self.sample_chunks,
            collection_name=self.collection_name,
            reset_collection=True
        )

        # Filter by category='Campus'
        results = self.vector_db.query_similar(
            query_vector=[0.03] * 1536,
            top_k=5,
            where={"category": "Campus"},
            collection_name=self.collection_name
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc2.md:0")
        self.assertEqual(results[0]["metadata"]["category"], "Campus")

    def test_incremental_document_reindexing(self):
        """Verify updating a document replaces only its chunks without touching others."""
        self.indexer.index_corpus(
            embedded_chunks=self.sample_chunks,
            collection_name=self.collection_name,
            reset_collection=True
        )
        self.assertEqual(self.vector_db.count(self.collection_name), 3)

        # Update doc1.md with 1 new chunk (replacing 2 previous doc1.md chunks)
        new_doc1_chunks = [
            {
                "id": "doc1.md:0",
                "text": "Updated unified password and recovery instructions.",
                "embedding": [0.015] * 1536,
                "metadata": {"source": "doc1.md", "chunk_index": 0, "section": "Unified Auth", "category": "Auth"}
            }
        ]

        reindex_res = self.indexer.reindex_document(
            doc_source="doc1.md",
            new_chunks=new_doc1_chunks,
            collection_name=self.collection_name
        )

        self.assertEqual(reindex_res["evicted_stale_chunks"], 2)
        self.assertEqual(reindex_res["inserted_new_chunks"], 1)
        self.assertEqual(reindex_res["collection_total_count"], 2) # 1 new doc1 + 1 doc2 = 2 total
        self.assertEqual(reindex_res["status"], "INCREMENTAL_REINDEX_SUCCESS")


if __name__ == "__main__":
    unittest.main()
