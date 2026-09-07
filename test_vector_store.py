"""
test_vector_store.py

Comprehensive Unit Test Suite for Assignment 3.30: Vector Database Setup & Collection Design.
Tests:
- Vector store connectivity and reachability
- Correct vector dimension enforcement (1536)
- Record schema validation and field integrity
- Test record insertion and readback verification
- Batch upsert and metadata filtering
- Dimension mismatch rejection
"""

import os
import unittest
import shutil
from typing import List, Dict, Any

from src.vector_store import VectorDatabase, STORED_RECORD_SCHEMA
from src.embedding_generator import EmbeddingGenerator


class TestVectorDatabaseSetup(unittest.TestCase):

    def setUp(self):
        self.test_persist_dir = os.path.join(".", "data", "test_chroma_db")
        self.vector_db = VectorDatabase(
            persist_dir=self.test_persist_dir,
            in_memory=True
        )
        self.collection_name = "test_rag_chunks"
        self.vector_db.create_or_get_collection(
            name=self.collection_name,
            dimension=1536,
            metric="cosine"
        )

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

    def test_database_reachability(self):
        """Verify that ChromaDB vector database is reachable and active."""
        self.assertTrue(self.vector_db.is_reachable())

    def test_collection_creation_and_dimension(self):
        """Verify collection exists with configured dimension and metadata."""
        col = self.vector_db.active_collection
        self.assertIsNotNone(col)
        self.assertEqual(col.name, self.collection_name)
        self.assertEqual(col.metadata["dimension"], 1536)
        self.assertEqual(col.metadata["hnsw:space"], "cosine")

    def test_insert_and_readback_record(self):
        """Verify inserting one test record and reading it back preserves all fields."""
        test_id = "account-guide.md:0"
        test_vector = [0.01] * 1536
        test_text = "Password reset instructions for learner accounts."
        test_meta = {
            "source": "account-guide.md",
            "chunk_index": 0,
            "section": "Account access",
            "category": "Authentication"
        }

        # Insert record
        success = self.vector_db.insert_record(
            id=test_id,
            vector=test_vector,
            text=test_text,
            metadata=test_meta,
            collection_name=self.collection_name
        )
        self.assertTrue(success)
        self.assertEqual(self.vector_db.count(self.collection_name), 1)

        # Read back record
        stored = self.vector_db.get_record(test_id, collection_name=self.collection_name)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["id"], test_id)
        self.assertEqual(stored["vector_length"], 1536)
        self.assertEqual(stored["text"], test_text)
        self.assertEqual(stored["metadata"]["source"], "account-guide.md")
        self.assertEqual(stored["metadata"]["chunk_index"], 0)
        self.assertEqual(stored["metadata"]["category"], "Authentication")

    def test_schema_validation_rejects_dimension_mismatch(self):
        """Verify that records with wrong vector dimension fail validation early."""
        bad_record = {
            "id": "bad_chunk:0",
            "vector": [0.1, 0.2, 0.3], # 3 dimensions instead of 1536
            "text": "Malformed vector length.",
            "metadata": {"source": "bad.md"}
        }
        is_valid, err = self.vector_db.validate_record_schema(bad_record)
        self.assertFalse(is_valid)
        self.assertIn("dimension mismatch", err)

        with self.assertRaises(ValueError):
            self.vector_db.upsert_records([bad_record], collection_name=self.collection_name)

    def test_schema_validation_rejects_missing_fields(self):
        """Verify that records missing id or text fail validation."""
        missing_id = {"vector": [0.0] * 1536, "text": "No ID"}
        is_valid, _ = self.vector_db.validate_record_schema(missing_id)
        self.assertFalse(is_valid)

        missing_text = {"id": "chunk:1", "vector": [0.0] * 1536}
        is_valid, _ = self.vector_db.validate_record_schema(missing_text)
        self.assertFalse(is_valid)

    def test_vector_similarity_query(self):
        """Verify nearest-neighbor vector query retrieves correct matching record."""
        vec1 = [0.1] * 1536
        vec2 = [-0.1] * 1536

        records = [
            {"id": "doc1:0", "vector": vec1, "text": "Auth policy", "metadata": {"source": "auth.md"}},
            {"id": "doc2:0", "vector": vec2, "text": "Dining menu", "metadata": {"source": "dining.md"}}
        ]
        self.vector_db.upsert_records(records, collection_name=self.collection_name)
        self.assertEqual(self.vector_db.count(self.collection_name), 2)

        # Query close to vec1
        results = self.vector_db.query_similar(vec1, top_k=1, collection_name=self.collection_name)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc1:0")
        self.assertGreater(results[0]["similarity"], 0.99)


if __name__ == "__main__":
    unittest.main()
