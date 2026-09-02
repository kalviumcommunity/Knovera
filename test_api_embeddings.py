"""
test_api_embeddings.py

Unit test suite for Assignment 3.26: Generating Embeddings via API.
Tests:
- Environment configuration resolution
- Batch chunk embedding and structure of stored records (text + metadata + vector)
- Vector dimension consistency and uniformity
- Metadata integrity across multiple source documents
- Semantic similarity search using identical model embeddings
"""

import os
import unittest
from src.embedding_generator import EmbeddingGenerator, cosine_similarity


class TestApiEmbeddingGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = EmbeddingGenerator()
        self.sample_chunks = [
            {
                "text": "Password reset instructions for learner accounts.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Reset"}
            },
            {
                "text": "Learners can recover access using their registered email.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Recovery"}
            },
            {
                "text": "Campus cafeteria pasta and soup menu for lunch.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Dining"}
            }
        ]

    def test_environment_configuration_defaults(self):
        """Verify that model name, base URL, and dimensions resolve properly."""
        gen = EmbeddingGenerator(model_name="text-embedding-3-small")
        self.assertEqual(gen.model_name, "text-embedding-3-small")
        self.assertEqual(gen.dimension, 1536)

    def test_embed_chunks_output_structure(self):
        """Verify each stored record contains id, text, metadata, embedding, and vector_length."""
        records = self.generator.embed_chunks(self.sample_chunks, batch_size=2)
        
        self.assertEqual(len(records), 3)
        for i, record in enumerate(records):
            self.assertIn("id", record)
            self.assertIn("text", record)
            self.assertIn("metadata", record)
            self.assertIn("embedding", record)
            self.assertIn("vector_length", record)
            self.assertIn("model", record)
            
            # Verify text matching
            self.assertEqual(record["text"], self.sample_chunks[i]["text"])
            
            # Verify metadata preserved
            self.assertEqual(record["metadata"]["source"], self.sample_chunks[i]["metadata"]["source"])
            self.assertEqual(record["metadata"]["chunk_index"], self.sample_chunks[i]["metadata"]["chunk_index"])

    def test_vector_dimension_and_uniformity(self):
        """Verify that all generated vectors have uniform length matching target dimension."""
        records = self.generator.embed_chunks(self.sample_chunks)
        embeddings = [r["embedding"] for r in records]
        
        is_uniform, exp_dim, lengths = self.generator.verify_dimensions(embeddings)
        self.assertTrue(is_uniform)
        self.assertEqual(exp_dim, 1536)
        self.assertEqual(lengths, [1536, 1536, 1536])
        
        # Verify vector values are floats
        self.assertIsInstance(records[0]["embedding"][0], float)

    def test_empty_chunks_handling(self):
        """Verify that passing empty chunk list returns empty list."""
        records = self.generator.embed_chunks([])
        self.assertEqual(records, [])

    def test_semantic_retrieval_with_same_model(self):
        """Verify that querying stored records with identical model retrieves relevant chunk."""
        records = self.generator.embed_chunks(self.sample_chunks)
        
        query = "How do I recover my login password?"
        results = self.generator.search_similar(query, records, top_k=2)
        
        self.assertEqual(len(results), 2)
        # Top result should be from account-guide.md (authentication domain)
        top_result = results[0]
        self.assertEqual(top_result["metadata"]["source"], "account-guide.md")
        self.assertGreater(top_result["similarity"], 0.70)


if __name__ == "__main__":
    unittest.main()
