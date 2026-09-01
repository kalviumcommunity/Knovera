"""
test_embedding_generator.py

Unit test suite for Assignment 3.25 Embeddings Fundamentals & Vector Representation.
Verifies vector dimension uniformity, cosine similarity mathematical accuracy,
semantic similarity ranking assertions, and edge case handling.
"""

import unittest
import numpy as np
from src.embedding_generator import EmbeddingGenerator, cosine_similarity


class TestEmbeddingGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = EmbeddingGenerator()
        self.texts = [
            "How do I reset my account password?",
            "Steps to recover access to my login",
            "The cafeteria menu has pasta today",
            "How can I change my user password?",
            "What options are available for lunch in the canteen?"
        ]

    def test_dimension_uniformity(self):
        """Test that generated embeddings all have the exact same vector dimension."""
        embeddings = self.generator.embed(self.texts)
        self.assertEqual(len(embeddings), 5)
        
        dim = self.generator.get_dimension(embeddings)
        self.assertEqual(dim, 1536)
        
        is_uniform, exp_dim, lengths = self.generator.verify_dimensions(embeddings)
        self.assertTrue(is_uniform)
        self.assertEqual(exp_dim, 1536)
        self.assertEqual(lengths, [1536] * 5)

    def test_cosine_similarity_identity(self):
        """Test that cosine similarity of a vector with itself is 1.0."""
        vec = np.random.randn(1536)
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        """Test that orthogonal vectors yield 0.0 cosine similarity."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        """Test that opposite vectors yield -1.0 cosine similarity."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_semantic_ranking(self):
        """Test that semantically similar texts score significantly higher than dissimilar texts."""
        embeddings = self.generator.embed(self.texts)
        
        sim_auth = cosine_similarity(embeddings[0], embeddings[1])
        dissim_cross = cosine_similarity(embeddings[0], embeddings[2])
        sim_dining = cosine_similarity(embeddings[2], embeddings[4])
        
        # Assertions
        self.assertGreater(sim_auth, dissim_cross)
        self.assertGreater(sim_dining, dissim_cross)
        self.assertGreater(sim_auth, 0.70)
        self.assertLess(dissim_cross, 0.30)

    def test_empty_and_zero_inputs(self):
        """Test graceful handling of empty inputs and zero vectors."""
        self.assertEqual(self.generator.embed([]), [])
        
        zero_vec = [0.0] * 1536
        valid_vec = [1.0] + [0.0] * 1535
        self.assertEqual(cosine_similarity(zero_vec, valid_vec), 0.0)


if __name__ == "__main__":
    unittest.main()
