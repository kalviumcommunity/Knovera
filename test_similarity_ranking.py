"""
test_similarity_ranking.py

Unit test suite for Assignment 3.27 Embedding Similarity & Distance Metrics.
Verifies metric formulas (Cosine Similarity, Cosine Distance, Dot Product, Euclidean Distance, Manhattan Distance),
identities/orthogonality, chunk ranking sorting order, top/bottom chunk metadata retrieval, cross-metric consistency,
and edge cases.
"""

import unittest
import numpy as np
from src.embedding_generator import (
    EmbeddingGenerator,
    cosine_similarity,
    cosine_distance,
    dot_product,
    euclidean_distance,
    manhattan_distance,
    calculate_metric,
    rank_chunks
)


class TestSimilarityRanking(unittest.TestCase):

    def setUp(self):
        self.generator = EmbeddingGenerator()
        self.raw_chunks = [
            {
                "text": "Password reset instructions for learner accounts.",
                "metadata": {"source": "account-guide.md", "chunk_index": 0, "category": "Authentication"}
            },
            {
                "text": "Learners can recover access using their registered email.",
                "metadata": {"source": "account-guide.md", "chunk_index": 1, "category": "Authentication"}
            },
            {
                "text": "The cafeteria menu changes every Friday.",
                "metadata": {"source": "campus-guide.md", "chunk_index": 3, "category": "Campus Life"}
            }
        ]
        self.records = self.generator.embed_chunks(self.raw_chunks)
        self.query = "How can a learner reset their password?"

    def test_cosine_similarity_identities(self):
        """Test Cosine Similarity identities: self = 1.0, orthogonal = 0.0, opposite = -1.0."""
        vec = np.random.randn(1536)
        vec_norm = vec / np.linalg.norm(vec)
        
        # Self similarity
        self.assertAlmostEqual(cosine_similarity(vec_norm, vec_norm), 1.0, places=5)
        
        # Orthogonal
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)
        
        # Opposite
        v_opp = -vec_norm
        self.assertAlmostEqual(cosine_similarity(vec_norm, v_opp), -1.0, places=5)

    def test_distance_metrics(self):
        """Test Cosine Distance, Dot Product, Euclidean, and Manhattan formulas."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        
        # Cosine Distance = 1.0 - 0.0 = 1.0
        self.assertAlmostEqual(cosine_distance(v1, v2), 1.0, places=5)
        
        # Dot Product = 0.0
        self.assertAlmostEqual(dot_product(v1, v2), 0.0, places=5)
        
        # Euclidean Distance = sqrt((1-0)^2 + (0-1)^2) = sqrt(2) ~ 1.41421
        self.assertAlmostEqual(euclidean_distance(v1, v2), np.sqrt(2), places=4)
        
        # Manhattan Distance = |1-0| + |0-1| = 2.0
        self.assertAlmostEqual(manhattan_distance(v1, v2), 2.0, places=5)

    def test_calculate_metric_dispatch(self):
        """Test calculate_metric router function across all supported metrics."""
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        self.assertEqual(calculate_metric(v1, v2, "cosine"), 1.0)
        self.assertEqual(calculate_metric(v1, v2, "cosine_distance"), 0.0)
        self.assertEqual(calculate_metric(v1, v2, "dot_product"), 1.0)
        self.assertEqual(calculate_metric(v1, v2, "euclidean"), 0.0)
        self.assertEqual(calculate_metric(v1, v2, "manhattan"), 0.0)

    def test_rank_chunks_query_similarity(self):
        """Test ranking query against chunk corpus puts auth chunk first and dining last."""
        result = self.generator.rank_chunks(query=self.query, stored_records=self.records, metric="cosine")
        
        ranked = result["ranked_chunks"]
        self.assertEqual(len(ranked), 3)
        
        most_sim = result["most_similar"]
        least_sim = result["least_similar"]
        
        # Top result must be password reset / account guide chunk
        self.assertEqual(most_sim["rank"], 1)
        self.assertEqual(most_sim["metadata"]["source"], "account-guide.md")
        self.assertIn("Password reset", most_sim["text"])
        
        # Least similar result must be cafeteria menu / campus guide chunk
        self.assertEqual(least_sim["rank"], 3)
        self.assertEqual(least_sim["metadata"]["source"], "campus-guide.md")
        self.assertIn("cafeteria menu", least_sim["text"])
        
        # Assert score comparison
        self.assertGreater(most_sim["score"], least_sim["score"])
        self.assertGreater(most_sim["score"], 0.70)

    def test_rank_chunks_distance_metric_sorting(self):
        """Test that distance metrics sort in ascending order (lower score = rank #1)."""
        result = self.generator.rank_chunks(query=self.query, stored_records=self.records, metric="cosine_distance")
        
        most_sim = result["most_similar"]
        least_sim = result["least_similar"]
        
        # In distance metrics, top result has lowest distance score
        self.assertEqual(most_sim["rank"], 1)
        self.assertEqual(most_sim["metadata"]["source"], "account-guide.md")
        self.assertLess(most_sim["score"], least_sim["score"])

    def test_cross_metric_rank_order_equivalency(self):
        """Test that Cosine Similarity, Dot Product, and Cosine Distance yield identical rank ordering."""
        res_cos = self.generator.rank_chunks(query=self.query, stored_records=self.records, metric="cosine")
        res_dot = self.generator.rank_chunks(query=self.query, stored_records=self.records, metric="dot_product")
        res_dist = self.generator.rank_chunks(query=self.query, stored_records=self.records, metric="cosine_distance")
        
        order_cos = [item["metadata"]["source"] for item in res_cos["ranked_chunks"]]
        order_dot = [item["metadata"]["source"] for item in res_dot["ranked_chunks"]]
        order_dist = [item["metadata"]["source"] for item in res_dist["ranked_chunks"]]
        
        self.assertEqual(order_cos, order_dot)
        self.assertEqual(order_cos, order_dist)

    def test_top_k_limiting(self):
        """Test top_k parameter restricts the number of returned ranked records."""
        result = self.generator.rank_chunks(query=self.query, stored_records=self.records, top_k=2)
        self.assertEqual(len(result["ranked_chunks"]), 2)
        self.assertEqual(result["ranked_chunks"][0]["rank"], 1)
        self.assertEqual(result["ranked_chunks"][1]["rank"], 2)

    def test_edge_cases_and_graceful_handling(self):
        """Test empty record list, zero vectors, and invalid metrics."""
        empty_res = self.generator.rank_chunks(query=self.query, stored_records=[])
        self.assertEqual(empty_res["ranked_chunks"], [])
        self.assertIsNone(empty_res["most_similar"])
        self.assertIsNone(empty_res["least_similar"])

        # Zero vector handling
        zero_rec = [{
            "id": "zero_chunk",
            "text": "empty",
            "metadata": {},
            "embedding": [0.0] * 1536
        }]
        res_zero = self.generator.rank_chunks(query=self.query, stored_records=zero_rec)
        self.assertEqual(res_zero["ranked_chunks"][0]["score"], 0.0)

        # Unsupported metric raises ValueError
        with self.assertRaises(ValueError):
            calculate_metric([1.0], [1.0], "unsupported_metric_xyz")


if __name__ == "__main__":
    unittest.main()
