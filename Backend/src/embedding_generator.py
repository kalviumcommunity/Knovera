"""
src/embedding_generator.py

Backward-compatible re-export from src.services.embedding_generator.
"""

from src.services.embedding_generator import (
    EmbeddingGenerator,
    cosine_similarity,
    cosine_distance,
    dot_product,
    euclidean_distance,
    manhattan_distance,
    calculate_metric,
    rank_chunks
)

__all__ = [
    "EmbeddingGenerator",
    "cosine_similarity",
    "cosine_distance",
    "dot_product",
    "euclidean_distance",
    "manhattan_distance",
    "calculate_metric",
    "rank_chunks"
]
