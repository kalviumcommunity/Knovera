"""
src/embedding_quality_checker.py

Backward-compatible re-export of EmbeddingQualityChecker from src.services.embedding_quality_checker.
"""

from src.services.embedding_quality_checker import (
    EmbeddingQualityChecker,
    check_embedding_quality
)

__all__ = [
    "EmbeddingQualityChecker",
    "check_embedding_quality"
]
