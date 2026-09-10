"""
src/hybrid_retriever.py

Backward-compatible re-export from src.services.hybrid_retriever.
"""

from src.services.hybrid_retriever import (
    HybridRetriever,
    extract_search_tokens,
    keyword_score,
    hybrid_rank
)

__all__ = [
    "HybridRetriever",
    "extract_search_tokens",
    "keyword_score",
    "hybrid_rank"
]
