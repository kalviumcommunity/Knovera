"""
src/token_chunker.py

Backward-compatible re-export of TokenChunker from src.services.token_chunker.
"""

from src.services.token_chunker import (
    TokenChunker,
    estimate_tokens,
    chunk_text_by_tokens
)

__all__ = [
    "TokenChunker",
    "estimate_tokens",
    "chunk_text_by_tokens"
]
