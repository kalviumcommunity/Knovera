"""
src/chunk_tagger.py

Backward-compatible re-export of ChunkTagger from src.services.chunk_tagger.
"""

from src.services.chunk_tagger import (
    ChunkTagger,
    tag_chunk_metadata,
    extract_sections_and_metadata
)

__all__ = [
    "ChunkTagger",
    "tag_chunk_metadata",
    "extract_sections_and_metadata"
]
