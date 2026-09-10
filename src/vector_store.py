"""
src/vector_store.py

Backward-compatible re-export of VectorDatabase from src.services.vector_store.
"""

from src.services.vector_store import (
    VectorDatabase,
    STORED_RECORD_SCHEMA
)

__all__ = [
    "VectorDatabase",
    "STORED_RECORD_SCHEMA"
]
