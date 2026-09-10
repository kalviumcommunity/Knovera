"""
src/document_loader.py

Backward-compatible re-export of DocumentLoader from src.services.document_loader.
"""

from src.services.document_loader import (
    DocumentLoader,
    load_document,
    extract_metadata_from_filename
)

__all__ = [
    "DocumentLoader",
    "load_document",
    "extract_metadata_from_filename"
]
