"""
src/rag_pipeline.py

Backward-compatible re-export from src.services.rag_pipeline.
"""

from src.services.rag_pipeline import (
    RAGPipeline,
    embed_query,
    retrieve_context,
    assemble_context,
    generate_answer,
    answer_query,
    conversational_answer,
    EMPTY_RETRIEVAL_FALLBACK_MESSAGE
)

__all__ = [
    "RAGPipeline",
    "embed_query",
    "retrieve_context",
    "assemble_context",
    "generate_answer",
    "answer_query",
    "conversational_answer",
    "EMPTY_RETRIEVAL_FALLBACK_MESSAGE"
]
