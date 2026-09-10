"""
src/services/__init__.py

Core RAG business logic and pipeline services for Knovera.
"""

from src.services.text_cleaner import clean_text, normalize_whitespace
from src.services.document_loader import load_document
from src.services.token_chunker import chunk_text_by_tokens, estimate_tokens
from src.services.chunk_tagger import tag_chunk_metadata
from src.services.embedding_generator import EmbeddingGenerator
from src.services.embedding_quality_checker import check_embedding_quality
from src.services.vector_store import VectorDatabase
from src.services.document_indexer import store_upload, process_uploaded_document
from src.services.top_k_retriever import TopKRetriever
from src.services.hybrid_retriever import HybridRetriever
from src.services.reranker import LLMReranker
from src.services.context_injector import assemble_augmented_context
from src.services.hallucination_guardrail import evaluate_retrieval_quality, STANDARD_SAFE_REFUSAL
from src.services.grounded_generator import GroundedAnswerGenerator
from src.services.source_tracer import extract_citations, verify_citation_grounding
from src.services.history_manager import HistoryManager
from src.services.rag_pipeline import RAGPipeline, embed_query, retrieve_context, assemble_context, generate_answer
from src.services.rag_evaluator import RAGEvaluator

__all__ = [
    "clean_text",
    "normalize_whitespace",
    "load_document",
    "chunk_text_by_tokens",
    "estimate_tokens",
    "tag_chunk_metadata",
    "EmbeddingGenerator",
    "check_embedding_quality",
    "VectorDatabase",
    "store_upload",
    "process_uploaded_document",
    "TopKRetriever",
    "HybridRetriever",
    "LLMReranker",
    "assemble_augmented_context",
    "evaluate_retrieval_quality",
    "STANDARD_SAFE_REFUSAL",
    "GroundedAnswerGenerator",
    "extract_citations",
    "verify_citation_grounding",
    "HistoryManager",
    "RAGPipeline",
    "embed_query",
    "retrieve_context",
    "assemble_context",
    "generate_answer",
    "RAGEvaluator",
]
