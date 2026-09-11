"""
src/rag_pipeline.py

RAG Pipeline Architecture & Flow Design Engine.
Implements the end-to-end Retrieval-Augmented Generation pipeline:
User Query -> Embed Query -> Retrieve Context (Top-K) -> Assemble Context -> Generate Grounded Answer -> Return Answer + Sources

Key Architectural Principles:
1. Clear Stage Separation: embed, retrieve, assemble, and generate are isolated into distinct, testable functions.
2. Grounded Prompt Assembly: Assembles retrieved chunks with deterministic indexed citation markers.
3. Strict Non-Hallucination Guard: Gracefully handles empty retrieval states without fabricating information.
4. Flexible Generation: Integrates OpenAI/OpenRouter API completions with robust deterministic offline fallback.
5. End-to-End Orchestrator: Coordinates pipeline execution, tracking per-stage telemetry, scores, and source metadata.
"""

import os
import re
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.services.vector_store import VectorDatabase
from src.services.embedding_generator import EmbeddingGenerator
from src.services.history_manager import HistoryManager, rewrite_followup
from templates.prompts import QA_PROMPT_TEMPLATE, render_prompt

# Try importing OpenAI client
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

# Fallback response message when retrieval yields no evidence
EMPTY_RETRIEVAL_FALLBACK_MESSAGE = "I could not find relevant context for that question."


# ============================================================================
# STAGE 1: QUERY EMBEDDING
# ============================================================================

def embed_query(
    query: str,
    generator: Optional[EmbeddingGenerator] = None
) -> List[float]:
    """
    Stage 1: Embeds a user query string into a dense numerical vector representation
    using the model-aligned EmbeddingGenerator.
    
    Args:
        query: Raw user natural language query string.
        generator: Optional EmbeddingGenerator instance.
        
    Returns:
        List[float]: Normalized embedding vector matching the collection dimension.
        
    Raises:
        ValueError: If query is empty or whitespace only.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace only.")
        
    active_gen = generator or EmbeddingGenerator()
    embeddings = active_gen.embed([query.strip()])
    
    if not embeddings or len(embeddings) == 0:
        raise RuntimeError(f"Embedding generation failed for query: '{query}'")
        
    return embeddings[0]


# ============================================================================
# STAGE 2: CONTEXT RETRIEVAL
# ============================================================================

def retrieve_context(
    query_vector: List[float],
    vector_db: Optional[VectorDatabase] = None,
    collection_name: str = "knovera_knowledge_base",
    k: int = 4,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Stage 2: Queries the Vector Database to retrieve top-k nearest neighbor chunks
    based on vector similarity.
    
    Args:
        query_vector: Dense numerical vector of the user query.
        vector_db: Optional VectorDatabase instance.
        collection_name: Name of the vector database collection to query.
        k: Maximum number of top relevant chunks to retrieve.
        score_threshold: Optional minimum cosine similarity score cutoff [0.0 - 1.0].
        metadata_filter: Optional dictionary of exact-match metadata filters.
        
    Returns:
        List[Dict[str, Any]]: Ranked list of retrieved chunk dictionaries containing
                              'id', 'text', 'metadata', 'similarity' (or 'score'), and 'distance'.
    """
    if not query_vector:
        return []
        
    if k <= 0:
        raise ValueError(f"k must be a positive integer > 0, got {k}")

    active_db = vector_db or VectorDatabase()
    
    # Query vector store
    raw_results = active_db.query_similar(
        query_vector=query_vector,
        top_k=k,
        where=metadata_filter,
        collection_name=collection_name
    )
    
    formatted_chunks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_results, start=1):
        similarity = float(item.get("similarity", 0.0))
        
        # Apply score threshold if specified
        if score_threshold is not None and similarity < score_threshold:
            continue
            
        metadata = dict(item.get("metadata", {}))
        source = metadata.get("source", metadata.get("doc_title", "Unknown Source"))
        
        chunk_dict = {
            "rank": idx,
            "id": item.get("id", f"chunk_{idx}"),
            "text": item.get("text", ""),
            "score": round(similarity, 4),
            "similarity": round(similarity, 4),
            "distance": round(float(item.get("distance", 0.0)), 4),
            "metadata": metadata,
            "source": source
        }
        formatted_chunks.append(chunk_dict)
        
    return formatted_chunks


# ============================================================================
# STAGE 3: CONTEXT ASSEMBLY
# ============================================================================

def assemble_context(
    chunks: List[Dict[str, Any]],
    include_scores: bool = False
) -> str:
    """
    Stage 3: Assembles retrieved document chunks into a structured, citation-indexed
    context block for downstream LLM prompt injection.
    
    Format:
        [1] Source: account-guide.md
        Password reset instructions...
        
        [2] Source: campus-guide.md
        Cafeteria operating hours...
        
    Args:
        chunks: List of retrieved chunk dictionaries.
        include_scores: If True, attaches similarity scores to context headers.
        
    Returns:
        str: Assembled context block string, or empty string if chunks is empty.
    """
    if not chunks:
        return ""
        
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = chunk.get("source") or meta.get("source") or meta.get("doc_title") or f"Doc_{index}"
        section = meta.get("section") or meta.get("section_heading")
        text = chunk.get("text", "").strip()
        
        header = f"[{index}] Source: {source}"
        if section:
            header += f" | Section: {section}"
        if include_scores and "score" in chunk:
            header += f" | Score: {chunk['score']}"
            
        parts.append(f"{header}\n{text}")
        
    return "\n\n".join(parts)


# ============================================================================
# STAGE 4: GROUNDED ANSWER GENERATION
# ============================================================================

def generate_answer(
    query: str,
    context: str,
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
    prompt_template: str = QA_PROMPT_TEMPLATE,
    use_api: bool = True
) -> str:
    """
    Stage 4: Synthesizes a factual, grounded response based strictly on the
    assembled context and user question.
    
    If OpenAI/OpenRouter API credentials are not active, timeout, or when use_api=False,
    falls back to a deterministic extractive generation engine that extracts key
    factual sentences directly from the assembled context.
    
    Args:
        query: User question string.
        context: Assembled context string containing evidence passages.
        model_name: Model identifier (e.g., 'gpt-3.5-turbo', 'meta-llama/llama-3-8b-instruct').
        client: Optional OpenAI client instance.
        temperature: LLM temperature parameter (default 0.1 for high determinism).
        max_tokens: Maximum completion tokens.
        prompt_template: Prompt template with {context} and {question} placeholders.
        use_api: If False, skips network call and directly uses deterministic grounded synthesis.
        
    Returns:
        str: Grounded answer string.
    """
    if not context or not context.strip():
        return EMPTY_RETRIEVAL_FALLBACK_MESSAGE
        
    # Render prompt safely using template
    prompt = render_prompt(
        prompt_template,
        context=context,
        question=query
    )
    
    if use_api:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
        base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
        target_model = model_name or os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        
        if HAS_OPENAI and api_key and api_key.strip():
            try:
                active_client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=4.0)
                response = active_client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are Knovera, an expert technical assistant. Answer the question using ONLY the provided context. If the context does not contain enough information, explicitly state what is missing. Cite your sources using [1], [2] notation corresponding to the context sources."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                generated_text = response.choices[0].message.content
                if generated_text and generated_text.strip():
                    return generated_text.strip()
            except Exception as e:
                logger.warning(f"Live LLM call skipped or timed out ({e}). Utilizing deterministic offline synthesis.")

    # Deterministic Offline Synthesizer: Extractive grounded response
    return _synthesize_offline_grounded_answer(query=query, context=context)


def _synthesize_offline_grounded_answer(query: str, context: str) -> str:
    """
    Generates an intelligent semantic grounded answer from context passages
    handling synonym equivalence (e.g. describe vs define) and extracting
    actionable policy details.
    """
    from src.services.semantic_engine import synthesize_semantic_grounded_answer
    return synthesize_semantic_grounded_answer(query=query, context=context)


# ============================================================================
# STAGE 5: END-TO-END PIPELINE ORCHESTRATOR
# ============================================================================

def answer_query(
    query: str,
    k: int = 4,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    collection_name: str = "knovera_knowledge_base",
    vector_db: Optional[VectorDatabase] = None,
    generator: Optional[EmbeddingGenerator] = None,
    model_name: Optional[str] = None,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    End-to-End RAG Flow Orchestrator Function.
    Connects the individual stages:
    1. Embed Query
    2. Retrieve Context Chunks
    3. Check for empty retrieval (guard against hallucination)
    4. Assemble Context with Source Citations
    5. Generate Grounded Answer
    6. Return Answer + Source Metadata
    
    Args:
        query: User question string.
        k: Number of chunks to retrieve.
        score_threshold: Optional cosine similarity cutoff.
        metadata_filter: Optional metadata filtering dictionary.
        collection_name: Target vector store collection.
        vector_db: VectorDatabase instance.
        generator: EmbeddingGenerator instance.
        model_name: LLM model name.
        use_api: Whether to call live OpenAI/OpenRouter API or use fast deterministic generation.
        
    Returns:
        Dict[str, Any]: Comprehensive pipeline output dictionary:
            - 'query': str
            - 'answer': str
            - 'sources': List[Dict[str, Any]]
            - 'context': str
            - 'num_retrieved': int
            - 'stage_latencies_ms': Dict[str, float]
            - 'status': str
    """
    start_total = time.perf_counter()
    latencies = {}
    
    if not query or not query.strip():
        return {
            "query": query,
            "answer": "Please provide a valid, non-empty question.",
            "sources": [],
            "context": "",
            "num_retrieved": 0,
            "stage_latencies_ms": {"total": 0.0},
            "status": "EMPTY_QUERY"
        }
        
    # Stage 1: Embed Query
    t0 = time.perf_counter()
    query_vector = embed_query(query, generator=generator)
    latencies["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    
    # Stage 2: Retrieve Context
    t0 = time.perf_counter()
    chunks = retrieve_context(
        query_vector=query_vector,
        vector_db=vector_db,
        collection_name=collection_name,
        k=k,
        score_threshold=score_threshold,
        metadata_filter=metadata_filter
    )
    latencies["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    
    # Stage 3 Guard: Empty Retrieval Handling
    if not chunks:
        latencies["total_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
        return {
            "query": query,
            "answer": EMPTY_RETRIEVAL_FALLBACK_MESSAGE,
            "sources": [],
            "context": "",
            "num_retrieved": 0,
            "stage_latencies_ms": latencies,
            "status": "NO_CONTEXT_FOUND"
        }
        
    # Stage 3: Assemble Context
    t0 = time.perf_counter()
    context = assemble_context(chunks)
    latencies["assemble_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    
    # Stage 4: Generate Answer
    t0 = time.perf_counter()
    answer = generate_answer(
        query=query,
        context=context,
        model_name=model_name,
        use_api=use_api
    )
    latencies["generate_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    
    # Compile sources metadata list
    sources = []
    for chunk in chunks:
        meta = dict(chunk.get("metadata", {}))
        meta["id"] = chunk.get("id")
        meta["score"] = chunk.get("score")
        meta["rank"] = chunk.get("rank")
        sources.append(meta)
        
    latencies["total_ms"] = round((time.perf_counter() - start_total) * 1000, 2)
    
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context": context,
        "num_retrieved": len(chunks),
        "stage_latencies_ms": latencies,
        "status": "SUCCESS"
    }


def conversational_answer(
    history: Union[List[Dict[str, str]], HistoryManager],
    user_question: str,
    k: int = 4,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    collection_name: str = "knovera_knowledge_base",
    vector_db: Optional[VectorDatabase] = None,
    generator: Optional[EmbeddingGenerator] = None,
    model_name: Optional[str] = None,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Executes a multi-turn conversational RAG flow:
    1. Rewrites raw follow-up user_question into a standalone search query using history context.
    2. Embeds the rewritten standalone query.
    3. Retrieves relevant chunks for the standalone query.
    4. Evaluates context availability / strength.
    5. Synthesizes a grounded answer to the original user_question using retrieved context.
    6. Appends turn to conversation history.
    7. Returns complete response metadata including rewritten_query, answer, and sources.
    
    Args:
        history: List of turn dicts or HistoryManager instance.
        user_question: Raw user follow-up question.
        k: Top-k chunks to retrieve.
        score_threshold: Minimum similarity cutoff.
        metadata_filter: Optional filter dict.
        collection_name: Target vector database collection.
        vector_db: VectorDatabase instance.
        generator: EmbeddingGenerator instance.
        model_name: Optional LLM model identifier.
        use_api: Whether to call live OpenAI/OpenRouter API.
        
    Returns:
        Dict[str, Any]: Response dictionary with 'raw_query', 'rewritten_query',
                        'answer', 'sources', 'context', 'num_retrieved', and 'status'.
    """
    # 1. Rewrite follow-up question using conversation history context
    standalone_query = rewrite_followup(
        history=history,
        question=user_question,
        use_api=use_api,
        model_name=model_name
    )
    
    # 2. Embed standalone query
    active_gen = generator or EmbeddingGenerator()
    query_vector = embed_query(standalone_query, generator=active_gen)
    
    # 3. Retrieve context using standalone query vector
    active_db = vector_db or VectorDatabase(embedding_generator=active_gen)
    chunks = retrieve_context(
        query_vector=query_vector,
        vector_db=active_db,
        collection_name=collection_name,
        k=k,
        score_threshold=score_threshold,
        metadata_filter=metadata_filter
    )
    
    # 4. Handle context strength & answer generation
    if not chunks:
        answer = EMPTY_RETRIEVAL_FALLBACK_MESSAGE
        status = "NO_CONTEXT_FOUND"
        context = ""
    else:
        context = assemble_context(chunks)
        answer = generate_answer(
            query=user_question,
            context=context,
            model_name=model_name,
            use_api=use_api
        )
        status = "SUCCESS"
        
    # 5. Append interaction turn to history
    if isinstance(history, HistoryManager):
        history.add_user_message(user_question)
        history.add_assistant_message(answer)
        if history.should_trim():
            history.trim_to_budget()
    elif isinstance(history, list):
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": answer})
        
    # 6. Build sources metadata
    sources = []
    for chunk in chunks:
        meta = dict(chunk.get("metadata", {}))
        meta["id"] = chunk.get("id")
        meta["score"] = chunk.get("score")
        meta["rank"] = chunk.get("rank")
        sources.append(meta)
        
    return {
        "raw_query": user_question,
        "rewritten_query": standalone_query,
        "answer": answer,
        "sources": sources,
        "context": context,
        "num_retrieved": len(chunks),
        "status": status
    }


# ============================================================================
# OBJECT-ORIENTED PIPELINE CLASS
# ============================================================================

class RAGPipeline:
    """
    Object-Oriented RAG Pipeline Orchestrator for Knovera Assistant.
    Maintains persistent connections to EmbeddingGenerator, VectorDatabase, and LLM configuration.
    """
    
    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        generator: Optional[EmbeddingGenerator] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        default_collection: str = "knovera_knowledge_base",
        collection_name: Optional[str] = None,
        default_k: int = 4,
        score_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        use_api: bool = False
    ):
        """
        Initialize the RAG Pipeline.
        """
        load_dotenv()
        self.generator = generator or embedding_generator or EmbeddingGenerator()
        self.vector_db = vector_db or VectorDatabase(embedding_generator=self.generator)
        self.default_collection = collection_name or default_collection
        self.default_k = default_k
        self.score_threshold = score_threshold
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        self.use_api = use_api
        
    def embed(self, query: str) -> List[float]:
        """Embed a query using pipeline generator."""
        return embed_query(query, generator=self.generator)
        
    def retrieve(
        self,
        query_vector: List[float],
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks from vector store."""
        target_k = k if k is not None else self.default_k
        target_thresh = score_threshold if score_threshold is not None else self.score_threshold
        target_collection = collection_name or self.default_collection
        
        return retrieve_context(
            query_vector=query_vector,
            vector_db=self.vector_db,
            collection_name=target_collection,
            k=target_k,
            score_threshold=target_thresh,
            metadata_filter=metadata_filter
        )
        
    def assemble(self, chunks: List[Dict[str, Any]]) -> str:
        """Assemble retrieved chunks into formatted context."""
        return assemble_context(chunks)
        
    def generate(self, query: str, context: str) -> str:
        """Generate answer from query and context."""
        return generate_answer(query=query, context=context, model_name=self.model_name, use_api=self.use_api)
        
    def query(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        use_api: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Runs the full end-to-end RAG pipeline on a user query.
        """
        target_k = k if k is not None else self.default_k
        target_thresh = score_threshold if score_threshold is not None else self.score_threshold
        target_collection = collection_name or self.default_collection
        active_api = use_api if use_api is not None else self.use_api
        
        return answer_query(
            query=query,
            k=target_k,
            score_threshold=target_thresh,
            metadata_filter=metadata_filter,
            collection_name=target_collection,
            vector_db=self.vector_db,
            generator=self.generator,
            model_name=self.model_name,
            use_api=active_api
        )

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for query()."""
        return self.query(*args, **kwargs)

    def guarded_query(
        self,
        query: str,
        k: Optional[int] = None,
        min_top_score: float = 0.72,
        min_supporting_chunks: int = 1,
        min_avg_score: float = 0.60,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the end-to-end RAG pipeline protected by Hallucination Guardrails.
        Retrieves top-k context, evaluates retrieval quality against similarity thresholds,
        and returns safe refusal if context is weak or grounded answer if context is strong.
        """
        from src.hallucination_guardrail import guarded_answer
        
        target_k = k if k is not None else self.default_k
        target_collection = collection_name or self.default_collection
        
        # 1. Embed & Retrieve
        query_vector = self.embed(query)
        chunks = self.retrieve(
            query_vector=query_vector,
            k=target_k,
            metadata_filter=metadata_filter,
            collection_name=target_collection
        )
        
        # 2. Guarded Answer Generation
        return guarded_answer(
            question=query,
            chunks=chunks,
            model_name=self.model_name,
            min_top_score=min_top_score,
            min_supporting_chunks=min_supporting_chunks,
            min_avg_score=min_avg_score,
            use_api=self.use_api
        )

    def conversational_query(
        self,
        user_question: str,
        history: Union[List[Dict[str, str]], HistoryManager],
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a multi-turn conversational query using history tracking and query rewriting.
        """
        target_k = k if k is not None else self.default_k
        target_thresh = score_threshold if score_threshold is not None else self.score_threshold
        target_collection = collection_name or self.default_collection
        
        return conversational_answer(
            history=history,
            user_question=user_question,
            k=target_k,
            score_threshold=target_thresh,
            metadata_filter=metadata_filter,
            collection_name=target_collection,
            vector_db=self.vector_db,
            generator=self.generator,
            model_name=self.model_name,
            use_api=self.use_api
        )


