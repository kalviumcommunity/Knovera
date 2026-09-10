"""
src/api/routes/query.py

Question answering and conversational chat endpoints for Knovera RAG Backend.
"""

import time
import datetime
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends

from src.config import get_config, APIConfig
from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    Source,
    ChatRequest,
    ChatResponse
)
from src.services.embedding_generator import EmbeddingGenerator
from src.services.vector_store import VectorDatabase
from src.services.rag_pipeline import (
    embed_query,
    retrieve_context,
    assemble_context,
    generate_answer,
    EMPTY_RETRIEVAL_FALLBACK_MESSAGE
)
from src.services.hallucination_guardrail import (
    evaluate_retrieval_quality,
    STANDARD_SAFE_REFUSAL
)
from src.services.history_manager import HistoryManager

logger = logging.getLogger("knovera.api.query")
router = APIRouter(tags=["RAG Core"])

# Session history store in-memory
_session_histories: Dict[str, HistoryManager] = {}


class RAGService:
    """Reusable pipeline execution engine."""
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or get_config()
        self._vector_db: Optional[VectorDatabase] = None
        self._embedding_generator: Optional[EmbeddingGenerator] = None

    @property
    def vector_db(self) -> VectorDatabase:
        if self._vector_db is None:
            self._vector_db = VectorDatabase(
                persist_dir=self.config.vector_db_url,
                embedding_generator=self.embedding_generator
            )
        return self._vector_db

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator(
                model_name=self.config.embedding_model,
                api_key=self.config.active_api_key or None,
                base_url=self.config.openrouter_base_url if self.config.openrouter_api_key else self.config.openai_base_url
            )
        return self._embedding_generator

    def execute_query(
        self,
        question: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_api: Optional[bool] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}

        target_k = k if k is not None else self.config.top_k
        active_use_api = use_api if use_api is not None else self.config.use_live_api

        # Stage 1: Embed Query
        t0 = time.perf_counter()
        query_vector = embed_query(question, generator=self.embedding_generator)
        latencies["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 2: Retrieve Chunks
        t0 = time.perf_counter()
        chunks = retrieve_context(
            query_vector=query_vector,
            vector_db=self.vector_db,
            collection_name=self.config.collection_name,
            k=target_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter
        )
        latencies["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3: Guardrail & Retrieval Quality Check
        effective_min_score = score_threshold if score_threshold is not None else self.config.min_top_score
        quality_eval = evaluate_retrieval_quality(
            chunks=chunks,
            min_top_score=effective_min_score,
            min_avg_score=0.0
        )

        if not chunks or not quality_eval["is_strong"]:
            status_code = quality_eval.get("status_code", "refused_empty_context")
            answer_text = (
                EMPTY_RETRIEVAL_FALLBACK_MESSAGE 
                if not chunks 
                else f"{STANDARD_SAFE_REFUSAL} (Retrieval quality below confidence threshold)."
            )
            latencies["total_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            
            formatted_sources = []
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                formatted_sources.append({
                    "source": chunk.get("source") or meta.get("source") or "unknown",
                    "chunk_id": chunk.get("id"),
                    "score": chunk.get("score"),
                    "rank": chunk.get("rank"),
                    "section": meta.get("section") or meta.get("section_heading"),
                    "doc_title": meta.get("doc_title")
                })

            return {
                "query": question,
                "answer": answer_text,
                "sources": formatted_sources,
                "status": status_code,
                "latency_ms": latencies["total_ms"],
                "stage_latencies_ms": latencies
            }

        # Stage 4: Assemble Context
        t0 = time.perf_counter()
        context = assemble_context(chunks)
        latencies["assemble_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 5: Generate Grounded Answer
        t0 = time.perf_counter()
        answer = generate_answer(
            query=question,
            context=context,
            model_name=self.config.chat_model,
            use_api=active_use_api
        )
        latencies["generate_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        formatted_sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            formatted_sources.append({
                "source": chunk.get("source") or meta.get("source") or meta.get("doc_title") or "unknown",
                "chunk_id": chunk.get("id"),
                "score": chunk.get("score"),
                "rank": chunk.get("rank"),
                "section": meta.get("section") or meta.get("section_heading"),
                "doc_title": meta.get("doc_title")
            })

        latencies["total_ms"] = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "query": question,
            "answer": answer,
            "sources": formatted_sources,
            "status": "answered",
            "latency_ms": latencies["total_ms"],
            "stage_latencies_ms": latencies
        }


_rag_service_instance: Optional[RAGService] = None


def get_rag_service(config: APIConfig = Depends(get_config)) -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(config)
    return _rag_service_instance


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Bad request (invalid query syntax or parameters)"},
        422: {"description": "Input validation error (e.g., question too short/long)"},
        500: {"description": "Internal server error"}
    }
)
def query_rag(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> QueryResponse:
    """
    Core RAG query endpoint.
    Accepts a user question, queries vector database, applies guardrails,
    synthesizes a grounded answer, and returns structured JSON with source citations.
    """
    try:
        result = rag_service.execute_query(
            question=request.question,
            k=request.k,
            score_threshold=request.score_threshold,
            use_api=request.use_api,
            metadata_filter=request.metadata_filter
        )

        source_objects = [
            Source(
                source=s.get("source", "unknown"),
                chunk_id=s.get("chunk_id"),
                score=s.get("score"),
                rank=s.get("rank"),
                section=s.get("section"),
                doc_title=s.get("doc_title")
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            query=result.get("query", request.question),
            answer=result.get("answer", ""),
            sources=source_objects,
            status=result.get("status", "answered"),
            latency_ms=result.get("latency_ms"),
            stage_latencies_ms=result.get("stage_latencies_ms"),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    except ValueError as err:
        logger.warning(f"ValueError processing query '{request.question}': {err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    except Exception as err:
        logger.error(f"Exception processing query '{request.question}': {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG service failed: {str(err)}"
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK
)
def chat_rag(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> ChatResponse:
    """
    Conversational RAG endpoint with session memory.
    """
    start_time = time.perf_counter()
    session_id = request.session_id

    # Retrieve or create session history
    if session_id not in _session_histories:
        _session_histories[session_id] = HistoryManager(max_token_budget=2000)
    history = _session_histories[session_id]

    # Ingest incoming history if provided
    if request.history:
        for msg in request.history:
            history.add_message(msg.role, msg.content)

    # Add current user message
    history.add_message("user", request.message)

    # Execute query
    result = rag_service.execute_query(
        question=request.message,
        k=request.k,
        score_threshold=request.score_threshold,
        use_api=request.use_api
    )

    answer = result.get("answer", "")
    history.add_message("assistant", answer)

    source_objects = [
        Source(
            source=s.get("source", "unknown"),
            chunk_id=s.get("chunk_id"),
            score=s.get("score"),
            rank=s.get("rank"),
            section=s.get("section"),
            doc_title=s.get("doc_title")
        )
        for s in result.get("sources", [])
    ]

    total_latency = round((time.perf_counter() - start_time) * 1000, 2)

    return ChatResponse(
        session_id=session_id,
        message=answer,
        sources=source_objects,
        status=result.get("status", "answered"),
        latency_ms=total_latency,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
