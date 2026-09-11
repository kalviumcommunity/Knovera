"""
src/api/routes/query.py

Question answering and conversational chat endpoints for Knovera RAG Backend.
"""

import time
import datetime
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Header

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
from src.services.semantic_engine import reformulate_conversational_query

from src.services.guardrail_engine import get_guardrail_engine

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
        metadata_filter: Optional[Dict[str, Any]] = None,
        admin_id: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}

        target_k = k if k is not None else self.config.top_k
        active_use_api = use_api if use_api is not None else self.config.use_live_api
        target_admin = admin_id or "admin@knovera.ai"

        # Stage 0: AI Safety Guardrails Pre-Execution Scan
        guardrail_engine = get_guardrail_engine()
        decision = guardrail_engine.evaluate_input(question, admin_id=target_admin)

        if decision.is_refused:
            latencies["total_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "query": decision.sanitized_query,
                "answer": decision.refusal_message,
                "sources": [],
                "status": "refused_guardrail",
                "guardrail_status": "refused",
                "guardrail_name": decision.triggered_guardrail or "Prompt Injection & Jailbreak Prevention",
                "latency_ms": latencies["total_ms"],
                "stage_latencies_ms": latencies
            }

        effective_query = decision.sanitized_query

        # Stage 1: Embed Query (using sanitized query with PII masked)
        t0 = time.perf_counter()
        query_vector = embed_query(effective_query, generator=self.embedding_generator)
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
                "query": effective_query,
                "answer": answer_text,
                "sources": formatted_sources,
                "status": status_code,
                "guardrail_status": "refused",
                "guardrail_name": "Grounded Hallucination Shield",
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
            query=effective_query,
            context=context,
            model_name=self.config.chat_model,
            use_api=active_use_api
        )
        latencies["generate_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 6: Output Guardrail & Secret Scrubbing
        sanitized_answer = guardrail_engine.sanitize_output(
            answer,
            append_disclaimer=decision.append_disclaimer
        )

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
            "query": effective_query,
            "answer": sanitized_answer,
            "sources": formatted_sources,
            "status": "answered",
            "guardrail_status": decision.guardrail_status,
            "guardrail_name": decision.triggered_guardrail or "Grounded Hallucination Shield",
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
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id"),
    rag_service: RAGService = Depends(get_rag_service)
) -> QueryResponse:
    """
    Core RAG query endpoint.
    Accepts a user question, queries vector database, applies guardrails,
    synthesizes a grounded answer, and returns structured JSON with source citations.
    """
    try:
        target_admin = x_admin_id or "admin@knovera.ai"
        result = rag_service.execute_query(
            question=request.question,
            k=request.k,
            score_threshold=request.score_threshold,
            use_api=request.use_api,
            metadata_filter=request.metadata_filter,
            admin_id=target_admin
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

        # Record query log in MongoDB Atlas
        try:
            from src.services.mongo_storage import get_mongo_storage
            storage = get_mongo_storage()
            storage.create_log({
                "session_id": "api_query",
                "user": "api_client",
                "action": f"API Query: {request.question[:45]}",
                "query_snippet": request.question,
                "input": request.question,
                "output": result.get("answer", ""),
                "sources": [
                    {
                        "source": s.source,
                        "doc_title": s.doc_title,
                        "section": s.section,
                        "score": s.score,
                        "chunk_id": s.chunk_id
                    }
                    for s in source_objects
                ],
                "latency_ms": result.get("latency_ms", 0.0),
                "groundedness_score": source_objects[0].score if source_objects else 0.0,
                "guardrail_status": result.get("guardrail_status", "passed"),
                "guardrail_name": result.get("guardrail_name", "Grounded Hallucination Shield"),
                "status": "success" if result.get("guardrail_status") in ["passed", "redacted"] else ("error" if result.get("guardrail_status") == "refused" else "warning"),
                "details": f"Guardrail: {result.get('guardrail_name')} ({result.get('guardrail_status')}). Citations: {len(source_objects)}."
            }, admin_id=target_admin)
        except Exception as log_err:
            logger.warning(f"Could not record query log in MongoDB: {log_err}")

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
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id"),
    rag_service: RAGService = Depends(get_rag_service)
) -> ChatResponse:
    """
    Conversational RAG endpoint with session memory and MongoDB multi-tenant telemetry.
    """
    start_time = time.perf_counter()
    session_id = request.session_id
    target_admin = x_admin_id or "admin@knovera.ai"

    # Retrieve or create session history
    if session_id not in _session_histories:
        _session_histories[session_id] = HistoryManager(max_token_budget=2000)
    history = _session_histories[session_id]

    # Ingest incoming history if provided
    if request.history:
        for msg in request.history:
            history.add_message(msg.role, msg.content)

    # Contextualize conversational follow-ups (e.g., "can you tell more about this", "elaborate")
    combined_history = request.history if request.history else history.get_messages()
    effective_question = reformulate_conversational_query(
        current_query=request.message,
        history=combined_history
    )

    # Execute query through guardrail pipeline
    result = rag_service.execute_query(
        question=effective_question,
        k=request.k,
        score_threshold=request.score_threshold,
        use_api=request.use_api,
        admin_id=target_admin
    )

    answer = result.get("answer", "")
    sanitized_query = result.get("query", effective_question)

    # Add interaction turns to history using user's query
    history.add_message("user", request.message)
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

    source_records = [
        {
            "source": s.source,
            "doc_title": s.doc_title,
            "section": s.section,
            "score": s.score,
            "chunk_id": s.chunk_id
        }
        for s in source_objects
    ]

    total_latency = round((time.perf_counter() - start_time) * 1000, 2)
    gr_status = result.get("guardrail_status", "passed")
    gr_name = result.get("guardrail_name", "Grounded Hallucination Shield")

    # Record sanitized audit log in MongoDB Atlas (guarantees no raw PII in database logs)
    try:
        from src.services.mongo_storage import get_mongo_storage
        storage = get_mongo_storage()
        storage.create_log({
            "session_id": session_id,
            "user": x_user_id or "user@knovera.ai",
            "action": f"Chat Query: {sanitized_query[:45]}",
            "query_snippet": sanitized_query,
            "input": request.message,
            "output": answer,
            "sources": source_records,
            "latency_ms": total_latency,
            "groundedness_score": source_objects[0].score if source_objects else 0.0,
            "guardrail_status": gr_status,
            "guardrail_name": gr_name,
            "status": "success" if gr_status in ["passed", "redacted"] else ("error" if gr_status == "refused" else "warning"),
            "details": f"Guardrail: {gr_name} ({gr_status}). Citations: {len(source_objects)}."
        }, admin_id=target_admin)
    except Exception as log_err:
        logger.warning(f"Could not record query log in MongoDB: {log_err}")

    return ChatResponse(
        session_id=session_id,
        message=answer,
        sources=source_objects,
        status=result.get("status", "answered"),
        latency_ms=total_latency,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
