"""
src/rag_api.py

Backend REST API for Knovera Retrieval-Augmented Generation (RAG) Service.
Exposes a production-ready FastAPI endpoint that accepts user questions,
orchestrates the RAG retrieval and generation pipeline, validates input,
handles errors gracefully, and returns structured JSON responses with grounded
answers and source attributions.
"""

import time
import datetime
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure Knovera root is in sys.path
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

from src.api_config import get_config, APIConfig
from src.rag_pipeline import (
    embed_query,
    retrieve_context,
    assemble_context,
    generate_answer,
    EMPTY_RETRIEVAL_FALLBACK_MESSAGE
)
from src.hallucination_guardrail import (
    evaluate_retrieval_quality,
    STANDARD_SAFE_REFUSAL
)
from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger("knovera.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ============================================================================
# PYDANTIC REQUEST & RESPONSE MODELS
# ============================================================================

class Source(BaseModel):
    """Structured representation of an attributed source chunk."""
    source: str = Field(..., description="Document filename or source identifier (e.g., 'submission-rubric.md')")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier (e.g., 'submission-rubric.md:2')")
    score: Optional[float] = Field(default=None, description="Cosine similarity score [0.0 - 1.0]")
    rank: Optional[int] = Field(default=None, description="Retrieval ranking position (1-indexed)")
    section: Optional[str] = Field(default=None, description="Document section or header")
    doc_title: Optional[str] = Field(default=None, description="Human-readable document title")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "submission-rubric.md",
                "chunk_id": "submission-rubric.md:2",
                "score": 0.8425,
                "rank": 1,
                "section": "Submission Requirements",
                "doc_title": "Project Submission Rubric"
            }
        }
    }


class QueryRequest(BaseModel):
    """Request payload for RAG question answering endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question to submit to the RAG system."
    )
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional maximum number of context chunks to retrieve (defaults to server config TOP_K)."
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity score cutoff."
    )
    use_api: Optional[bool] = Field(
        default=None,
        description="Optional flag to toggle between live LLM generation and deterministic synthesis."
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filtering dictionary for targeted vector search."
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 3:
            raise ValueError("Question must contain at least 3 non-whitespace characters.")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What evidence is required for project submission?",
                "k": 4,
                "score_threshold": 0.50
            }
        }
    }


class QueryResponse(BaseModel):
    """Structured response payload returned by the RAG query endpoint."""
    query: str = Field(..., description="Original user query received by the endpoint.")
    answer: str = Field(..., description="Grounded answer generated from retrieved context.")
    sources: List[Source] = Field(default_factory=list, description="List of source citations supporting the answer.")
    status: str = Field(..., description="Execution status: 'answered', 'refused_empty_context', 'refused_low_similarity', etc.")
    latency_ms: Optional[float] = Field(default=None, description="Total end-to-end request processing time in milliseconds.")
    stage_latencies_ms: Optional[Dict[str, float]] = Field(default=None, description="Per-stage latency breakdown in milliseconds.")
    timestamp: Optional[str] = Field(default=None, description="ISO-formatted UTC timestamp of the response.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What evidence is required for project submission?",
                "answer": "The submission requires a PR link, sample output, and a video explanation.",
                "sources": [
                    {
                        "source": "submission-rubric.md",
                        "chunk_id": "submission-rubric.md:2",
                        "score": 0.8425,
                        "rank": 1,
                        "section": "Submission Requirements"
                    }
                ],
                "status": "answered",
                "latency_ms": 245.8,
                "timestamp": "2026-09-09T08:30:00Z"
            }
        }
    }


class HealthResponse(BaseModel):
    """Service health and diagnostics response."""
    status: str = Field(..., description="Overall health status ('ok', 'degraded', 'unhealthy')")
    service: str = Field(default="knovera-rag-backend-api")
    version: str = Field(..., description="API Version")
    environment: str = Field(..., description="Deployment environment")
    collection_name: str = Field(..., description="Active Vector DB collection")
    embedding_model: str = Field(..., description="Active embedding model")
    chat_model: str = Field(..., description="Active chat LLM model")
    vector_db_available: bool = Field(..., description="Whether vector database connection is healthy")
    timestamp: str = Field(..., description="Current ISO timestamp")


class ConfigResponse(BaseModel):
    """Sanitized system configuration view."""
    environment: str
    api_host: str
    api_port: int
    embedding_model: str
    chat_model: str
    vector_db_url: str
    collection_name: str
    top_k: int
    min_top_score: float
    use_live_api: bool
    api_key_status: str


class ErrorDetail(BaseModel):
    """Structured error payload for API exceptions."""
    detail: str
    error_code: str
    status_code: int
    timestamp: str


# ============================================================================
# RAG SERVICE PIPELINE EXECUTION ENGINE
# ============================================================================

class RAGService:
    """
    RAG service engine wrapping embedding, retrieval, guardrails, and answer synthesis.
    Maintains reusable vector store and embedding generator singletons.
    """

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
        """
        Executes end-to-end RAG pipeline for a given user question.
        
        Args:
            question: Cleaned user question string.
            k: Top-k chunks to retrieve.
            score_threshold: Optional similarity cutoff.
            use_api: Flag for live LLM API call.
            metadata_filter: Metadata filtering conditions.
            
        Returns:
            Dict[str, Any]: Formatted result dictionary.
        """
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

        # Handle Empty or Weak Retrieval States
        if not chunks or not quality_eval["is_strong"]:
            status_code = quality_eval.get("status_code", "refused_empty_context")
            answer_text = (
                EMPTY_RETRIEVAL_FALLBACK_MESSAGE 
                if not chunks 
                else f"{STANDARD_SAFE_REFUSAL} (Retrieval quality below confidence threshold)."
            )
            latencies["total_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            
            # Format any available weak sources for diagnostic inspection if present
            formatted_sources: List[Dict[str, Any]] = []
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

        # Format sources according to Source schema
        formatted_sources: List[Dict[str, Any]] = []
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


# ============================================================================
# FASTAPI APPLICATION FACTORY
# ============================================================================

def create_app(service: Optional[RAGService] = None) -> FastAPI:
    """
    Creates and configures the FastAPI application instance.
    
    Args:
        service: Optional pre-configured RAGService instance.
        
    Returns:
        FastAPI: Configured web application.
    """
    config = get_config()
    rag_service = service or RAGService(config)

    app = FastAPI(
        title=config.api_title,
        version=config.api_version,
        description=(
            "Knovera RAG Backend API: High-performance, production-ready REST service "
            "for grounded Question Answering with strict hallucination guardrails and source citations."
        ),
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Enable Cross-Origin Resource Sharing (CORS) for frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Measure and attach X-Response-Time header
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(process_time)
        return response

    # ------------------------------------------------------------------------
    # CUSTOM EXCEPTION HANDLERS
    # ------------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Custom handler for Pydantic request validation failures (HTTP 422)."""
        errors = exc.errors()
        error_messages = []
        for err in errors:
            loc = " -> ".join([str(p) for p in err.get("loc", [])])
            msg = err.get("msg", "Validation error")
            error_messages.append(f"{loc}: {msg}")
            
        detail_msg = "; ".join(error_messages)
        logger.warning(f"Validation failure on {request.url.path}: {detail_msg}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": detail_msg,
                "error_code": "VALIDATION_ERROR",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Custom handler for bad input values (HTTP 400)."""
        logger.warning(f"Value error on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_code": "BAD_REQUEST",
                "status_code": status.HTTP_400_BAD_REQUEST,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Custom fallback handler for uncaught server errors (HTTP 500)."""
        logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "RAG service failed to process request due to an internal server error.",
                "error_code": "INTERNAL_SERVER_ERROR",
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )

    # ------------------------------------------------------------------------
    # API ENDPOINTS
    # ------------------------------------------------------------------------

    @app.get("/", tags=["General"])
    def root_endpoint() -> Dict[str, Any]:
        """Root endpoint providing service metadata and navigation links."""
        return {
            "name": config.api_title,
            "version": config.api_version,
            "environment": config.app_env,
            "status": "online",
            "documentation": "/docs",
            "endpoints": {
                "query": "POST /query",
                "health": "GET /health",
                "config": "GET /config"
            }
        }

    @app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
    def health_check() -> HealthResponse:
        """
        Health & readiness check endpoint.
        Verifies vector store availability and returns active model configurations.
        """
        vector_db_ok = False
        try:
            # Check vector DB collection accessibility
            if rag_service.vector_db.is_reachable():
                item_count = rag_service.vector_db.count(config.collection_name)
                vector_db_ok = item_count >= 0
        except Exception as e:
            logger.error(f"Healthcheck vector DB probe failed: {e}")
            vector_db_ok = False

        overall_status = "ok" if vector_db_ok else "degraded"

        return HealthResponse(
            status=overall_status,
            service="knovera-rag-backend-api",
            version=config.api_version,
            environment=config.app_env,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            chat_model=config.chat_model,
            vector_db_available=vector_db_ok,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    @app.get("/config", response_model=ConfigResponse, tags=["Administration"])
    def get_system_config() -> ConfigResponse:
        """Returns sanitized runtime configuration with masked credentials."""
        cfg_dict = config.to_dict(mask_secrets=True)
        return ConfigResponse(
            environment=cfg_dict["app_env"],
            api_host=cfg_dict["api_host"],
            api_port=cfg_dict["api_port"],
            embedding_model=cfg_dict["embedding_model"],
            chat_model=cfg_dict["chat_model"],
            vector_db_url=cfg_dict["vector_db_url"],
            collection_name=cfg_dict["collection_name"],
            top_k=cfg_dict["top_k"],
            min_top_score=cfg_dict["min_top_score"],
            use_live_api=cfg_dict["use_live_api"],
            api_key_status=cfg_dict["api_key_masked"]
        )

    @app.post(
        "/query",
        response_model=QueryResponse,
        status_code=status.HTTP_200_OK,
        tags=["RAG Core"],
        responses={
            400: {"description": "Bad request (invalid query syntax or parameters)"},
            422: {"description": "Input validation error (e.g., question too short/long)"},
            500: {"description": "Internal server error"}
        }
    )
    def query_rag(request: QueryRequest) -> QueryResponse:
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

            # Map raw sources list into Source Pydantic models
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

    return app


# Default application instance for ASGI servers (e.g. `uvicorn src.rag_api:app`)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    print(f"Starting Knovera RAG Backend API on http://{cfg.api_host}:{cfg.api_port}")
    uvicorn.run("src.rag_api:app", host=cfg.api_host, port=cfg.api_port, reload=True)
