"""
src/models/schemas.py

Pydantic Request, Response, and Data Transfer Models for Knovera RAG Backend.
Provides strict validation, descriptive field definitions, and schema examples
for seamless integration with modern web frontends like Next.js.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# RAG QUERY & SOURCE CITATIONS
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


# ============================================================================
# CONVERSATIONAL CHAT
# ============================================================================

class ChatMessage(BaseModel):
    """Single chat message in conversation history."""
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    """Conversational chat request with session tracking and optional history."""
    session_id: str = Field(default="default_session", description="Unique session identifier for history tracking")
    message: str = Field(..., min_length=1, description="Latest user message")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Optional prior conversation turns")
    k: Optional[int] = Field(default=None, ge=1, le=20, description="Top-K chunks to retrieve")
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    use_api: Optional[bool] = Field(default=None)


class ChatResponse(BaseModel):
    """Conversational chat response."""
    session_id: str
    message: str
    sources: List[Source] = Field(default_factory=list)
    status: str = Field(default="answered")
    latency_ms: Optional[float] = None
    timestamp: Optional[str] = None


# ============================================================================
# HEALTH & CONFIGURATION
# ============================================================================

class HealthResponse(BaseModel):
    """Service health and diagnostics response."""
    status: str = Field(..., description="Overall health status ('ok', 'degraded', 'unhealthy')")
    service: str = Field(default="knovera-rag-backend-api")
    version: str = Field(..., description="API Version")
    environment: str = Field(..., description="Deployment environment")
    collection_name: str = Field(..., description="Active Vector DB collection")
    total_indexed_chunks: Optional[int] = Field(default=None, description="Total chunks currently in vector database")
    embedding_model: str = Field(..., description="Active embedding model")
    chat_model: str = Field(..., description="Active chat LLM model")
    vector_db_available: bool = Field(..., description="Whether vector database connection is healthy")
    timestamp: str = Field(..., description="Current ISO timestamp")


class ConfigResponse(BaseModel):
    """Sanitized system configuration view."""
    environment: str
    api_host: str
    api_port: int
    cors_origins: List[str]
    embedding_model: str
    chat_model: str
    vector_db_url: str
    collection_name: str
    top_k: int
    min_top_score: float
    use_live_api: bool
    api_key_status: str


# ============================================================================
# DOCUMENT UPLOAD & INDEXING
# ============================================================================

class DocumentIndexingSummary(BaseModel):
    """Summary metadata of document chunking and vector indexing."""
    document: str = Field(..., description="Local path to the stored document")
    filename: str = Field(..., description="Original filename")
    chunks: int = Field(..., description="Total token chunks generated")
    indexed: int = Field(..., description="Total chunks indexed into vector store")
    raw_characters: Optional[int] = Field(default=None, description="Raw character count before cleaning")
    cleaned_characters: Optional[int] = Field(default=None, description="Cleaned character count after normalization")
    collection_name: Optional[str] = Field(default=None, description="Vector store collection name")
    stage_latencies_ms: Optional[Dict[str, float]] = Field(default=None, description="Ingestion stage latencies in milliseconds")


class DocumentUploadResponse(BaseModel):
    """Structured response payload for document upload and indexing endpoint."""
    status: str = Field(default="indexed", description="Indexing status: 'indexed', 'failed'")
    filename: str = Field(..., description="Name of the uploaded file")
    summary: DocumentIndexingSummary = Field(..., description="Detailed ingestion, chunking, and embedding summary")
    message: Optional[str] = Field(
        default="Document successfully ingested, chunked, embedded, and indexed into knowledge base.",
        description="Human-readable success message"
    )
    timestamp: Optional[str] = Field(default=None, description="ISO-formatted UTC timestamp of the upload operation")


class DocumentInfo(BaseModel):
    """Document record in the knowledge base."""
    filename: str
    chunk_count: int
    collection_name: str
    indexed_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response containing list of documents in knowledge base."""
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


# ============================================================================
# EVALUATION SCHEMAS
# ============================================================================

class EvaluationItem(BaseModel):
    """Single query evaluation input."""
    query: str
    expected_answer: Optional[str] = None
    expected_doc: Optional[str] = None
    min_score: Optional[float] = 0.5


class EvaluationRequest(BaseModel):
    """Batch RAG evaluation request."""
    test_cases: List[EvaluationItem]


class EvaluationMetricSummary(BaseModel):
    """Aggregated evaluation metrics."""
    total_queries: int
    mean_precision_at_k: float
    mean_recall: float
    mean_reciprocal_rank: float
    mean_groundedness: float
    average_latency_ms: float


class EvaluationResponse(BaseModel):
    """Batch RAG evaluation results."""
    summary: EvaluationMetricSummary
    details: List[Dict[str, Any]]
    timestamp: str


# ============================================================================
# ERROR SCHEMA
# ============================================================================

class ErrorDetail(BaseModel):
    """Structured error payload for API exceptions."""
    detail: str
    error_code: str
    status_code: int
    timestamp: str
