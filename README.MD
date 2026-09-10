# Knovera - Production-Ready Modular RAG Backend

Knovera is a high-performance, production-ready Retrieval-Augmented Generation (RAG) backend engineered to power modern enterprise search and conversational AI applications. Built with **FastAPI**, **ChromaDB**, and **OpenAI/OpenRouter**, it features full CORS support for rapid integration with upcoming **Next.js** frontend applications.

---

## Architecture & Segregated Directory Structure

The codebase has been refactored and segregated into clean, modular architectural layers:

```
Knovera/
├── main.py                     # Unified FastAPI application entrypoint (Uvicorn server)
├── run_tests.py                # Master test runner (executes consolidated test suite)
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment configuration
│
├── src/                        # Core backend source code
│   ├── config.py               # Centralized configuration manager & CORS policy
│   ├── api/                    # REST API layer
│   │   ├── app.py              # FastAPI application factory & middleware setup
│   │   └── routes/             # Modular route handlers
│   │       ├── health.py       # Health check & system diagnostics
│   │       ├── query.py        # Question answering & conversational chat endpoints
│   │       ├── documents.py    # Document upload, indexing, and vector management
│   │       └── evaluation.py   # RAG pipeline evaluation & benchmark metrics
│   ├── models/                 # Pydantic data contracts & schemas
│   │   └── schemas.py          # QueryRequest, QueryResponse, ChatRequest, Source, etc.
│   └── services/               # 14-layer core RAG algorithms & service engines
│       ├── text_cleaner.py     # Whitespace normalization & artifact removal
│       ├── document_loader.py  # Multi-format document parser (PDF, MD, TXT, HTML)
│       ├── token_chunker.py    # Token-aware sliding chunker with overlap
│       ├── chunk_tagger.py     # Metadata extraction & structural tagging
│       ├── embedding_generator.py # OpenRouter/OpenAI API vector embeddings
│       ├── embedding_quality_checker.py # Vector validation & norm/NaN checks
│       ├── vector_store.py     # ChromaDB vector store integration
│       ├── top_k_retriever.py  # Cosine similarity dense retrieval
│       ├── hybrid_retriever.py # Reciprocal Rank Fusion (RRF) dense + sparse search
│       ├── reranker.py         # Cross-encoder & LLM-based re-ranking
│       ├── context_injector.py # Structured prompt augmentation with XML tags
│       ├── hallucination_guardrail.py # Score gating & confidence-based refusal
│       ├── grounded_generator.py # Grounded text synthesis strictly from context
│       ├── source_tracer.py    # Citation extraction, attribution & grounding checks
│       ├── history_manager.py  # Sliding token budget conversation history
│       ├── document_indexer.py # Document ingestion & embedding storage
│       ├── rag_pipeline.py     # End-to-end RAG pipeline orchestration
│       └── rag_evaluator.py    # Precision, recall, and answer quality scoring
│
├── tests/                      # Consolidated test suite
│   └── test_all.py             # Single comprehensive test suite covering all 14 layers
│
├── data/                       # Local datasets & vector store storage
│   ├── chroma_db/              # Persistent ChromaDB vector index
│   └── samples/                # Sample chunks, embeddings, and test datasets
│
├── docs/                       # Technical reports & design documentation
│   └── reports/                # 25+ detailed architectural & benchmark reports
│
├── scripts/                    # Standalone utility & demo scripts
│   └── demos/                  # Individual layer demonstration scripts
│
├── prompts/                    # System prompt templates & notes
├── uploads/                    # User-uploaded documents ready for indexing
└── outputs/                    # Evaluation metrics & benchmark run outputs
```

---

## Quickstart

### 1. Prerequisites
- Python 3.8+ (Python 3.10+ recommended)
- Node.js 18+ (for upcoming Next.js frontend)

### 2. Environment Setup
```bash
# Clone the repository and navigate to root
cd Knovera

# Create & activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

Key variables in `.env`:
```ini
# Choose one of:
OPENROUTER_API_KEY=your_openrouter_api_key_here
# or
OPENAI_API_KEY=your_openai_api_key_here

# Model Selection
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=openai/gpt-4o-mini

# Server & CORS Configuration
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

---

## Running the Backend

Start the unified FastAPI server:
```bash
python main.py
```

The server will initialize on `http://localhost:8000`:
- **API Base**: `http://localhost:8000/api`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## Running the Unified Test Suite

All unit tests, integration tests, and API validation tests across all 14 architectural layers have been consolidated into a single master test file.

To run the complete test suite:
```bash
python run_tests.py
```
Or via standard unittest:
```bash
python -m unittest tests/test_all.py
```

The test suite validates:
1. **Text Cleaner**: Whitespace normalization, artifact & noisy character removal.
2. **Document Loader**: Multi-format document extraction and metadata extraction.
3. **Token Chunker**: Sliding-window chunking, token estimation, and edge-case handling.
4. **Chunk Tagger**: Metadata attribution and schema validation.
5. **Embedding Generator & Quality Checker**: Real/mock vector creation, zero-vector checks, and dimension validation.
6. **Vector Store**: ChromaDB collection lifecycle, upsert, query, and deletion.
7. **Retrievers & Reranker**: Dense top-k similarity, Reciprocal Rank Fusion (hybrid), and cross-encoder re-ranking.
8. **Context Injection & Guardrails**: Safe refusal triggers, confidence thresholds, and augmented prompt formatting.
9. **Grounded Generator & Source Tracer**: Grounded synthesis, citation extraction (`[Source: chunk_id]`), and source verification.
10. **History Manager**: Sliding conversation window and token-budget pruning.
11. **RAG Pipeline**: End-to-end question answering execution.
12. **RAG Evaluator**: Precision, recall, and answer quality scoring.
13. **FastAPI REST Endpoints**: `/api/health`, `/api/query`, `/api/chat`, `/api/documents`, `/api/evaluate`.

---

## Next.js Frontend Integration Guide

Knovera is pre-configured with CORS for `http://localhost:3000` to connect directly with your Next.js application (App Router or Pages Router).

### Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, active model, and vector DB stats |
| `POST` | `/api/query` | Single-turn grounded question answering with source citations |
| `POST` | `/api/chat` | Multi-turn conversational search with session memory |
| `GET` | `/api/documents` | List indexed documents and collection metadata |
| `POST` | `/api/documents/upload` | Upload & automatically index new document (PDF, TXT, MD) |
| `POST` | `/api/evaluate` | Benchmark RAG pipeline accuracy across sample questions |

---

### TypeScript Data Interfaces (for Next.js)

Save these types in `types/rag.ts` in your Next.js project:

```typescript
export interface SourceCitation {
  source: string;
  chunk_id: string;
  score?: number;
  rank?: number;
  section?: string;
  doc_title?: string;
}

export interface QueryRequest {
  question: string;
  k?: number;
  score_threshold?: number;
  use_api?: boolean;
  metadata_filter?: Record<string, any>;
}

export interface QueryResponse {
  query: string;
  answer: string;
  sources: SourceCitation[];
  status: "answered" | "refused_low_confidence" | "refused_empty_context";
  latency_ms?: number;
  stage_latencies_ms?: Record<string, number>;
  timestamp: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatRequest {
  session_id: string;
  message: string;
  history?: ChatMessage[];
  k?: number;
  score_threshold?: number;
  use_api?: boolean;
}

export interface ChatResponse {
  session_id: string;
  message: string;
  sources: SourceCitation[];
  status: string;
  latency_ms?: number;
  timestamp: string;
}
```

---

### Next.js API Client Example

Create an API client helper in your Next.js app (`lib/api.ts`):

```typescript
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

/**
 * Ask a single-turn question with grounded sources
 */
export async function askQuestion(question: string, topK: number = 4): Promise<QueryResponse> {
  const res = await fetch(`${BACKEND_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, k: topK }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Query failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Send a message in an ongoing conversational session
 */
export async function sendChatMessage(
  sessionId: string,
  message: string,
  history: ChatMessage[] = []
): Promise<ChatResponse> {
  const res = await fetch(`${BACKEND_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message, history }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Chat failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Check backend health & database connection
 */
export async function checkBackendHealth() {
  const res = await fetch(`${BACKEND_URL}/api/health`);
  return res.json();
}
```

---

## License & Contributing
Internal RAG Assistant Foundation for enterprise search and document intelligence.