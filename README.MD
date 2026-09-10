# Knovera - Enterprise Grounded RAG Studio

Knovera is a full-stack, enterprise-grade Retrieval-Augmented Generation (RAG) platform consisting of a modular **FastAPI** Python backend and a modern **Next.js** (React 19 / TypeScript) web application.

---

## Monorepo Architecture

```
Knovera/
├── backend/                    # Complete Python RAG backend service
│   ├── main.py                 # FastAPI server launcher (Port 8000)
│   ├── run_tests.py            # Master test suite runner (35 automated tests)
│   ├── requirements.txt        # Python package dependencies
│   ├── .env / .env.example     # Backend environment configuration
│   ├── src/
│   │   ├── api/                # FastAPI routers (/query, /chat, /documents, /health, /evaluate)
│   │   ├── models/             # Pydantic schemas (QueryRequest, ChatRequest, Source, etc.)
│   │   ├── services/           # 14 core RAG algorithms (chunking, embedding, hybrid search, etc.)
│   │   └── config.py           # Configuration & CORS policy
│   ├── tests/                  # Consolidated unittest suite
│   ├── data/                   # ChromaDB vector index & test samples
│   ├── docs/                   # Architectural & technical reports
│   ├── scripts/                # Standalone demo & validation scripts
│   ├── prompts/                # System prompts & instruction guides
│   ├── templates/              # Prompt templates
│   └── uploads/                # User document uploads
│
└── frontend/                   # Modern Next.js (App Router) web application
    ├── package.json
    ├── next.config.ts
    ├── tsconfig.json
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx      # Root layout & Google typography
    │   │   ├── page.tsx        # Main Studio dashboard with tab navigation
    │   │   └── globals.css     # Premium Vanilla CSS design system (dark mode & glassmorphism)
    │   ├── components/
    │   │   ├── Header.tsx      # App header with live backend telemetry & status pill
    │   │   ├── QueryView.tsx   # Grounded Q&A with top-k controls & citation modal
    │   │   ├── ChatView.tsx    # Conversational chat with session memory & token budget meter
    │   │   ├── DocumentsView.tsx # Drag-and-drop document upload & live vector indexing
    │   │   ├── EvaluationView.tsx # RAG benchmark dashboard (Precision, Recall, Groundedness)
    │   │   └── CitationCard.tsx # Interactive source attribution cards & score meters
    │   └── lib/
    │       ├── api.ts          # Strongly typed client for backend REST API
    │       └── types.ts        # TypeScript interfaces matching backend models
```

---

## Quickstart Guide

### 1. Start the Backend (FastAPI)

```bash
cd backend

# Create & activate virtual environment (if not already active)
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure .env
cp .env.example .env

# Launch FastAPI server on http://localhost:8000
python main.py
```

The backend starts on `http://localhost:8000`:
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Telemetry**: `http://localhost:8000/api/health`

To execute backend automated tests:
```bash
cd backend
python run_tests.py
```

---

### 2. Start the Frontend (Next.js)

In a new terminal:

```bash
cd frontend

# Install Node dependencies (already initialized)
npm install

# Start Next.js development server
npm run dev
```

Open your browser at `http://localhost:3000` to access the **Knovera Studio**:
- **🔍 Grounded Search**: Submit natural language queries, inspect grounded answers, and expand source citation cards with similarity scores.
- **💬 Conversational Chat**: Multi-turn dialog with sliding window token budget tracking (up to 2,000 tokens) and session memory.
- **📂 Knowledge Documents**: Drag-and-drop `.pdf`, `.md`, `.txt`, and `.html` files for instant cleaning, token-chunking, embedding, and ChromaDB vector indexing.
- **📊 Quality Evaluation**: Run automated benchmarks over policy test cases and review Precision@K, Recall, Groundedness, and Latency scorecards.

---

## REST API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status, active models, and ChromaDB chunk count |
| `POST` | `/api/query` | Single-turn grounded question answering with citations |
| `POST` | `/api/chat` | Multi-turn conversational chat with session history |
| `GET` | `/api/documents` | List indexed documents and collection metadata |
| `POST` | `/api/documents` | Upload and runtime vector index new document |
| `POST` | `/api/evaluate` | Run batch RAG evaluation benchmarks |

---

## License & Contributing
Internal RAG Assistant & Enterprise Document Intelligence Platform.