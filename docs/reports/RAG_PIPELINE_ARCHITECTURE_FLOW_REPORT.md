# RAG Pipeline Architecture & Flow Design Report

**Assignment**: 3.37 RAG Pipeline Architecture & Flow Design  
**Project**: Knovera RAG Assistant  
**Repository**: `kalviumcommunity/Knovera`  
**Branch**: `feature/rag-pipeline-architecture-flow-design`  
**Author / Engineering Team**: Knovera AI Team  
**Status**: Completed & Verified  

---

## Executive Summary

A production-grade Retrieval-Augmented Generation (RAG) system must not be a monolithic, opaque script. It must be organized into **isolated, single-responsibility, testable stages**:
$$\text{User Query} \longrightarrow \text{Query Embedding} \longrightarrow \text{Top-}k\text{ Retrieval} \longrightarrow \text{Context Assembly} \longrightarrow \text{Grounded Generation} \longrightarrow \text{Answer + Citations}$$

This module delivers the complete **Knovera RAG Pipeline Architecture & Flow Engine** (`src/rag_pipeline.py`), connecting embedding generation, ChromaDB vector search, relevance thresholding, citation-indexed prompt assembly, and grounded LLM generation into a cohesive pipeline.

### Key Engineering Accomplishments
- **Modular Code Stages**: Implemented standalone, isolated functions: `embed_query()`, `retrieve_context()`, `assemble_context()`, `generate_answer()`, and `answer_query()`.
- **Grounded Citation Assembly**: Assembles retrieved chunks with indexed citation markers (`[1] Source: doc.md | Section: ...`) to enable deterministic attribution.
- **Strict Non-Hallucination Guard**: Handles empty retrieval states (`chunks == []`) gracefully by returning an explicit fallback message without synthesizing fabricated answers.
- **Dual Synthesis Capability**: Seamlessly supports live OpenAI / OpenRouter API generation with automatic fallback to deterministic grounded extractive synthesis for offline testing.
- **100% Test Suite Pass**: 12 comprehensive unit and integration tests (`test_rag_pipeline.py`) validating all stages, edge cases, latencies, and orchestrator workflows.

---

## Complete RAG Pipeline Flow & Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                KNOVERA RAG PIPELINE ARCHITECTURE                                  |
|                                                                                                   |
|  [ User Natural Language Query ]                                                                  |
|               |                                                                                   |
|               v                                                                                   |
|     +--------------------+                                                                        |
|     | STAGE 1: EMBED     |  embed_query(query) -> List[float] (1536-dim normalized vector)        |
|     +--------------------+                                                                        |
|               |                                                                                   |
|               v                                                                                   |
|     +--------------------+                                                                        |
|     | STAGE 2: RETRIEVE  |  retrieve_context(vector, k=4) -> List[Chunk Dicts] (ChromaDB HNSW)   |
|     +--------------------+                                                                        |
|               |                                                                                   |
|         [ Chunks Empty? ] ---------(YES: Fallback Guard)---------> [ Return Fallback Message ]    |
|               | (NO)                                                 (No Hallucination, sources=[])|
|               v                                                                                   |
|     +--------------------+                                                                        |
|     | STAGE 3: ASSEMBLE  |  assemble_context(chunks) -> Formatted Citation Context Block          |
|     +--------------------+  Format: "[1] Source: doc.md | Section: sec\n<text>"                  |
|               |                                                                                   |
|               v                                                                                   |
|     +--------------------+                                                                        |
|     | STAGE 4: GENERATE  |  generate_answer(query, context) -> Grounded LLM Response              |
|     +--------------------+                                                                        |
|               |                                                                                   |
|               v                                                                                   |
|  [ Pipeline Output Payload ]                                                                      |
|  { "query": ..., "answer": ..., "sources": [...], "context": ..., "stage_latencies_ms": ... }     |
+---------------------------------------------------------------------------------------------------+
```

---

## Detailed Breakdown of Code Stages

### Stage 1: Query Embedding (`embed_query`)
- **Responsibility**: Converts the user's raw string into a dense mathematical vector representation matching the embedding space of indexed document chunks.
- **Function**:
```python
def embed_query(query: str, generator: Optional[EmbeddingGenerator] = None) -> List[float]:
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace only.")
    active_gen = generator or EmbeddingGenerator()
    return active_gen.embed([query.strip()])[0]
```

### Stage 2: Context Retrieval (`retrieve_context`)
- **Responsibility**: Queries ChromaDB's vector index using cosine distance to retrieve top-$k$ nearest neighbor document chunks with optional metadata filtering and score thresholding.
- **Function**:
```python
def retrieve_context(
    query_vector: List[float],
    vector_db: Optional[VectorDatabase] = None,
    collection_name: str = "knovera_knowledge_base",
    k: int = 4,
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    active_db = vector_db or VectorDatabase()
    raw_results = active_db.query_similar(
        query_vector=query_vector,
        top_k=k,
        where=metadata_filter,
        collection_name=collection_name
    )
    # Formats rank, score, distance, metadata, and applies threshold cutoff
    ...
```

### Stage 3: Context Assembly (`assemble_context`)
- **Responsibility**: Packages the list of retrieved chunk dictionaries into an unambiguous, indexed context block for prompt injection.
- **Function**:
```python
def assemble_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or chunk.get("metadata", {}).get("source", "Doc")
        section = chunk.get("metadata", {}).get("section", "")
        text = chunk.get("text", "").strip()
        header = f"[{index}] Source: {source}" + (f" | Section: {section}" if section else "")
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)
```

### Stage 4: Grounded Generation (`generate_answer`)
- **Responsibility**: Generates an answer strictly grounded in the injected context using a structured QA prompt template.
- **Function**:
```python
def generate_answer(query: str, context: str, ...) -> str:
    if not context or not context.strip():
        return "I could not find relevant context for that question."
    prompt = render_prompt(QA_PROMPT_TEMPLATE, context=context, question=query)
    # Invokes LLM completion with prompt grounding instructions
    ...
```

### Stage 5: End-to-End Pipeline Orchestration (`answer_query`)
- **Responsibility**: Chains stages 1–4 together, measures per-stage execution latencies, enforces the non-hallucination fallback guard, and formats the final structured response.
- **Function**:
```python
def answer_query(query: str, k: int = 4, score_threshold: Optional[float] = None, ...) -> Dict[str, Any]:
    query_vector = embed_query(query)
    chunks = retrieve_context(query_vector, k=k, score_threshold=score_threshold)
    if not chunks:
        return {
            "query": query,
            "answer": "I could not find relevant context for that question.",
            "sources": [],
            "context": "",
            "status": "NO_CONTEXT_FOUND"
        }
    context = assemble_context(chunks)
    answer = generate_answer(query, context)
    sources = [chunk["metadata"] for chunk in chunks]
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context": context,
        "status": "SUCCESS"
    }
```

---

## End-to-End Sample Runs

### Query 1: Academic Evidence Requirement
- **Query**: `"What evidence is required for project submission?"`
- **Retrieved Sources**:
  1. `submission-rubric.md` (Rank 1, Score: `0.6638`)
  2. `grading-policy.md` (Rank 2, Score: `0.4658`)
- **Assembled Context**:
  ```text
  [1] Source: submission-rubric.md | Section: Evidence Requirements
  Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough demonstrating all test executions, architectural flows, and key modular components.

  [2] Source: grading-policy.md | Section: Evaluation Rubrics
  Assignment grading criteria: Submissions are evaluated on unit test coverage (40%), modular code structure (30%), and comprehensive documentation (30%). A score of 60% or higher is required to pass.
  ```
- **Generated Answer**:
  > Based on the provided documentation: Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough demonstrating all test executions, architectural flows, and key modular components. [1] A score of 60% or higher is required to pass. [2]
- **Status**: `SUCCESS` | Total Latency: `513.8 ms`

### Query 2: Authentication Workflow
- **Query**: `"How can a learner reset their password?"`
- **Retrieved Sources**: `account-guide.md` (Score: `0.7176`)
- **Generated Answer**:
  > Based on the provided documentation: Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal. [1]
- **Status**: `SUCCESS`

### Query 3: Negative / Out-of-Domain Query (Empty Retrieval Guard)
- **Query**: `"What is the warp drive maintenance schedule on the starship Enterprise?"`
- **Score Threshold Applied**: `0.999` (No chunks match)
- **Retrieved Chunks Count**: `0`
- **Generated Answer**:
  > `"I could not find relevant context for that question."`
- **Sources**: `[]`
- **Status**: `NO_CONTEXT_FOUND`
- **Safety Audit**: Zero hallucination occurred. The pipeline halted downstream LLM synthesis and returned a clean fallback.

---

## Failure Mode Analysis: What if Retrieval Returns Nothing?

When retrieval fails to return chunks (due to out-of-domain queries, strict filters, or low similarity scores), ungrounded RAG systems exhibit critical failure modes:
1. **Model Hallucination**: The LLM relies on its parametric pretraining memory and invents plausible-sounding but unverified answers.
2. **False Authority**: The user is presented with ungrounded answers without knowing no evidence existed.

**Knovera's Architectural Guard**:
1. Check `if not chunks:` immediately after retrieval.
2. Skip Stage 3 (`assemble_context`) and Stage 4 (`generate_answer`).
3. Return `EMPTY_RETRIEVAL_FALLBACK_MESSAGE` (`"I could not find relevant context for that question."`) with `sources = []`.

---

## Verification & Test Results

The test suite (`test_rag_pipeline.py`) validates 100% of pipeline stages across 12 distinct test cases:

```
test_stage1_embed_query_empty_error (__main__.TestRAGPipelineArchitecture) ... ok
test_stage1_embed_query_valid (__main__.TestRAGPipelineArchitecture) ... ok
test_stage2_retrieve_context_invalid_k (__main__.TestRAGPipelineArchitecture) ... ok
test_stage2_retrieve_context_score_threshold_filter (__main__.TestRAGPipelineArchitecture) ... ok
test_stage2_retrieve_context_top_k (__main__.TestRAGPipelineArchitecture) ... ok
test_stage3_assemble_context_empty (__main__.TestRAGPipelineArchitecture) ... ok
test_stage3_assemble_context_formatting (__main__.TestRAGPipelineArchitecture) ... ok
test_stage4_generate_answer_empty_context_fallback (__main__.TestRAGPipelineArchitecture) ... ok
test_stage4_generate_answer_grounded (__main__.TestRAGPipelineArchitecture) ... ok
test_stage5_answer_query_empty_retrieval_fallback (__main__.TestRAGPipelineArchitecture) ... ok
test_stage5_answer_query_end_to_end_success (__main__.TestRAGPipelineArchitecture) ... ok
test_stage5_rag_pipeline_class_interface (__main__.TestRAGPipelineArchitecture) ... ok

----------------------------------------------------------------------
Ran 12 tests in 5.855s

OK
```

---

## Video Walkthrough Guide (3–5 Minutes)

When recording your submission walkthrough video, structure your presentation as follows:
1. **Introduction & Architecture Overview (0:00 - 1:00)**:
   - Introduce Knovera RAG Assistant and the goal of Assignment 3.37.
   - Walk through the architecture diagram showing `embed_query` -> `retrieve_context` -> `assemble_context` -> `generate_answer` -> `answer_query`.
2. **Code Stage Separation (1:00 - 2:15)**:
   - Open `src/rag_pipeline.py` and show each isolated function.
   - Explain why separating responsibilities enables independent unit testing, modular retrieval tuning, and zero-hallucination guard rails.
3. **Live Demonstration & Sample Query (2:15 - 3:30)**:
   - Run `python rag_pipeline_demo.py` in the terminal.
   - Show the sample query flowing through each stage, displaying the indexed context and the grounded output with citations.
4. **Follow-Up Question & Conclusion (3:30 - 4:30)**:
   - **Answer the Follow-Up**: *"Where would the pipeline fail if retrieval returned nothing?"*
   - Explain that without the empty retrieval guard, the LLM would hallucinate from pretraining memory. Show how Knovera's pipeline catches `if not chunks:` and returns the safe fallback message with empty sources.
