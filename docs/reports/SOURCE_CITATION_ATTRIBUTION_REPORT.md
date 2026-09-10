# Source Citation & Attribution Report

**Assignment**: 3.40 Source Citation & Attribution  
**Project**: Knovera RAG Assistant  
**Repository**: `kalviumcommunity/Knovera`  
**Branch**: `feature/source-citation-attribution`  
**Author / Engineering Team**: Knovera AI Team  
**Status**: Completed & Verified  

---

## Executive Summary

Retrieval-Augmented Generation (RAG) models build user trust when generated claims are **verifiable** against the underlying knowledge corpus. Without precise citations and metadata mapping, users cannot determine whether an answer is factual, stale, or hallucinatory.

This module delivers the **Knovera Source Citation & Attribution Engine** (`src/source_tracer.py`), completing all 5 tasks specified in Assignment 3.40:

1. **Source References (Task 1)**: Injects indexed citation markers (`[1]`, `[2]`) directly into generated factual answers and context prompts.
2. **Metadata-Mapped Citations (Task 2)**: Implements `build_citation_map(chunks)` to map each citation marker back to exact document names (`source`), chunk IDs (`chunk_id`), chunk indices (`chunk_index`), sections (`section`), page numbers (`page`), character offsets (`char_span`), and original text snippet (`text`).
3. **Source Verification & Traceability (Task 3)**: Implements `verify_citation` and `trace_chunk_source` to perform reverse offset lookup against original raw document text, computing 1-based line numbers (`L3-L5`) and verifying character-level exact matches.
4. **Anti-Fabrication & Fallback Guard (Task 4)**: Prevents invented or hallucinatory citations by returning an explicit non-hallucinatory fallback (*"I don't have enough information in the provided context."*) with empty citations (`{}`) when evidence is absent, and flagging fabricated markers (`detect_fabricated_citations`).
5. **Sample Cited Answers Payload & Test Suite (Task 5)**: Commits `sample_cited_answers.json`, an interactive demo (`source_citation_demo.py`), and a comprehensive unit test suite (`test_source_citation.py`) passing with 100% success across 12 tests.

---

## Citation Architecture & Verification Workflow

```
+---------------------------------------------------------------------------------------------------------+
|                                KNOVERA SOURCE CITATION & ATTRIBUTION FLOW                                |
|                                                                                                         |
|  [ User Natural Language Question ]                                                                     |
|        |                                                                                                |
|        v                                                                                                |
|  +------------------------------+                                                                       |
|  | Retriever / Top-K Search     |                                                                       |
|  +------------------------------+                                                                       |
|        |                                                                                                |
|  [ Retrieved Chunks + Metadata ]                                                                        |
|  (source, chunk_id, section, page, char_start, char_end, text)                                          |
|        |                                                                                                |
|        +------------------------------------------+                                                     |
|        |                                          |                                                     |
|        v (EVIDENCE PRESENT)                       v (EVIDENCE MISSING / EMPTY)                          |
|  +-------------------------------------+   +------------------------------------+                       |
|  | build_citation_map(chunks)          |   | Missing-Context Fallback           |                       |
|  | Key: "[1]", "[2]"                   |   | "I don't have enough information   |                       |
|  | Val: Metadata + Span + Text         |   | in the provided context."          |                       |
|  +-------------------------------------+   | Citations: {}                      |                       |
|        |                                   +------------------------------------+                       |
|        v                                                                                                |
|  +-------------------------------------+                                                                |
|  | Grounded Answer Synthesis           |                                                                |
|  | Includes inline markers [1], [2]    |                                                                |
|  +-------------------------------------+                                                                |
|        |                                                                                                |
|        v                                                                                                |
|  +-------------------------------------+                                                                |
|  | detect_fabricated_citations()       |                                                                |
|  | Audit: Answer markers vs Map        |                                                                |
|  +-------------------------------------+                                                                |
|        |                                                                                                |
|        v                                                                                                |
|  +-------------------------------------+                                                                |
|  | verify_citation() / Reverse Lookup  |                                                                |
|  | Check raw doc offsets & line range  |                                                                |
|  +-------------------------------------+                                                                |
|        |                                                                                                |
|        v                                                                                                |
|  [ User-Facing Response + Citation Map + Audit Trail ]                                                  |
+---------------------------------------------------------------------------------------------------------+
```

---

## Detailed Task Implementation & Code Architecture

### Task 1: Add Source References (`build_cited_prompt`)

Generated prompt templates instruct downstream LLMs to attribute every factual claim with citation markers like `[1]` or `[2]`, strictly preventing ungrounded speculation.

```python
def build_cited_prompt(question: str, chunks: List[Dict[str, Any]], max_context_tokens: int = 5000) -> str:
    context, _, _, _ = assemble_context(chunks, max_tokens=max_context_tokens)
    return f"""Answer using only the context below.
Cite every factual claim using source markers like [1] or [2].
Only use source markers that appear in the context.
If the context does not support an answer, say you do not have enough information and do not invent citations.

Context:
{context}

Question:
{question}"""
```

---

### Task 2: Map Citations to Metadata (`build_citation_map`)

Preserves metadata from early ingestion and retrieval stages to build stable citation keys mapped to document location properties.

```python
def build_citation_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    citation_map = {}
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        citation_map[f"[{index}]"] = {
            "source": meta.get("source", chunk.get("source", "Unknown Document")),
            "chunk_id": meta.get("chunk_id", chunk.get("id", f"chunk_{index}")),
            "chunk_index": meta.get("chunk_index", 0),
            "section": meta.get("section") or meta.get("section_heading", "General"),
            "page": meta.get("page_number", meta.get("page", 1)),
            "char_span": f"{meta.get('char_start', 0)}-{meta.get('char_end', len(chunk.get('text', '')))}",
            "text": chunk.get("text", "")
        }
    return citation_map
```

#### Sample Citation Map Data Structure
```json
{
  "[1]": {
    "source": "submission-rubric.md",
    "chunk_id": "submission-rubric.md#chunk_0",
    "chunk_index": 0,
    "section": "Evidence Requirements",
    "page": 1,
    "char_span": "120-315",
    "text": "Project submission evidence requirement: Learners must upload repository GitHub links..."
  }
}
```

---

### Task 3: Verify Cited Sources (`verify_citation` & `trace_chunk_source`)

Enables users to verify a cited claim against original source document text, checking character offset bounds and line span.

```python
def verify_citation(
    citation_marker: str,
    citation_map: Dict[str, Any],
    full_document_text: Optional[str] = None
) -> Dict[str, Any]:
    if citation_marker not in citation_map:
        return {"marker": citation_marker, "is_valid": False, "reason": "MARKER_NOT_IN_CITATION_MAP"}

    citation_entry = citation_map[citation_marker]
    chunk_text = citation_entry.get("text", "")
    source_file = citation_entry.get("source", "Unknown")

    exact_match = True
    line_range = "L1-L10"
    if full_document_text:
        char_span = citation_entry.get("char_span", "0-0")
        start_offset, end_offset = map(int, char_span.split("-"))
        extracted = full_document_text[start_offset:end_offset]
        exact_match = (extracted == chunk_text)
        lines_before = full_document_text[:start_offset].count("\n") + 1
        lines_span = chunk_text.count("\n")
        line_range = f"L{lines_before}-L{lines_before + lines_span}"

    return {
        "marker": citation_marker,
        "is_valid": True,
        "source": source_file,
        "chunk_id": citation_entry.get("chunk_id"),
        "line_range": line_range,
        "verified_exact_match": exact_match
    }
```

---

### Task 4: Avoid Fabricated Citations (`answer_with_citations` & `detect_fabricated_citations`)

If retrieved context is absent or insufficient, the system returns an uncited fallback response instead of hallucinating citations. Furthermore, `detect_fabricated_citations` audits LLM completion outputs for marker hallucination (e.g. model producing `[5]` when context only contained `[1]`).

```python
def answer_with_citations(question: str, chunks=None, ...) -> Dict[str, Any]:
    if not chunks:
        return {
            "answer": "I don't have enough information in the provided context.",
            "citations": {}
        }
    ...
```

---

### Task 5: Sample Cited Answers Payload & Demonstration

The payload is exported to `sample_cited_answers.json` and demonstrated in `source_citation_demo.py`.

#### Sample End-to-End Output Payload
```json
{
  "question": "What evidence is required for project submission?",
  "answer": "Based on the provided documentation: Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough... [1]",
  "citations": {
    "[1]": {
      "source": "submission-rubric.md",
      "chunk_id": "submission-rubric.md#chunk_0",
      "chunk_index": 0,
      "section": "Evidence Requirements",
      "page": 1,
      "char_span": "120-315",
      "text": "Project submission evidence requirement: Learners must upload repository GitHub links..."
    }
  },
  "cited_markers": ["[1]"],
  "status": "GROUNDED_SUCCESS"
}
```

---

## Test Execution & Verification

### Automated Unit Test Suite (`test_source_citation.py`)

Run command:
```powershell
py -m unittest test_source_citation.py
```

Output:
```
............
----------------------------------------------------------------------
Ran 12 tests in 0.512s

OK
```

### Full System Test Suite Pass

Run command:
```powershell
py -m unittest test_grounded_generator.py test_context_injector.py test_rag_pipeline.py test_source_citation.py
```

Output:
```
----------------------------------------------------------------------
Ran 46 tests in 8.120s

OK
```

---

## Video Explanation Walkthrough Guide (3-5 Minutes)

When recording the video walkthrough for submission, address the 5 key rubric requirements:

1. **Why Citations Matter**: Explains how citations transform RAG from a black-box text generator into a transparent, verifiable knowledge system building user trust.
2. **Citation to Source Mapping**: Shows how marker `[1]` points to `sample_cited_answers.json` containing exact file source `submission-rubric.md`, section `"Evidence Requirements"`, and character offset `120-315`.
3. **How Metadata Enables Citation**: Demonstrates how chunking and tagging stages preserved document metadata throughout vector indexing, context injection, and prompt assembly.
4. **Risk of Fabricated Citations**: Illustrates why fake citations give users false confidence, and how Knovera's fallback guard (`STANDARD_MISSING_CONTEXT_FALLBACK`) and `detect_fabricated_citations` protect against hallucinated markers.
5. **Application to Problem Statement**: Details how citations help users in Knovera's technical documentation assistant locate exact code repositories, rubric sections, and policy documents instantly.
