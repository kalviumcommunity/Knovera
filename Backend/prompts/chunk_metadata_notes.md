# Chunk Metadata & Source Tracking in RAG Systems

## Executive Summary

In Retrieval-Augmented Generation (RAG) architectures, floating text chunks without metadata lead to hallucinated citations, untraceable model outputs, and poor context precision. The **Chunk Metadata & Source Tracking** engine ensures that every text chunk is paired with a consistent metadata dictionary at ingestion time. This enables exact source attribution, character offset verification, RAG answer citation formatting, and metadata-driven pre/post retrieval filtering.

---

## Key Concepts & Task Solutions

### Task 1 — Store the Source Identifier
- **Requirement**: Ensure each chunk stores its source document identifier (e.g. filename, document ID).
- **Implementation**: The `source` key is stored on every chunk (e.g. `"customer_policy.txt"`, `"api_documentation.md"`). Without `source`, a RAG system cannot attribute retrieved information back to its origin file.

### Task 2 — Attach Additional Metadata
- **Requirement**: Attach additional structural and document metadata (section headings, page numbers, position indices, titles, dates).
- **Implementation**:
  - **Position/Offsets**: `chunk_index` (0-based order), `char_start` (start offset in full doc text), `char_end` (end offset in full doc text), `char_length`.
  - **Structural Context**: `section_heading` (dynamically extracted from `#` headings, `<h2>` tags, or numbered section headers like `1. Overview`), `page_number` (page number for PDFs).
  - **Document Context**: `doc_title` (extracted top title), `file_type` (`.txt`, `.md`, `.html`, `.pdf`), `effective_date` (extracted date string if present).

### Task 3 — Consistent Structure Across Corpus
- **Requirement**: Keep metadata alongside chunk text in a consistent structure across all files.
- **Rule**: *"A chunk is text plus metadata. Never store a chunk as a bare string."*
- **Schema**:
```json
{
  "text": "1. Overview\nKnovera is committed to delivering high-quality enterprise support...",
  "metadata": {
    "source": "customer_policy.txt",
    "chunk_index": 1,
    "char_start": 101,
    "char_end": 317,
    "char_length": 216,
    "file_type": ".txt",
    "doc_title": "KNOVERA CUSTOMER SERVICE & REFUND POLICY",
    "section_heading": "1. Overview",
    "page_number": 1,
    "effective_date": "January 15, 2026"
  }
}
```
Every chunk across `.txt`, `.md`, `.html`, and `.pdf` files shares the exact same 10 standard metadata keys, ensuring zero key errors downstream.

### Task 4 — Trace a Chunk to its Exact Source
- **Requirement**: Demonstrate that a retrieved chunk can be traced back to its exact source using metadata.
- **Implementation**:
  - `SourceTracer.trace_chunk_source(chunk, full_text)` extracts `full_text[char_start:char_end]` and performs an exact string equality audit (`verified_exact_match: True`).
  - Computes exact 1-based line number ranges (`L5-L6`) in original source document.
  - Generates standard answer citations: `"[Source: customer_policy.txt | Section: '1. Overview' | Chunk #1 | Page: 1 | Chars: 101-317]"`.

### Task 5 — Commit with Sample Chunks
- Metadata-tagging engine (`src/chunk_tagger.py`), source tracer (`src/source_tracer.py`), updated `chunking.py`, demo script (`chunk_metadata_demo.py`), unit tests (`test_chunk_metadata.py`), `sample_tagged_chunks.json`, and execution log (`outputs/chunk_metadata_output.txt`).

---

## Why Metadata Powers Citations and Filtering

### 1. Citation Generation at Answer Time
When retrieval returns candidate chunks, their metadata travels directly into the prompt context or answer formatting function. The model or post-processor can cite:
> *"According to customer_policy.txt, section '3. Customer Refund & Cancellation Criteria', full refunds are eligible within 30 days."*

### 2. Retrieval Filtering & Scoping (Concept 24 Preview)
Metadata empowers structured vector database queries (e.g. metadata filtering before vector similarity search):
- **Temporal Scoping**: `"only policies with effective_date >= 2026-01-01"`
- **Format Scoping**: `"only file_type == '.pdf'"`
- **Section Scoping**: `"only section_heading == 'Refund Terms'"`

---

## 3–5 Minute Video Explanation Script

### Slide / Topic Breakdown for Video Recording:

1. **Introduction (0:00 - 0:45)**:
   - Introduce yourself and the assignment **3.22 Chunk Metadata & Source Tracking**.
   - **The Problem**: Bare text strings lose all context once chunked and embedded. The RAG model cannot cite sources or verify claims without document attribution.
   - **The Solution**: Pair every chunk with a structured metadata dictionary at chunking time.

2. **Metadata Attached to Each Chunk (0:45 - 1:45)**:
   - Walk through `src/chunk_tagger.py` and show `sample_tagged_chunks.json`.
   - Explain the mandatory `source` field and additional fields (`chunk_index`, `char_start`, `char_end`, `section_heading`, `page_number`, `doc_title`, `file_type`, `effective_date`).
   - Highlight **Task 3**: Enforcing a consistent 10-key schema across TXT, MD, HTML, and PDF formats.

3. **Tracing a Chunk to its Source & Citations (1:45 - 2:45)**:
   - Demonstrate running `chunk_metadata_demo.py`.
   - Show how `SourceTracer.trace_chunk_source` verifies `char_start:char_end` against original source files, calculating exact line ranges (`L5-L6`) and returning `verified_exact_match: True`.
   - Show formatted citation links: `[Source: customer_policy.txt | Section: '1. Overview' | Chunk #1 | Page: 1 | Chars: 101-317]`.

4. **Future Filtering & Problem Statement Metadata (2:45 - 4:00)**:
   - Explain how metadata supports pre/post retrieval filtering (e.g., scoping search to 2026 policies or `.txt` docs).
   - **Problem Statement Follow-up**: *"What metadata would your problem statement need?"*
     - *Answer*: For enterprise customer support & policy RAG systems (Knovera), essential metadata includes **Document ID** (e.g. `POL-2026-089`), **Effective Date** (versioning to prevent obsolete policy retrieval), **Access Tier / Permissions** (Confidential vs. Public), **Section Heading** (domain context), and **Source URL / Path**.
