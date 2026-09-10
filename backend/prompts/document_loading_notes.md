# Document Loading & Multi-Format Intake Notes (Assignment 3.19)

## 1. Why Documents Must Be Converted to a Common Text Form

In a Retrieval-Augmented Generation (RAG) architecture, the core components — tokenizers, embedding models, vector indices, and Large Language Models (LLMs) — process text strings and tokens. 

Raw documents arrive in diverse formats:
- **PDFs**: Complex layout files storing text streams, graphics commands, and embedded font mappings.
- **HTML**: Structured documents containing layout markup, scripts, and styling tags.
- **Markdown & Plain Text**: Unstructured or lightly marked-up text files.

Before chunking, embedding, or retrieving any document, all intake sources must be normalized into clean plain text. Converting all documents to a common string representation ensures:
1. **Uniform Processing**: Downstream chunking splitters can split text based on character counts or semantic boundaries without handling format-specific syntax.
2. **Clean Vector Embeddings**: Embedding models represent semantic content. Markup tags (like `<div>`, `<script>`, or PDF streams) introduce noise into vector embeddings, reducing retrieval precision.
3. **Optimized Context Windows**: Stripping formatting boilerplate minimizes token overhead when passing retrieved context to the LLM.

---

## 2. Challenges of Extracting Text: PDFs vs Plain Text & HTML

| Format | Extraction Complexity | Key Technical Challenges | Solutions / Approaches |
|---|---|---|---|
| **Plain Text (`.txt`, `.md`)** | Low | Character encoding mismatches (e.g., UTF-8 vs ASCII vs ISO-8859-1). | Standard UTF-8 reading with `errors="ignore"` or fallback encoding detection. |
| **HTML (`.html`, `.htm`)** | Medium | Irrelevant DOM nodes (navigation headers, footers, inline JavaScript, CSS blocks). | HTML parsing using `BeautifulSoup` to decompose script/style tags and extract visible body text. |
| **PDF (`.pdf`)** | High | 1. **Scattered Text Streams**: Words stored out of reading order.<br>2. **Multi-Column Layouts**: Text read horizontally across columns.<br>3. **Scanned Documents**: Pages saved as raster images with no extractable text.<br>4. **Corrupt Header/EOF**: Truncated PDF downloads. | 1. Use `pypdf.PdfReader` page-by-page text extraction.<br>2. Fallback to OCR (`pytesseract` / `pdf2image`) for zero-text scanned pages.<br>3. Catch `PdfReadError` to skip corrupt files. |

---

## 3. Preserving Source Identity Metadata for Citations

When a RAG pipeline ingests a document, it must attach metadata to enable **source attribution** and **verifiable citations** in model responses.

In Knovera's `DocumentLoader`, every ingested file produces a standardized dictionary object:
```python
{
    "source": "customer_policy.txt",                # Filename identifier for citations
    "path": "D:/Project/Knovera/data/customer_policy.txt", # Full file path
    "file_type": ".txt",                            # Document format extension
    "char_count": 992,                             # Extracted text length
    "text": "KNOVERA CUSTOMER SERVICE & REFUND POLICY..." # Plain text payload
}
```

When documents are later chunked, each chunk inherits the parent document's `source` identifier. When the LLM generates an answer, the RAG client maps the retrieved chunk back to its source filename (e.g., `[Source: customer_policy.txt]`).

---

## 4. Resilient Error Handling & Bad Input Survival

In a enterprise corpus of 4,000+ files, input data is inherently dirty:
- Files may be missing or deleted mid-run.
- PDF downloads may be truncated or corrupted.
- Files may have unsupported extensions (e.g., `.bin`, `.exe`, `.xyz`).

### Failure Survival Strategy:
- **Per-File Scoped Scans**: The loader processes each file inside an isolated `try...except` block within `load_corpus()`.
- **Explicit Error Logging**: Unreadable files log a `SKIP` message detailing the file name and specific error (e.g., `PdfReadError`, `ValueError`, `FileNotFoundError`).
- **Pipeline Continuity**: One bad file never aborts the ingestion run. The remaining 3,999 documents continue loading smoothly.
- **Summary Auditing**: A final intake summary reports `total_scanned`, `total_successful`, and `total_skipped` alongside specific failure reasons for administrative review.

---

## 5. Mapping to the Knovera Enterprise Corpus

Knovera's real-world internal documentation corpus consists of ~4,000 files:
1. **Format Composition**:
   - ~60% PDF files (Employee handbooks, formal compliance policies, vendor contracts).
   - ~25% HTML exports (Internal wiki pages, Confluence documentation).
   - ~15% Plain Text & Markdown files (Technical READMEs, system configuration notes).
2. **Extraction Strategy**:
   - **PDFs**: `pypdf` handles 90%+ of digital PDFs; scanned legacy PDFs are flagged for OCR processing.
   - **HTML**: `BeautifulSoup` strips site headers/footers to preserve core policy documentation text.
   - **Markdown/TXT**: Directly loaded into UTF-8 text.

---

## 6. Video Walkthrough Outline (3–5 Minutes)

Use this structured script outline when recording your video walkthrough submission:

1. **Introduction (0:00 - 0:45)**:
   - State your name and project (Knovera RAG Assistant - Document Loading & Intake).
   - Explain why documents must be converted to plain text before chunking and embedding.
2. **Multi-Format Loader Architecture (0:45 - 1:45)**:
   - Show `src/document_loader.py`.
   - Explain how PDF (`pypdf`), HTML (`BeautifulSoup`), and TXT/MD files are loaded into a common plain-text payload.
   - Highlight how `source` identity (filename, path, file_type) is attached to every loaded document.
3. **Resilient Failure Handling Demo (1:45 - 2:45)**:
   - Run `python document_loading_demo.py` in the terminal.
   - Point out how missing files, corrupt PDFs (`corrupt_file.pdf`), and unsupported formats (`unsupported_file.xyz`) are caught gracefully with `SKIP` messages without crashing the run.
   - Point out the intake confirmation preview: character length and text sample for each loaded file.
4. **Mapping to Knovera Corpus & Conclusion (2:45 - 3:30)**:
   - Summarize how this loader handles Knovera's 4,000-document mixed corpus.
   - Conclude and confirm public GitHub PR link and video link preparation.
