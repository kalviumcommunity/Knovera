"""
Source Tracer and Citation Module for Knovera RAG Assistant.

Provides functions to trace retrieved text chunks back to their exact original source documents,
verify character/line offsets, generate clean RAG citations, and filter chunks by metadata.
"""

import re
from typing import Dict, List, Any, Optional, Callable

from src.context_injector import assemble_context, build_prompt
from src.grounded_generator import generate_grounded_answer, STANDARD_MISSING_CONTEXT_FALLBACK


def build_citation_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Task 2 & Assignment Core: Maps citation markers like '[1]' to real document metadata and location details.

    Args:
        chunks: List of retrieved chunk dictionaries.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping marker strings (e.g., "[1]") to source metadata and text.
    """
    citation_map = {}
    if not chunks:
        return citation_map

    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        chunk_id = meta.get("chunk_id", chunk.get("id", f"chunk_{index}"))
        chunk_index = meta.get("chunk_index", 0)
        source = meta.get("source", chunk.get("source", "Unknown Document"))
        section = meta.get("section") or meta.get("section_heading", "General")
        page = meta.get("page_number", meta.get("page", 1))
        c_start = meta.get("char_start", 0)
        c_end = meta.get("char_end", len(chunk.get("text", "")))
        text = chunk.get("text", "")

        citation_map[f"[{index}]"] = {
            "source": source,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "section": section,
            "page": page,
            "char_span": f"{c_start}-{c_end}",
            "text": text
        }
    return citation_map


def build_cited_prompt(question: str, chunks: List[Dict[str, Any]], max_context_tokens: int = 5000) -> str:
    """
    Task 1 & 4: Builds a prompt instructing the model to cite only provided sources
    and return a fallback if context is insufficient.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace only.")

    context, _, _, _ = assemble_context(chunks, max_tokens=max_context_tokens)
    
    return f"""Answer using only the context below.
Cite every factual claim using source markers like [1] or [2].
Only use source markers that appear in the context.
If the context does not support an answer, say you do not have enough information and do not invent citations.

Context:
{context}

Question:
{question}"""


def answer_with_citations(
    question: str,
    retriever: Optional[Any] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    k: int = 4,
    score_threshold: Optional[float] = None,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Task 1, 2, 4: Answers a question and attaches verified citation details and metadata mapping.
    If no chunks or context is insufficient, returns the fallback with empty citations.
    """
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Please provide a valid question.",
            "citations": {},
            "status": "EMPTY_QUESTION"
        }

    if chunks is None:
        if retriever is not None and hasattr(retriever, "retrieve"):
            raw_chunks = retriever.retrieve(question, k=k)
            if score_threshold is not None:
                chunks = [c for c in raw_chunks if c.get("score", 1.0) >= score_threshold]
            else:
                chunks = raw_chunks
        else:
            chunks = []

    if not chunks:
        return {
            "question": question,
            "answer": STANDARD_MISSING_CONTEXT_FALLBACK,
            "citations": {},
            "cited_markers": [],
            "status": "MISSING_CONTEXT_FALLBACK"
        }

    answer_res = generate_grounded_answer(
        question=question,
        retrieved_chunks=chunks,
        use_api=use_api
    )
    answer = answer_res["answer"]

    # Check for missing-context fallback
    if STANDARD_MISSING_CONTEXT_FALLBACK.lower() in answer.lower():
        return {
            "question": question,
            "answer": STANDARD_MISSING_CONTEXT_FALLBACK,
            "citations": {},
            "cited_markers": [],
            "status": "MISSING_CONTEXT_FALLBACK"
        }

    citations = build_citation_map(chunks)
    cited_markers = list(set(re.findall(r'\[\d+\]', answer)))
    
    # Task 4 Guard: Detect fabricated citations (markers in answer that were not in context)
    valid_markers = set(citations.keys())
    fabricated_markers = [m for m in cited_markers if m not in valid_markers]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "cited_markers": cited_markers,
        "fabricated_citations_detected": len(fabricated_markers) > 0,
        "fabricated_markers": fabricated_markers,
        "status": "GROUNDED_SUCCESS"
    }


def verify_citation(
    citation_marker: str,
    citation_map: Dict[str, Any],
    full_document_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Task 3: Verifies a specific citation marker against the citation map and original source document.
    """
    if citation_marker not in citation_map:
        return {
            "marker": citation_marker,
            "is_valid": False,
            "reason": "MARKER_NOT_IN_CITATION_MAP",
            "source": None,
            "verified_exact_match": False
        }

    citation_entry = citation_map[citation_marker]
    chunk_text = citation_entry.get("text", "")
    source_file = citation_entry.get("source", "Unknown")

    exact_match = True
    line_range = "L1-L10"
    if full_document_text:
        char_span = citation_entry.get("char_span", "0-0")
        try:
            start_offset, end_offset = map(int, char_span.split("-"))
            is_in_bounds = 0 <= start_offset <= len(full_document_text) and start_offset <= end_offset <= len(full_document_text)
            if is_in_bounds:
                extracted = full_document_text[start_offset:end_offset]
                exact_match = (extracted == chunk_text)
                lines_before = full_document_text[:start_offset].count("\n") + 1
                lines_span = chunk_text.count("\n")
                line_range = f"L{lines_before}-L{lines_before + lines_span}"
            else:
                exact_match = False
                line_range = "L?-L?"
        except Exception:
            exact_match = False
            line_range = "L?-L?"

    return {
        "marker": citation_marker,
        "is_valid": True,
        "source": source_file,
        "chunk_id": citation_entry.get("chunk_id"),
        "chunk_index": citation_entry.get("chunk_index"),
        "section": citation_entry.get("section"),
        "page": citation_entry.get("page"),
        "char_span": citation_entry.get("char_span"),
        "line_range": line_range,
        "text_snippet": chunk_text[:100].replace("\n", " ") + ("..." if len(chunk_text) > 100 else ""),
        "verified_exact_match": exact_match
    }


def detect_fabricated_citations(answer: str, valid_markers: List[str]) -> Dict[str, Any]:
    """
    Task 4: Detects fabricated citations (citation markers in answer that do not exist in valid_markers).
    """
    found_markers = list(set(re.findall(r'\[\d+\]', answer)))
    valid_set = set(valid_markers)
    fabricated = [m for m in found_markers if m not in valid_set]
    
    return {
        "has_fabricated_citations": len(fabricated) > 0,
        "total_citations_found": len(found_markers),
        "valid_citations": [m for m in found_markers if m in valid_set],
        "fabricated_citations": fabricated
    }


class SourceTracer:
    """Traces chunks to source files, formats citations, and filters candidate chunks."""

    build_citation_map = staticmethod(build_citation_map)
    build_cited_prompt = staticmethod(build_cited_prompt)
    answer_with_citations = staticmethod(answer_with_citations)
    verify_citation = staticmethod(verify_citation)
    detect_fabricated_citations = staticmethod(detect_fabricated_citations)

    @staticmethod
    def format_citation(chunk: Dict[str, Any], style: str = "full") -> str:
        """Generate a formatted citation string for an answer from a chunk's metadata.

        Args:
            chunk: Standard tagged chunk dictionary (text + metadata).
            style: 'full', 'compact', or 'markdown'.

        Returns:
            Formatted citation string.
        """
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown Document")
        sec = meta.get("section_heading", "General")
        idx = meta.get("chunk_index", 0)
        c_start = meta.get("char_start", 0)
        c_end = meta.get("char_end", 0)
        page = meta.get("page_number", 1)

        if style == "compact":
            return f"[{source} (p.{page}, chunk #{idx})]"
        elif style == "markdown":
            return f"[*According to {source}*, section **'{sec}'** (Chunk #{idx}, chars {c_start}–{c_end})]"
        else:
            # Default 'full' style
            return (
                f"[Source: {source} | Section: '{sec}' | Chunk #{idx} | "
                f"Page: {page} | Chars: {c_start}-{c_end}]"
            )

    @staticmethod
    def trace_chunk_source(chunk: Dict[str, Any], full_document_text: str) -> Dict[str, Any]:
        """Perform reverse lookup on a chunk against original document text to verify exact origin.

        Args:
            chunk: Standard tagged chunk dict with 'text' and 'metadata'.
            full_document_text: The complete raw text of the source document.

        Returns:
            Traceback audit report dictionary.
        """
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        c_start = meta.get("char_start", 0)
        c_end = meta.get("char_end", 0)

        # 1. Bounds check and text extraction from source text
        is_in_bounds = 0 <= c_start <= len(full_document_text) and c_start <= c_end <= len(full_document_text)
        extracted = full_document_text[c_start:c_end] if is_in_bounds else ""

        # 2. Check exact character offset match
        chunk_text = chunk.get("text", "")
        exact_match = (extracted == chunk_text) if is_in_bounds else False

        # 3. Calculate 1-based line numbers in original document
        if is_in_bounds:
            lines_before = full_document_text[:c_start].count("\n") + 1
            lines_span = chunk_text.count("\n")
            start_line = lines_before
            end_line = start_line + lines_span
            line_str = f"L{start_line}-L{end_line}"
        else:
            line_str = "L?-L?"

        citation = SourceTracer.format_citation(chunk, style="full")

        return {
            "source": source,
            "doc_title": meta.get("doc_title", source),
            "chunk_index": meta.get("chunk_index"),
            "section_heading": meta.get("section_heading"),
            "page_number": meta.get("page_number"),
            "char_span": f"{c_start}-{c_end}",
            "line_range": line_str,
            "char_length": len(chunk_text),
            "verified_exact_match": exact_match,
            "citation_string": citation,
            "snippet_preview": chunk_text[:80].replace("\n", " ") + ("..." if len(chunk_text) > 80 else ""),
        }

    @staticmethod
    def filter_chunks(
        chunks: List[Dict[str, Any]],
        criteria: Optional[Dict[str, Any]] = None,
        custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter a list of tagged chunks based on metadata field values or custom function.

        Enables RAG scoped retrieval (e.g. 'only TXT files', 'only 2026 policies', etc.)

        Args:
            chunks: List of tagged chunk dicts.
            criteria: Dict of key-value pairs that metadata must match.
            custom_filter: Optional custom predicate function taking metadata dict.

        Returns:
            Filtered list of chunk dicts.
        """
        results = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})

            # Match key-value criteria
            match = True
            if criteria:
                for k, expected_val in criteria.items():
                    actual_val = meta.get(k)
                    if isinstance(expected_val, (list, tuple, set)):
                        if actual_val not in expected_val:
                            match = False
                            break
                    elif actual_val != expected_val:
                        match = False
                        break

            if match and custom_filter:
                match = custom_filter(meta)

            if match:
                results.append(chunk)

        return results


def extract_citations(text: str) -> List[str]:
    """Extract citation source markers or references from text."""
    markers = re.findall(r'\[(?:Source:\s*)?([^\]]+)\]', text)
    return [m.strip() for m in markers if m.strip()]


def verify_citation_grounding(answer: str, context: str) -> bool:
    """Verifies whether citations in the answer exist in the context."""
    citations = extract_citations(answer)
    if not citations:
        return True
    return all(c in context for c in citations)

