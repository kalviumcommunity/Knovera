"""
Source Tracer and Citation Module for Knovera RAG Assistant.

Provides functions to trace retrieved text chunks back to their exact original source documents,
verify character/line offsets, generate clean RAG citations, and filter chunks by metadata.
"""

from typing import Dict, List, Any, Optional, Callable


class SourceTracer:
    """Traces chunks to source files, formats citations, and filters candidate chunks."""

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
