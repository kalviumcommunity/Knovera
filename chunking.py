"""
Chunking Strategies & Metadata Tagging Engine for Knovera RAG Assistant.

Provides fixed-size with overlap and paragraph chunking with exact character offset tracking,
consistent metadata dictionary tagging, and stats reporting.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from src.chunk_tagger import ChunkTagger, extract_sections_and_metadata
from src.source_tracer import SourceTracer
from src.token_chunker import TokenChunker


def token_chunks(text: str, size: int = 400, overlap: int = 60, encoding_name: str = "cl100k_base") -> List[str]:
    """Token-aware chunking: splits text by token counts using tiktoken with controlled overlap."""
    chunker = TokenChunker(encoding_name=encoding_name)
    return chunker.token_chunks(text, size=size, overlap=overlap)


def token_chunks_with_offsets(
    text: str, size: int = 400, overlap: int = 60, encoding_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    """Token-aware chunking with token offset and span tracking."""
    chunker = TokenChunker(encoding_name=encoding_name)
    return chunker.token_chunks_with_offsets(text, size=size, overlap=overlap)


def fixed_size_chunking_with_offsets(text: str, chunk_size: int, overlap: int) -> List[Tuple[str, int, int]]:
    """Split text into fixed-size chunks with overlap and track character start/end offsets."""
    chunks_with_offsets = []
    i = 0
    text_len = len(text)

    while i < text_len:
        start_pos = i
        end_pos = min(i + chunk_size, text_len)
        chunk = text[start_pos:end_pos]

        chunks_with_offsets.append((chunk, start_pos, end_pos))

        i += (chunk_size - overlap)
        if i >= text_len:
            break

    return chunks_with_offsets


def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Task 1 backward-compatible fixed-size chunking (returns raw strings)."""
    return [c[0] for c in fixed_size_chunking_with_offsets(text, chunk_size, overlap)]


def paragraph_chunking_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Split text into paragraph-based chunks and track exact character start/end offsets."""
    chunks_with_offsets = []
    current_pos = 0

    # Split by double newline while tracking character offsets
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        stripped_para = para.strip()
        if stripped_para:
            # Locate exact position of paragraph in original text
            start_pos = text.find(para, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            # Offset for stripped text leading spaces
            leading_whitespace = len(para) - len(para.lstrip())
            exact_start = start_pos + leading_whitespace
            exact_end = exact_start + len(stripped_para)

            chunks_with_offsets.append((stripped_para, exact_start, exact_end))
            current_pos = start_pos + len(para) + 2  # plus '\n\n'

    return chunks_with_offsets


def paragraph_chunking(text: str) -> List[str]:
    """Task 1 backward-compatible paragraph chunking (returns raw strings)."""
    return [c[0] for c in paragraph_chunking_with_offsets(text)]


def tag_chunks(
    source: str,
    chunks: List[Tuple[str, int]],
    file_type: str = ".txt",
    doc_title: Optional[str] = None,
    section_heading: Optional[str] = None,
    effective_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Task Requirement: Never store a chunk as a bare string. Pair it with a metadata dict in a consistent shape.

    Accepts tuples of (text, char_start) or (text, char_start, char_end).
    """
    tagger = ChunkTagger()
    tagged = []

    for i, item in enumerate(chunks):
        if len(item) == 3:
            c, pos, end_pos = item
        else:
            c, pos = item
            end_pos = pos + len(c)

        chunk_dict = tagger.tag_chunk(
            text=c,
            source=source,
            chunk_index=i,
            char_start=pos,
            char_end=end_pos,
            file_type=file_type,
            doc_title=doc_title,
            section_heading=section_heading,
            effective_date=effective_date,
        )
        tagged.append(chunk_dict)

    return tagged


def get_stats(chunks: List[Any]) -> Dict[str, Any]:
    """Compute statistics for raw string chunks or tagged chunk objects."""
    if not chunks:
        return {"count": 0, "avg_size": 0}

    first = chunks[0]
    if isinstance(first, dict) and "text" in first:
        sizes = [len(c["text"]) for c in chunks]
    elif isinstance(first, tuple):
        sizes = [len(c[0]) for c in chunks]
    else:
        sizes = [len(c) for c in chunks]

    total_size = sum(sizes)
    return {
        "count": len(chunks),
        "avg_size": round(total_size / len(chunks), 2),
    }


def main():
    # Read sample policy document
    file_path = Path("data/customer_policy.txt")
    try:
        document_text = file_path.read_text(encoding="utf-8")
        source_name = file_path.name
    except FileNotFoundError:
        document_text = "KNOVERA CUSTOMER SERVICE POLICY\nDocument ID: POL-2026-089\nEffective Date: January 15, 2026\n\n1. Overview\nKnovera enterprise support."
        source_name = "customer_policy.txt"

    # Extract metadata attributes and section map
    doc_title, effective_date, section_map = extract_sections_and_metadata(
        document_text, file_type=".txt", source_name=source_name
    )

    # Chunk with character offsets
    fixed_tuples = fixed_size_chunking_with_offsets(document_text, chunk_size=200, overlap=50)
    para_tuples = paragraph_chunking_with_offsets(document_text)

    # Tag chunks with consistent metadata
    tagger = ChunkTagger()
    fixed_tagged = tagger.tag_chunks_from_tuples(
        source=source_name,
        chunk_tuples=fixed_tuples,
        file_type=".txt",
        doc_title=doc_title,
        section_map=section_map,
        effective_date=effective_date,
    )
    para_tagged = tagger.tag_chunks_from_tuples(
        source=source_name,
        chunk_tuples=para_tuples,
        file_type=".txt",
        doc_title=doc_title,
        section_map=section_map,
        effective_date=effective_date,
    )

    print("=== Chunk Metadata & Source Tracking Demonstration ===")
    print(f"Document Source: {source_name}")
    print(f"Title: {doc_title} | Effective Date: {effective_date}\n")

    print(f"1. Fixed-Size Chunking (Size: 200, Overlap: 50)")
    print(f"   Count: {len(fixed_tagged)} tagged chunks")
    print(f"   Sample Chunk #0 Metadata: {json.dumps(fixed_tagged[0]['metadata'], indent=2)}\n")

    print(f"2. Paragraph Chunking")
    print(f"   Count: {len(para_tagged)} tagged chunks")
    print(f"   Sample Chunk #0 Citation: {SourceTracer.format_citation(para_tagged[0])}\n")

    # Trace retrieved chunk back to source
    trace_res = SourceTracer.trace_chunk_source(para_tagged[1], document_text)
    print("=== Source Traceback Verification ===")
    print(f"Retrieved Chunk Index : #{trace_res['chunk_index']}")
    print(f"Section Heading       : {trace_res['section_heading']}")
    print(f"Character Span        : {trace_res['char_span']}")
    print(f"Line Range            : {trace_res['line_range']}")
    print(f"Exact Match Verified  : {trace_res['verified_exact_match']}")
    print(f"Citation Link         : {trace_res['citation_string']}\n")

    # Save tagged sample output for Task 5
    sample_output = {
        "fixed_size_strategy": {
            "stats": get_stats(fixed_tagged),
            "sample_chunks": fixed_tagged[:3],
        },
        "paragraph_strategy": {
            "stats": get_stats(para_tagged),
            "sample_chunks": para_tagged[:3],
        },
    }

    with open("sample_chunks.json", "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)

    print("Tagged sample chunks saved to 'sample_chunks.json' for review.")


if __name__ == "__main__":
    main()
