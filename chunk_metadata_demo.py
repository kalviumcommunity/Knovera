"""
Comprehensive Demonstration Script for Chunk Metadata & Source Tracking.

Covers Tasks 1 through 5:
- Task 1: Storing source document identifiers across multi-format files (.txt, .md, .html, .pdf).
- Task 2: Attaching section headings, page numbers, character spans, titles, and dates.
- Task 3: Enforcing a consistent metadata schema dictionary structure across all chunks.
- Task 4: Tracing retrieved chunks to exact source locations with offset & line-number audit verification.
- Task 5: Demonstrating citation string generation, metadata-driven retrieval filtering, and JSON export.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from src.document_loader import DocumentLoader
from src.chunk_tagger import ChunkTagger, extract_sections_and_metadata
from src.source_tracer import SourceTracer
from chunking import paragraph_chunking_with_offsets, fixed_size_chunking_with_offsets


def process_and_tag_corpus(data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load multi-format document corpus and tag all chunks with consistent metadata."""
    loader = DocumentLoader(verbose=False)
    tagger = ChunkTagger()

    all_tagged_chunks: List[Dict[str, Any]] = []
    document_raw_texts: Dict[str, str] = {}

    target_files = [
        data_dir / "customer_policy.txt",
        data_dir / "api_documentation.md",
        data_dir / "release_notes.html",
        data_dir / "employee_handbook.pdf",
    ]

    for file_path in target_files:
        if not file_path.exists():
            continue

        doc = loader.load_document(file_path)
        raw_text = doc["text"]
        source_name = doc["source"]
        file_type = doc["file_type"]
        document_raw_texts[source_name] = raw_text

        # Extract title, effective date, and section offset map
        doc_title, effective_date, section_map = extract_sections_and_metadata(
            raw_text, file_type=file_type, source_name=source_name
        )

        # Handle PDF page mapping if PDF format
        page_map = None
        if file_type == ".pdf":
            # Simple page map simulation based on page text length or paragraph breaks
            page_map = [(0, 1)]  # page 1 starts at index 0

        # Chunk the document using paragraph-based chunking with offsets
        chunk_tuples = paragraph_chunking_with_offsets(raw_text)

        # Fallback to fixed size if paragraph chunking yielded 1 big chunk
        if len(chunk_tuples) <= 1 and len(raw_text) > 300:
            chunk_tuples = fixed_size_chunking_with_offsets(raw_text, chunk_size=250, overlap=50)

        # Tag all chunks with consistent metadata schema
        tagged_chunks = tagger.tag_chunks_from_tuples(
            source=source_name,
            chunk_tuples=chunk_tuples,
            file_type=file_type,
            doc_title=doc_title,
            section_map=section_map,
            page_map=page_map,
            effective_date=effective_date,
        )

        all_tagged_chunks.extend(tagged_chunks)

    return all_tagged_chunks, document_raw_texts


def verify_metadata_schema_consistency(chunks: List[Dict[str, Any]]) -> bool:
    """Verify that every chunk across the corpus shares the exact same metadata keys."""
    if not chunks:
        return False

    first_keys = set(chunks[0]["metadata"].keys())
    for idx, chunk in enumerate(chunks):
        current_keys = set(chunk["metadata"].keys())
        if current_keys != first_keys:
            print(f"[SCHEMA ERROR] Chunk #{idx} from '{chunk['metadata']['source']}' schema mismatch!")
            print(f"  Missing: {first_keys - current_keys}, Extra: {current_keys - first_keys}")
            return False

    return True


def main():
    data_dir = Path("data")
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    print("================================================================================")
    print("      KNOVERA RAG ASSISTANT - CHUNK METADATA & SOURCE TRACKING DEMO            ")
    print("================================================================================\n")

    # 1. Process Corpus and Tag Chunks
    all_chunks, raw_texts = process_and_tag_corpus(data_dir)
    print(f"Loaded and tagged {len(all_chunks)} total chunks across multi-format document corpus.\n")

    # 2. Verify Metadata Schema Consistency (Task 3)
    is_consistent = verify_metadata_schema_consistency(all_chunks)
    print("--- TASK 3: METADATA SCHEMA CONSISTENCY AUDIT ---")
    print(f"Consistent Schema Enforced Across All Chunks: {is_consistent}")
    sample_meta = all_chunks[0]["metadata"]
    print(f"Standard Metadata Keys ({len(sample_meta.keys())}): {list(sample_meta.keys())}\n")

    # 3. Display Multi-Format Chunk Samples with Metadata (Task 1 & Task 2)
    print("--- TASK 1 & 2: MULTI-FORMAT CHUNKS WITH SOURCE & EXTRA METADATA ---")
    source_samples = {}
    for chunk in all_chunks:
        src = chunk["metadata"]["source"]
        if src not in source_samples:
            source_samples[src] = chunk

    for src, sample_chunk in source_samples.items():
        print(f"\n[DOCUMENT SOURCE: {src}]")
        print(f"Text Snippet : {sample_chunk['text'][:110]!r}...")
        print("Metadata Dict:")
        print(json.dumps(sample_chunk["metadata"], indent=2))
        print(f"Formatted Citation: {SourceTracer.format_citation(sample_chunk)}")

    # 4. Source Traceback Verification (Task 4)
    print("\n--------------------------------------------------------------------------------")
    print("--- TASK 4: RETRIEVED CHUNK SOURCE TRACEBACK VERIFICATION ---")
    print("--------------------------------------------------------------------------------")

    # Pick 3 representative chunks from different documents to trace
    retrieved_candidates = [
        all_chunks[1] if len(all_chunks) > 1 else all_chunks[0],  # Customer Policy
        next((c for c in all_chunks if c["metadata"]["file_type"] == ".md"), all_chunks[0]),  # API Docs
        next((c for c in all_chunks if c["metadata"]["file_type"] == ".html"), all_chunks[0]),  # HTML Release Notes
    ]

    for idx, cand in enumerate(retrieved_candidates, 1):
        src_name = cand["metadata"]["source"]
        orig_text = raw_texts.get(src_name, "")
        trace_report = SourceTracer.trace_chunk_source(cand, orig_text)

        print(f"\nRetrieved Candidate #{idx} [{src_name}]:")
        print(f"  Doc Title            : {trace_report['doc_title']}")
        print(f"  Section Heading      : {trace_report['section_heading']}")
        print(f"  Chunk Index          : #{trace_report['chunk_index']}")
        print(f"  Character Offsets    : {trace_report['char_span']}")
        print(f"  Original Line Range  : {trace_report['line_range']}")
        print(f"  Exact Match Audit    : {'PASSED' if trace_report['verified_exact_match'] else 'FAILED'}")
        print(f"  RAG Citation String  : {trace_report['citation_string']}")
        print(f"  Text Content Preview : {trace_report['snippet_preview']!r}")

    # 5. Metadata-Driven Retrieval Scoping / Filtering Demonstration
    print("\n--------------------------------------------------------------------------------")
    print("--- METADATA-DRIVEN RETRIEVAL FILTERING DEMONSTRATION ---")
    print("--------------------------------------------------------------------------------")

    txt_chunks = SourceTracer.filter_chunks(all_chunks, criteria={"file_type": ".txt"})
    policy_chunks = SourceTracer.filter_chunks(
        all_chunks,
        custom_filter=lambda m: m.get("section_heading") and "Refund" in m["section_heading"],
    )

    print(f"Filter 1: Only '.txt' file chunks -> Found {len(txt_chunks)} chunks.")
    print(f"Filter 2: Section heading containing 'Refund' -> Found {len(policy_chunks)} chunks.")
    if policy_chunks:
        print(f"  Matched Chunk Source: {policy_chunks[0]['metadata']['source']}")
        print(f"  Matched Section     : {policy_chunks[0]['metadata']['section_heading']}")
        print(f"  Matched Text        : {policy_chunks[0]['text']!r}")

    # 6. Save JSON and Text Outputs (Task 5)
    sample_tagged_json_path = Path("sample_tagged_chunks.json")
    output_log_path = outputs_dir / "chunk_metadata_output.txt"

    # Export sample tagged chunks grouped by source
    export_data = {
        "metadata_schema": list(sample_meta.keys()),
        "total_corpus_chunks": len(all_chunks),
        "schema_consistency_verified": is_consistent,
        "sample_tagged_chunks": [
            {
                "source": chunk["metadata"]["source"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "citation": SourceTracer.format_citation(chunk),
            }
            for chunk in all_chunks[:6]
        ],
    }

    with open(sample_tagged_json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\nSaved sample tagged chunks to '{sample_tagged_json_path}'.")

    # Generate log output
    with open(output_log_path, "w", encoding="utf-8") as f:
        f.write("=== KNOVERA CHUNK METADATA & SOURCE TRACKING EXECUTION OUTPUT ===\n\n")
        f.write(f"Total Corpus Chunks Processed: {len(all_chunks)}\n")
        f.write(f"Metadata Schema Consistent Across Corpus: {is_consistent}\n")
        f.write(f"Standard Keys: {list(sample_meta.keys())}\n\n")

        f.write("--- SAMPLE TAGGED CHUNKS ACROSS FORMATS ---\n")
        for chunk in all_chunks[:4]:
            f.write(f"Source: {chunk['metadata']['source']}\n")
            f.write(f"Metadata: {json.dumps(chunk['metadata'])}\n")
            f.write(f"Citation: {SourceTracer.format_citation(chunk)}\n")
            f.write(f"Text: {chunk['text'][:120]}...\n\n")

        f.write("--- SOURCE TRACEBACK AUDIT RESULTS ---\n")
        for idx, cand in enumerate(retrieved_candidates, 1):
            trace = SourceTracer.trace_chunk_source(cand, raw_texts.get(cand["metadata"]["source"], ""))
            f.write(f"Candidate #{idx} ({trace['source']}):\n")
            f.write(f"  Section: {trace['section_heading']}\n")
            f.write(f"  Line Range: {trace['line_range']} ({trace['char_span']})\n")
            f.write(f"  Exact Match: {trace['verified_exact_match']}\n")
            f.write(f"  Citation: {trace['citation_string']}\n\n")

    print(f"Saved complete execution log to '{output_log_path}'.")


if __name__ == "__main__":
    main()
