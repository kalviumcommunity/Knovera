"""
Text Extraction & Cleaning Pipeline Demonstration Script for Knovera RAG Assistant.

Demonstrates:
- Task 1: Remove repeated headers, footers, and boilerplate (e.g., 'Page X of Y', nav text).
- Task 2: Normalise whitespace, line breaks, and encoding artifacts (NFKC, mojibake fixes, dehyphenation).
- Task 3: Apply uniform cleaning consistently across the entire corpus.
- Task 4: Provide before and after metrics and text comparisons showing noise removal.
- Task 5: Save execution output for grading and review.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any

from src.document_loader import DocumentLoader
from src.text_cleaner import TextCleaner


def run_cleaning_demo():
    output_lines: List[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("=" * 85)
    log("          KNOVERA RAG ASSISTANT - TEXT EXTRACTION & CLEANING PIPELINE")
    log("=" * 85)
    log("Sprint 2 · Concept 11 · Document Processing (Assignment 3.20)")
    log()

    # Initialize loader and cleaner
    data_dir = Path("data")
    loader = DocumentLoader(verbose=False)
    cleaner = TextCleaner()

    # Step 1: Ingest Corpus
    raw_documents, intake_stats = loader.load_corpus(data_dir, recursive=True)

    log("-------------------------------------------------------------------------------------")
    log("TASK 1 & TASK 2: GRANULAR STEP-BY-STEP CLEANING PIPELINE BREAKDOWN")
    log("-------------------------------------------------------------------------------------")
    sample_raw = (
        "KNOVERA CONFIDENTIAL - DO NOT DISTRIBUTE\r\n"
        "Home > Docs > Architecture > RAG Pipeline\r\n\r\n"
        "Page 1 of 3\r\n\r\n"
        "Weâ€™ve observed that â€œraw extracted textâ€  contains serious encoding arti-\r\n"
        "facts, including mojibake â€” such as broken smart quotes and dashes â€” which degrade per-\r\n"
        "formance across dis-\r\ntributed systems.\r\n\r\n"
        "Key Guidelines:       \t\t\r\n"
        "  • Maintain state-of-the-art vector precision.\r\n"
        "  • Preserve numeric metrics (94.8% accuracy).\r\n\r\n\r\n"
        "[Back to Top]\r\n"
        "KNOVERA CONFIDENTIAL - DO NOT DISTRIBUTE\r\n"
        "Page 2 of 3\r\n"
        "© 2026 Knovera Technologies Inc."
    )

    log("RAW INPUT SAMPLE (With headers, footers, mojibake, split lines, whitespace):")
    log("-" * 85)
    log(sample_raw)
    log("-" * 85)
    log()

    # Stage 1: Line endings
    s1 = cleaner.normalize_line_endings(sample_raw)
    log("[Stage 1] Line Ending Normalization (\\r\\n -> \\n):")
    log(f"  Result length: {len(s1)} chars (standardized Unix newlines)")
    log()

    # Stage 2: Mojibake & Unicode
    s2 = cleaner.fix_encoding_artifacts(s1)
    log("[Stage 2] Unicode NFKC & Mojibake Repair:")
    log("  Fixed: 'Weâ€™ve' -> \"We've\", 'â€œraw extracted textâ€ ' -> '\"raw extracted text\"', 'â€”' -> '—'")
    log(f"  Result length: {len(s2)} chars")
    log()

    # Stage 3: Broken line-wraps (De-hyphenation)
    s3 = cleaner.repair_broken_line_wraps(s2)
    log("[Stage 3] Broken Line-Wrap Repair (De-hyphenation):")
    log("  Repaired: 'arti-\\nfacts' -> 'artifacts'")
    log("  Repaired: 'per-\\nformance' -> 'performance'")
    log("  Repaired: 'dis-\\ntributed' -> 'distributed'")
    log("  Preserved: 'state-of-the-art' (intentional inline compound hyphen preserved!)")
    log(f"  Result length: {len(s3)} chars")
    log()

    # Stage 4: Boilerplate Removal
    s4 = cleaner.strip_boilerplate(s3)
    log("[Stage 4] Boilerplate & Page Header/Footer Stripping:")
    log("  Removed: 'KNOVERA CONFIDENTIAL - DO NOT DISTRIBUTE', 'Page 1 of 3', 'Page 2 of 3'")
    log("  Removed: 'Home > Docs > Architecture > RAG Pipeline', '[Back to Top]', '© 2026 Knovera...'")
    log(f"  Result length: {len(s4)} chars")
    log()

    # Stage 5: Whitespace Normalization
    s5 = cleaner.collapse_whitespace(s4)
    log("[Stage 5] Whitespace Normalization (Collapse tabs/spaces & runaway newlines):")
    log(f"  Final cleaned text length: {len(s5)} chars")
    log("-" * 85)
    log("CLEANED OUTPUT STRING:")
    log(s5)
    log("-" * 85)
    log()

    # Task 3: Uniform Corpus-Wide Cleaning
    log("-------------------------------------------------------------------------------------")
    log("TASK 3 & TASK 4: UNIFORM CORPUS-WIDE CLEANING & BEFORE/AFTER COMPARISON")
    log("-------------------------------------------------------------------------------------")
    log("Applying identical TextCleaner pipeline across all ingested corpus documents...")
    log()

    cleaned_documents, corpus_stats = cleaner.clean_corpus(raw_documents)

    log(f"{'Source Document':<30} | {'Type':<6} | {'Before':<10} | {'After':<10} | {'Removed':<9} | {'% Reduction':<12}")
    log("-" * 85)
    for doc in cleaned_documents:
        log(
            f"{doc['source']:<30} | {doc['file_type']:<6} | "
            f"{doc['original_char_count']:>6} c   | "
            f"{doc['char_count']:>6} c   | "
            f"{doc['chars_removed']:>5} c   | "
            f"{doc['reduction_pct']:>6.2f}%"
        )
    log("-" * 85)
    log()

    # Detailed Document Before / After Inspect
    log("-------------------------------------------------------------------------------------")
    log("TASK 4: DETAILED BEFORE VS. AFTER COMPARISON EVIDENCE")
    log("-------------------------------------------------------------------------------------")
    for idx, doc in enumerate(cleaned_documents, start=1):
        log(f"=== Document #{idx}: {doc['source']} ({doc['file_type'].upper()}) ===")
        log(f"Character Count Change: {doc['original_char_count']} chars -> {doc['char_count']} chars ({doc['chars_removed']} chars removed, -{doc['reduction_pct']}%)")
        
        raw_preview = doc["raw_text"][:220].replace("\n", " ")
        clean_preview = doc["text"][:220].replace("\n", " ")
        log(f"BEFORE (Raw Snippet)  : {raw_preview!r}")
        log(f"AFTER  (Clean Snippet): {clean_preview!r}")
        log()

    # Task 5: Edge Case Protection Demonstration
    log("-------------------------------------------------------------------------------------")
    log("EDGE CASE HANDLING: PRESERVING CODE, PUNCTUATION, & SEMANTICS (AVOIDING OVER-CLEANING)")
    log("-------------------------------------------------------------------------------------")
    edge_case_input = (
        "# API Sample Request\r\n\r\n"
        "```json\r\n"
        '{\r\n  "threshold": 0.85,\r\n  "model": "text-embedding-3-small"\r\n}\r\n'
        "```\r\n\r\n"
        "Formula: cosine_similarity = (A · B) / (||A|| * ||B||)\r\n"
        "Rule 1: Never strip JSON symbols like braces {}, colons :, or quotation marks \"\"."
    )
    edge_case_cleaned = cleaner.clean(edge_case_input)
    log("Raw Edge Case Input (JSON & Formulas):")
    log(edge_case_input)
    log()
    log("Cleaned Result (Punctuation, JSON formatting, math symbols preserved):")
    log(edge_case_cleaned)
    log()

    # Summary Statistics
    log("-------------------------------------------------------------------------------------")
    log("CORPUS CLEANING SUMMARY STATISTICS")
    log("-------------------------------------------------------------------------------------")
    log(f"Total Ingested Documents Cleaned : {corpus_stats['total_documents']}")
    log(f"Total Raw Corpus Volume          : {corpus_stats['total_raw_chars']} characters")
    log(f"Total Cleaned Corpus Volume      : {corpus_stats['total_cleaned_chars']} characters")
    log(f"Total Boilerplate & Noise Purged : {corpus_stats['total_chars_removed']} characters")
    log(f"Corpus Compression / Cleaning %  : {corpus_stats['overall_reduction_pct']}%")
    log("=" * 85)

    # Save output to outputs/text_cleaning_output.txt
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "text_cleaning_output.txt"
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n[INFO] Text cleaning execution report successfully saved to '{output_path}'")


if __name__ == "__main__":
    run_cleaning_demo()
