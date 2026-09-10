"""
Document Loading & Multi-Format Intake Demonstration Script for Knovera RAG Assistant.

Task 1: Load multiple formats (PDF, TXT, HTML, MD) into a clean common text representation.
Task 2: Handle missing, corrupt, or unsupported files gracefully without crashing.
Task 3: Preserve source identity metadata for citation in downstream RAG retrieval.
Task 4: Confirm intake by reporting character length and text preview sample.
Task 5: Save execution output for review.
"""

from pathlib import Path
import sys
import io
from src.document_loader import DocumentLoader


def run_intake_demo():
    output_lines = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("       KNOVERA RAG ASSISTANT - DOCUMENT LOADING & INTAKE REPORT")
    log("=" * 80)
    log("Sprint 2 · Concept 10 · Document Processing (Assignment 3.19)")
    log()

    data_dir = Path("data")
    loader = DocumentLoader(verbose=False)

    log("-------------------------------------------------------------------------")
    log("TASK 1 & TASK 3: MULTI-FORMAT INTAKE & SOURCE IDENTITY PRESERVATION")
    log("-------------------------------------------------------------------------")
    log("Scanning directory 'data/' for PDF, TXT, HTML, and Markdown files...")
    log()

    documents, stats = loader.load_corpus(data_dir, recursive=True)

    log()
    log("-------------------------------------------------------------------------")
    log("TASK 4: INTAKE CONFIRMATION & SAMPLE EXTRACTED TEXT")
    log("-------------------------------------------------------------------------")
    for idx, doc in enumerate(documents, start=1):
        log(f"--- Document #{idx} ---")
        log(f"Source Identifier : {doc['source']}")
        log(f"Format Extension  : {doc['file_type']}")
        log(f"Character Length  : {doc['char_count']} chars")
        log(f"Absolute Path     : {doc['path']}")
        sample_snippet = doc["text"][:180].replace("\n", " ")
        log(f"Text Preview      : {sample_snippet!r}...")
        log()

    log("-------------------------------------------------------------------------")
    log("TASK 2: GRACEFUL ERROR HANDLING & UNREADABLE FILE SURVIVAL")
    log("-------------------------------------------------------------------------")
    log("Demonstrating individual error handling for missing and invalid files:")
    log()

    # 1. Missing File
    missing_path = Path("data/missing_doc_99.pdf")
    try:
        loader.load_document(missing_path)
    except Exception as e:
        log(f"SKIP [MISSING] {missing_path.name}: Caught expected error -> {e}")

    # 2. Corrupt PDF
    corrupt_path = Path("data/corrupt_file.pdf")
    try:
        loader.load_document(corrupt_path)
    except Exception as e:
        log(f"SKIP [CORRUPT] {corrupt_path.name}: Caught expected error -> {e}")

    # 3. Unsupported Extension
    unsupported_path = Path("data/unsupported_file.xyz")
    try:
        loader.load_document(unsupported_path)
    except Exception as e:
        log(f"SKIP [UNSUPPORTED] {unsupported_path.name}: Caught expected error -> {e}")

    log()
    log("-------------------------------------------------------------------------")
    log("INTAKE SUMMARY STATISTICS")
    log("-------------------------------------------------------------------------")
    log(f"Total Files Scanned      : {stats['total_scanned']}")
    log(f"Successfully Loaded Files: {stats['total_successful']}")
    log(f"Skipped / Failed Files   : {stats['total_skipped']}")
    log(f"Total Characters Ingested: {stats['total_characters']} chars")
    log("Ingested Formats         : " + ", ".join(f"{k}: {v}" for k, v in stats['format_breakdown'].items()))
    log()
    log("Skipped File Reasons:")
    for skip in stats["skipped_details"]:
        log(f"  - {skip['source']}: {skip['reason']}")
    log("=" * 80)

    # Save output to file
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "document_loading_output.txt"
    output_file.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n[INFO] Document loading report saved to '{output_file}'")


if __name__ == "__main__":
    run_intake_demo()
