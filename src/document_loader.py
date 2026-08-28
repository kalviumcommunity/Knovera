"""
Document Loading & Multi-Format Intake Module for Knovera RAG Assistant.

Provides unified intake functionality to extract plain text from multiple file formats
(PDF, TXT, HTML, MD), preserve source document identities for citations, and handle missing
or unreadable/corrupt files without crashing the ingestion pipeline.
"""

import logging
from pathlib import Path
import re
from typing import Dict, List, Any, Tuple, Optional
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from bs4 import BeautifulSoup

# Suppress noisy low-level pypdf warnings during corrupt PDF testing
logging.getLogger("pypdf").setLevel(logging.ERROR)


class DocumentLoader:
    """Unified multi-format document loader with error handling and metadata tracking."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def load_text(self, path: Path) -> str:
        """Extract clean plain text from a file based on its extension.

        Args:
            path: Path to the target document.

        Returns:
            Extracted plain text string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported.
            PdfReadError: If the PDF file is corrupted or invalid.
            Exception: For unhandled I/O or decoding failures.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found at path: '{path}'")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: '{path}'")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported document format '{suffix}'")

        if suffix == ".pdf":
            try:
                reader = PdfReader(path)
                pages_text = []
                for i, page in enumerate(reader.pages):
                    page_content = page.extract_text() or ""
                    pages_text.append(page_content)
                raw_text = "\n".join(pages_text)
            except PdfReadError as pre:
                raise PdfReadError(f"Corrupt or malformed PDF: {pre}")
            except Exception as e:
                raise RuntimeError(f"Failed to extract PDF text: {e}")

        elif suffix in (".txt", ".md"):
            raw_text = path.read_text(encoding="utf-8", errors="ignore")

        elif suffix in (".html", ".htm"):
            raw_html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove script and style elements
            for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                script_or_style.decompose()

            raw_text = soup.get_text(separator=" ")
            # Normalize whitespace
            raw_text = re.sub(r"\s+", " ", raw_text).strip()

        # Final cleanup: replace null bytes or excessive consecutive newlines
        clean_text = raw_text.replace("\x00", "").strip()
        return clean_text

    def load_document(self, path: Path) -> Dict[str, Any]:
        """Load a single document and return it with preserved source metadata.

        Args:
            path: Path to the document.

        Returns:
            Dictionary containing document text and metadata.
        """
        text = self.load_text(path)
        return {
            "source": path.name,
            "path": str(path.resolve()),
            "file_type": path.suffix.lower(),
            "char_count": len(text),
            "text": text,
        }

    def load_corpus(self, directory: Path, recursive: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Scan a directory, load all valid multi-format documents, and survive bad input.

        Args:
            directory: Directory containing target document corpus.
            recursive: Whether to scan subdirectories.

        Returns:
            Tuple of (list of loaded document dicts, intake statistics dict).
        """
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: '{directory}'")

        file_iter = directory.rglob("*") if recursive else directory.glob("*")
        documents = []
        skipped_files = []
        format_breakdown: Dict[str, int] = {}

        if self.verbose:
            print(f"\nScanning document corpus directory: {directory.resolve()}")
            print("=" * 80)

        for path in file_iter:
            if not path.is_file():
                continue

            try:
                doc = self.load_document(path)
                documents.append(doc)
                ext = doc["file_type"]
                format_breakdown[ext] = format_breakdown.get(ext, 0) + 1

                if self.verbose:
                    snippet = doc["text"][:75].replace("\n", " ")
                    print(f"OK   [{doc['file_type'].upper()[1:]:<4}] {doc['source']:<30} | {doc['char_count']:>6} chars | Preview: {snippet!r}")

            except Exception as e:
                err_msg = str(e)
                skipped_files.append({"source": path.name, "reason": err_msg})
                if self.verbose:
                    print(f"SKIP [{path.suffix.upper()[1:] if path.suffix else 'UNK':<4}] {path.name:<30} | Reason: {err_msg}")

        stats = {
            "total_scanned": len(documents) + len(skipped_files),
            "total_successful": len(documents),
            "total_skipped": len(skipped_files),
            "format_breakdown": format_breakdown,
            "skipped_details": skipped_files,
            "total_characters": sum(d["char_count"] for d in documents),
        }

        if self.verbose:
            print("=" * 80)
            print(f"CORPUS INTAKE COMPLETE: {stats['total_successful']}/{stats['total_scanned']} files loaded successfully ({stats['total_skipped']} skipped).")
            print(f"Total extracted text volume: {stats['total_characters']} characters.")
            print("Format distribution:", ", ".join(f"{k}: {v}" for k, v in format_breakdown.items()))

        return documents, stats
