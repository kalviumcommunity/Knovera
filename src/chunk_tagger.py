"""
Chunk Metadata Tagger Module for Knovera RAG Assistant.

Pairs floating text chunks with rich, consistent metadata structures to enable
source citation, exact traceback verification, and metadata-driven retrieval filtering.
"""

from pathlib import Path
import re
from typing import Dict, List, Any, Optional, Tuple


class ChunkTagger:
    """Tags text chunks with consistent metadata structures across all document types."""

    REQUIRED_METADATA_KEYS = [
        "source",
        "chunk_index",
        "char_start",
        "char_end",
        "char_length",
        "file_type",
        "doc_title",
        "section_heading",
        "page_number",
        "effective_date",
    ]

    def __init__(self, default_title: Optional[str] = None):
        self.default_title = default_title

    def create_metadata(
        self,
        source: str,
        chunk_index: int,
        char_start: int,
        char_end: int,
        file_type: str,
        doc_title: Optional[str] = None,
        section_heading: Optional[str] = None,
        page_number: Optional[int] = 1,
        effective_date: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct a standardized, consistent metadata dictionary.

        Ensures that every chunk in the corpus shares the exact same metadata structure.
        """
        metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
            "char_length": max(0, char_end - char_start),
            "file_type": file_type.lower() if file_type.startswith(".") else f".{file_type.lower()}",
            "doc_title": doc_title or source,
            "section_heading": section_heading or "General",
            "page_number": page_number if page_number is not None else 1,
            "effective_date": effective_date,
        }

        # Validate that all required keys are present
        for key in self.REQUIRED_METADATA_KEYS:
            if key not in metadata:
                metadata[key] = None

        if extra:
            metadata.update(extra)

        return metadata

    def tag_chunk(
        self,
        text: str,
        source: str,
        chunk_index: int,
        char_start: int,
        char_end: int,
        file_type: str,
        doc_title: Optional[str] = None,
        section_heading: Optional[str] = None,
        page_number: Optional[int] = 1,
        effective_date: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pair a raw text string with a standardized metadata dictionary.

        Returns:
            Dict containing 'text' and 'metadata' keys.
        """
        metadata = self.create_metadata(
            source=source,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            file_type=file_type,
            doc_title=doc_title,
            section_heading=section_heading,
            page_number=page_number,
            effective_date=effective_date,
            extra=extra,
        )
        return {
            "text": text,
            "metadata": metadata,
        }

    def tag_chunks_from_tuples(
        self,
        source: str,
        chunk_tuples: List[Tuple[str, int, int]],
        file_type: str,
        doc_title: Optional[str] = None,
        section_map: Optional[List[Tuple[int, str]]] = None,
        page_map: Optional[List[Tuple[int, int]]] = None,
        effective_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Tag a list of chunk tuples (text, char_start, char_end) with consistent metadata.

        Args:
            source: Source document identifier (e.g. filename).
            chunk_tuples: List of (text, char_start, char_end) tuples.
            file_type: File extension (e.g. '.txt', '.md').
            doc_title: Optional document title.
            section_map: Optional list of (char_start_offset, section_name) to resolve section headings.
            page_map: Optional list of (char_start_offset, page_num) to resolve page numbers.
            effective_date: Optional document effective date.

        Returns:
            List of standard chunk dictionaries containing 'text' and 'metadata'.
        """
        tagged_chunks = []
        for idx, (chunk_text, start_pos, end_pos) in enumerate(chunk_tuples):
            # Resolve section heading dynamically based on start_pos offset
            heading = "General"
            if section_map:
                for sec_offset, sec_name in reversed(section_map):
                    if start_pos >= sec_offset:
                        heading = sec_name
                        break

            # Resolve page number dynamically based on start_pos offset
            pg_num = 1
            if page_map:
                for page_offset, p_num in reversed(page_map):
                    if start_pos >= page_offset:
                        pg_num = p_num
                        break

            tagged_chunk = self.tag_chunk(
                text=chunk_text,
                source=source,
                chunk_index=idx,
                char_start=start_pos,
                char_end=end_pos,
                file_type=file_type,
                doc_title=doc_title,
                section_heading=heading,
                page_number=pg_num,
                effective_date=effective_date,
            )
            tagged_chunks.append(tagged_chunk)

        return tagged_chunks


def extract_sections_and_metadata(
    full_text: str, file_type: str, source_name: str
) -> Tuple[Optional[str], Optional[str], List[Tuple[int, str]]]:
    """Helper to extract document title, effective date, and section offset map from raw text."""
    doc_title = None
    effective_date = None
    section_map: List[Tuple[int, str]] = []

    # 1. Extract Effective Date if present
    date_match = re.search(
        r"(?:Effective Date|Version Date|Date):\s*([A-Za-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2})",
        full_text,
        re.IGNORECASE,
    )
    if date_match:
        effective_date = date_match.group(1).strip()

    # 2. Extract Document Title
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    if lines:
        if file_type in (".md", "md"):
            for line in lines:
                if line.startswith("# "):
                    doc_title = line.lstrip("# ").strip()
                    break
        elif file_type in (".html", ".htm"):
            title_match = re.search(r"<title>(.*?)</title>", full_text, re.IGNORECASE)
            if title_match:
                doc_title = title_match.group(1).strip()

        if not doc_title and lines:
            doc_title = lines[0]  # Fallback to first non-empty line

    # 3. Extract Section Headings & Offset Positions
    if file_type in (".md", "md"):
        # Match markdown headers (# Heading, ## Heading)
        pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        for match in pattern.finditer(full_text):
            offset = match.start()
            heading = match.group(2).strip()
            section_map.append((offset, heading))

    elif file_type in (".txt", "txt"):
        # Match numbered headers like '1. Overview', '2. Service Level Agreements (SLAs)'
        pattern = re.compile(r"^(\d+\.\s+[A-Za-z0-9\s\(\)\-\&]+)$", re.MULTILINE)
        for match in pattern.finditer(full_text):
            offset = match.start()
            heading = match.group(1).strip()
            section_map.append((offset, heading))

    elif file_type in (".html", ".htm"):
        # Match H1/H2/H3 tags
        pattern = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
        for match in pattern.finditer(full_text):
            offset = match.start()
            clean_heading = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if clean_heading:
                section_map.append((offset, clean_heading))

    return doc_title, effective_date, section_map
