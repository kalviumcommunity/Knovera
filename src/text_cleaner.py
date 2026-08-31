"""
Text Extraction & Cleaning Pipeline Module for Knovera RAG Assistant.

Provides uniform, robust text cleaning across multi-format documents before chunking
and embedding. Strips repeated headers, footers, and boilerplate; fixes encoding artifacts
and mojibake; repairs broken mid-word line wraps; and normalizes whitespace while preserving
critical semantic content.
"""

import re
import unicodedata
from typing import Dict, List, Any, Tuple, Optional


class TextCleaner:
    """Enterprise text extraction and cleaning pipeline for RAG intake."""

    # Common mojibake and encoding artifact replacement mapping
    MOJIBAKE_MAP = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€\x9c": '"',
        "â€ ": '"',
        "â€\x98": "'",
        "â€\x99": "'",
        "â€”": "—",
        "â€“": "–",
        "â€\x93": "–",
        "â€\x94": "—",
        "â€¦": "…",
        "â€¢": "•",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã±": "ñ",
        "Â©": "©",
        "Â®": "®",
        "â„¢": "™",
        "\xa0": " ",      # Non-breaking space
        "\u200b": "",     # Zero-width space
        "\u200e": "",     # Left-to-right mark
        "\u200f": "",     # Right-to-left mark
        "\ufeff": "",     # Byte order mark (BOM)
    }

    # Regex patterns for boilerplate and header/footer removal
    BOILERPLATE_PATTERNS = [
        # Page numbering patterns: "Page 3 of 12", "Page 1", "[Page 4]", "- 5 -"
        re.compile(r"(?i)\bPage\s+\d+\s+of\s+\d+\b"),
        re.compile(r"(?i)\[\s*Page\s+\d+\s*(?:of\s+\d+)?\s*\]"),
        re.compile(r"(?i)^\s*Page\s+\d+\s*$", re.MULTILINE),
        re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),
        
        # Repeated corporate document headers & footers
        re.compile(r"(?i)^\s*KNOVERA\s+(?:CONFIDENTIAL|INTERNAL\s+USE\s+ONLY)(?:\s*-\s*DO\s+NOT\s+DISTRIBUTE)?\s*$", re.MULTILINE),
        re.compile(r"(?i)^\s*CONFIDENTIAL\s+AND\s+PROPRIETARY\s*$", re.MULTILINE),
        re.compile(r"(?i)^\s*All\s+Rights\s+Reserved\.\s*$", re.MULTILINE),
        re.compile(r"(?i)^\s*©\s*\d{4}\s+Knovera\s+Technologies.*$", re.MULTILINE),
        
        # Navigation breadcrumbs and UI artifacts
        re.compile(r"(?i)^\s*Home\s*>\s*Docs(?:\s*>\s*[\w\s-]+)*\s*$", re.MULTILINE),
        re.compile(r"(?i)^\s*\[?\s*Back\s+to\s+(?:Top|Contents|Overview)\s*\]?\s*$", re.MULTILINE),
        re.compile(r"(?i)^\s*\[?\s*Skip\s+to\s+(?:main\s+content|navigation)\s*\]?\s*$", re.MULTILINE),
    ]

    def __init__(
        self,
        normalize_unicode: bool = True,
        fix_mojibake: bool = True,
        remove_boilerplate: bool = True,
        repair_line_wraps: bool = True,
        normalize_whitespace: bool = True,
    ):
        self.normalize_unicode = normalize_unicode
        self.fix_mojibake = fix_mojibake
        self.remove_boilerplate = remove_boilerplate
        self.repair_line_wraps = repair_line_wraps
        self.normalize_whitespace = normalize_whitespace

    def fix_encoding_artifacts(self, text: str) -> str:
        """Fix known mojibake character sequences and strip invisible markers."""
        if not text:
            return ""

        # Replace specific mojibake mappings
        for bad_str, good_str in self.MOJIBAKE_MAP.items():
            if bad_str in text:
                text = text.replace(bad_str, good_str)

        # Fallback for remaining broken double-quote artifacts
        text = re.sub(r"â€(?=\s|[.,;!?\n]|$)", '"', text)
        text = re.sub(r"â€", "", text)

        # Apply Unicode NFKC normalization (Standardizes compatibility characters and canonical equivalence)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        return text

    def normalize_line_endings(self, text: str) -> str:
        """Convert Windows (\r\n) and classic Mac (\r) line endings to Unix (\n)."""
        if not text:
            return ""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def repair_broken_line_wraps(self, text: str) -> str:
        """Repair hyphenated words broken across line wraps during PDF/text extraction.
        
        Example:
            'integra-\\ntion' -> 'integration'
            'deve-\\nloper' -> 'developer'
        Preserves intentional hyphenated phrases like 'state-of-the-art' when on same line.
        """
        if not text:
            return ""
        # Match a word fragment (2+ letters) followed by hyphen, newline, and next word fragment
        return re.sub(r"([a-zA-Z]{2,})-\n([a-zA-Z]{2,})", r"\1\2", text)

    def strip_boilerplate(self, text: str) -> str:
        """Strip repetitive headers, footers, page counters, and navigation boilerplate."""
        if not text:
            return ""
        for pattern in self.BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        return text

    def collapse_whitespace(self, text: str) -> str:
        """Normalize spaces, tabs, and runaway blank lines while preserving code indentation and list structure."""
        if not text:
            return ""
        # Convert tabs to single spaces
        text = text.replace("\t", " ")
        # Remove trailing spaces at the end of each line
        text = re.sub(r" +$", "", text, flags=re.MULTILINE)
        # Collapse multiple horizontal spaces after non-space characters
        text = re.sub(r"(?<=\S) {2,}", " ", text)
        # Collapse 3 or more consecutive newlines into exactly 2 (clean paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def clean(self, text: str) -> str:
        """Run the complete, uniform text cleaning pipeline on a raw text string.
        
        Pipeline Execution Order:
            1. Normalize Line Endings (\\r\\n -> \\n)
            2. Fix Encoding Artifacts & Mojibake (NFKC + Replacement map)
            3. Repair Broken Line-Wraps (De-hyphenate split words)
            4. Remove Boilerplate (Headers, footers, page numbering, nav)
            5. Normalize Whitespace (Collapse horizontal spaces & runaway newlines)
        """
        if not text:
            return ""

        # Step 1: Normalize line breaks
        cleaned = self.normalize_line_endings(text)

        # Step 2: Unicode and encoding artifacts
        if self.fix_mojibake:
            cleaned = self.fix_encoding_artifacts(cleaned)

        # Step 3: Repair mid-word line wraps
        if self.repair_line_wraps:
            cleaned = self.repair_broken_line_wraps(cleaned)

        # Step 4: Strip repeated headers, footers, and boilerplate
        if self.remove_boilerplate:
            cleaned = self.strip_boilerplate(cleaned)

        # Step 5: Whitespace normalization
        if self.normalize_whitespace:
            cleaned = self.collapse_whitespace(cleaned)

        return cleaned

    def clean_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Apply uniform cleaning to a document dictionary produced by DocumentLoader.
        
        Preserves original metadata while attaching cleaning audit metrics.
        """
        raw_text = doc.get("text", "")
        cleaned_text = self.clean(raw_text)

        orig_len = len(raw_text)
        cleaned_len = len(cleaned_text)
        chars_removed = max(0, orig_len - cleaned_len)
        reduction_pct = round((chars_removed / orig_len * 100), 2) if orig_len > 0 else 0.0

        cleaned_doc = dict(doc)
        cleaned_doc["raw_text"] = raw_text
        cleaned_doc["text"] = cleaned_text
        cleaned_doc["original_char_count"] = orig_len
        cleaned_doc["char_count"] = cleaned_len
        cleaned_doc["chars_removed"] = chars_removed
        cleaned_doc["reduction_pct"] = reduction_pct
        cleaned_doc["is_cleaned"] = True

        return cleaned_doc

    def clean_corpus(
        self, documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Clean every document in the corpus using the exact same pipeline.
        
        Guarantees corpus-wide uniformity for reliable downstream vector embedding.
        """
        cleaned_documents = []
        total_raw_chars = 0
        total_cleaned_chars = 0

        for doc in documents:
            cleaned_doc = self.clean_document(doc)
            cleaned_documents.append(cleaned_doc)
            total_raw_chars += cleaned_doc["original_char_count"]
            total_cleaned_chars += cleaned_doc["char_count"]

        total_removed = total_raw_chars - total_cleaned_chars
        avg_reduction_pct = (
            round((total_removed / total_raw_chars * 100), 2) if total_raw_chars > 0 else 0.0
        )

        stats = {
            "total_documents": len(cleaned_documents),
            "total_raw_chars": total_raw_chars,
            "total_cleaned_chars": total_cleaned_chars,
            "total_chars_removed": total_removed,
            "overall_reduction_pct": avg_reduction_pct,
        }

        return cleaned_documents, stats
