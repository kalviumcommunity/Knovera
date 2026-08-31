"""
Unit tests for Chunk Metadata Tagging, Source Tracking, and Retrieval Filtering.
"""

import unittest
from pathlib import Path
from src.chunk_tagger import ChunkTagger, extract_sections_and_metadata
from src.source_tracer import SourceTracer
from chunking import fixed_size_chunking_with_offsets, paragraph_chunking_with_offsets, tag_chunks


class TestChunkMetadata(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "KNOVERA SUPPORT POLICY\n"
            "Document ID: POL-2026-001\n"
            "Effective Date: March 1, 2026\n\n"
            "1. Overview\n"
            "Knovera provides enterprise support for RAG applications.\n\n"
            "2. Refund Terms\n"
            "Full refunds are available within 30 days of purchase."
        )
        self.source = "support_policy.txt"
        self.tagger = ChunkTagger()

    def test_metadata_schema_consistency(self):
        """Verify that every tagged chunk has the exact standard set of metadata keys."""
        doc_title, effective_date, section_map = extract_sections_and_metadata(
            self.sample_text, file_type=".txt", source_name=self.source
        )
        chunk_tuples = paragraph_chunking_with_offsets(self.sample_text)
        tagged_chunks = self.tagger.tag_chunks_from_tuples(
            source=self.source,
            chunk_tuples=chunk_tuples,
            file_type=".txt",
            doc_title=doc_title,
            section_map=section_map,
            effective_date=effective_date,
        )

        self.assertGreater(len(tagged_chunks), 0)
        expected_keys = set(ChunkTagger.REQUIRED_METADATA_KEYS)

        for chunk in tagged_chunks:
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
            chunk_keys = set(chunk["metadata"].keys())
            self.assertEqual(chunk_keys, expected_keys)
            self.assertEqual(chunk["metadata"]["source"], self.source)

    def test_traceback_exact_match(self):
        """Verify that character offsets in metadata map directly to original text."""
        chunk_tuples = paragraph_chunking_with_offsets(self.sample_text)
        tagged_chunks = self.tagger.tag_chunks_from_tuples(
            source=self.source,
            chunk_tuples=chunk_tuples,
            file_type=".txt",
        )

        for chunk in tagged_chunks:
            report = SourceTracer.trace_chunk_source(chunk, self.sample_text)
            self.assertTrue(report["verified_exact_match"])
            self.assertEqual(report["source"], self.source)
            self.assertTrue(report["line_range"].startswith("L"))

    def test_citation_formatting(self):
        """Verify citation formatting styles."""
        chunk = {
            "text": "Full refunds are available within 30 days.",
            "metadata": {
                "source": "support_policy.txt",
                "chunk_index": 2,
                "char_start": 150,
                "char_end": 192,
                "section_heading": "2. Refund Terms",
                "page_number": 1,
            },
        }

        full_citation = SourceTracer.format_citation(chunk, style="full")
        compact_citation = SourceTracer.format_citation(chunk, style="compact")
        markdown_citation = SourceTracer.format_citation(chunk, style="markdown")

        self.assertIn("support_policy.txt", full_citation)
        self.assertIn("2. Refund Terms", full_citation)
        self.assertIn("Chars: 150-192", full_citation)
        self.assertIn("p.1", compact_citation)
        self.assertIn("*According to support_policy.txt*", markdown_citation)

    def test_metadata_filtering(self):
        """Verify scoping/filtering candidate chunks by metadata attributes."""
        chunks = [
            {
                "text": "Chunk 1",
                "metadata": {"source": "doc1.txt", "file_type": ".txt", "effective_date": "2026-01-01"},
            },
            {
                "text": "Chunk 2",
                "metadata": {"source": "doc2.md", "file_type": ".md", "effective_date": "2025-06-01"},
            },
            {
                "text": "Chunk 3",
                "metadata": {"source": "doc3.txt", "file_type": ".txt", "effective_date": "2026-01-01"},
            },
        ]

        txt_only = SourceTracer.filter_chunks(chunks, criteria={"file_type": ".txt"})
        self.assertEqual(len(txt_only), 2)

        date_filtered = SourceTracer.filter_chunks(
            chunks, custom_filter=lambda m: m.get("effective_date") == "2026-01-01"
        )
        self.assertEqual(len(date_filtered), 2)

    def test_tag_chunks_helper_function(self):
        """Test tag_chunks function matching problem statement signature."""
        sample_tuples = [("Hello World", 0, 11), ("Second Chunk", 13, 25)]
        tagged = tag_chunks(source="test_doc.txt", chunks=sample_tuples, file_type=".txt")

        self.assertEqual(len(tagged), 2)
        self.assertEqual(tagged[0]["metadata"]["source"], "test_doc.txt")
        self.assertEqual(tagged[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(tagged[0]["metadata"]["char_start"], 0)
        self.assertEqual(tagged[0]["metadata"]["char_end"], 11)


if __name__ == "__main__":
    unittest.main()
