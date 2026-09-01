"""
Unit tests for TokenChunker and Token-Aware Chunk Sizing & Overlap Engine.
"""

import unittest
from src.token_chunker import TokenChunker


class TestTokenChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = TokenChunker(encoding_name="cl100k_base", default_chunk_size=400, default_overlap=60)
        self.sample_text = (
            "Knovera enterprise customer support policy and returns processing guideline. "
            "All customers are entitled to standard refunds within thirty days of verified product delivery. "
        ) * 15

    def test_token_counting(self):
        """Test accurate token counting."""
        count = self.chunker.count_tokens("Hello world from Knovera RAG!")
        self.assertGreater(count, 0)
        self.assertIsInstance(count, int)

    def test_chunk_size_enforcement(self):
        """Verify that every generated chunk does not exceed the target token size."""
        chunks = self.chunker.token_chunks(self.sample_text, size=50, overlap=10)
        self.assertGreater(len(chunks), 1)

        for c in chunks:
            tok_count = self.chunker.count_tokens(c)
            self.assertLessEqual(tok_count, 50)

    def test_controlled_overlap_repetition(self):
        """Verify that consecutive chunks repeat the designated overlap tokens."""
        size = 60
        overlap = 15
        chunks_info = self.chunker.token_chunks_with_offsets(self.sample_text, size=size, overlap=overlap)

        self.assertGreater(len(chunks_info), 1)
        for i in range(1, len(chunks_info)):
            prev_chunk = chunks_info[i - 1]
            curr_chunk = chunks_info[i]

            # Current chunk token_start should equal prev_chunk token_start + (size - overlap)
            expected_start = prev_chunk["token_start"] + (size - overlap)
            self.assertEqual(curr_chunk["token_start"], expected_start)

            # Overlap should match
            self.assertEqual(curr_chunk["overlap_with_previous"], overlap)

    def test_boundary_context_preservation(self):
        """Demonstrate that an idea spanning across a boundary is preserved in the overlapping chunk."""
        # Create text with a distinctive phrase right near boundary
        part1 = "Part one sentence. " * 30  # ~90 tokens
        distinctive_phrase = "THE_MAGIC_KEY_PHRASE_IS_ALPHA_99"
        part2 = "Part two sentence. " * 30

        full_text = f"{part1} {distinctive_phrase} {part2}"

        # With 0 overlap: if it gets split or lands near boundary, check chunks
        chunks_with_ov = self.chunker.token_chunks(full_text, size=100, overlap=30)
        
        # Check that the phrase is present in at least one chunk intact
        phrase_found = any(distinctive_phrase in c for c in chunks_with_ov)
        self.assertTrue(phrase_found)

    def test_invalid_parameters(self):
        """Test that invalid chunk_size or overlap raises ValueError."""
        with self.assertRaises(ValueError):
            self.chunker.token_chunks(self.sample_text, size=100, overlap=100)

        with self.assertRaises(ValueError):
            self.chunker.token_chunks(self.sample_text, size=100, overlap=150)

        with self.assertRaises(ValueError):
            self.chunker.token_chunks(self.sample_text, size=0, overlap=10)

    def test_chunk_document_metadata_preservation(self):
        """Verify that document dictionary metadata is maintained during chunking."""
        doc = {
            "source": "policy.txt",
            "file_type": ".txt",
            "title": "Knovera Return Policy",
            "text": self.sample_text,
        }

        tagged_chunks = self.chunker.chunk_document(doc, size=100, overlap=20)
        self.assertGreater(len(tagged_chunks), 0)

        for c in tagged_chunks:
            self.assertIn("text", c)
            self.assertIn("metadata", c)
            self.assertEqual(c["metadata"]["source"], "policy.txt")
            self.assertEqual(c["metadata"]["doc_title"], "Knovera Return Policy")
            self.assertIn("token_count", c["metadata"])
            self.assertIn("token_start", c["metadata"])


if __name__ == "__main__":
    unittest.main()
