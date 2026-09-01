"""
Token-Aware Chunk Sizing & Overlap Engine for Knovera RAG Assistant.

Provides token-based document chunking using tiktoken (cl100k_base or configurable).
Enforces exact token budgets, applies controlled sliding-window overlap to preserve
boundary context, tracks token spans and character offsets, and integrates with
downstream metadata tagging and vector embeddings.
"""

from typing import Dict, List, Any, Tuple, Optional
import tiktoken


class TokenChunker:
    """Enterprise Token-Aware Text Chunker with Controlled Sliding-Window Overlap."""

    DEFAULT_ENCODING = "cl100k_base"
    DEFAULT_CHUNK_SIZE = 400
    DEFAULT_OVERLAP = 60

    def __init__(
        self,
        encoding_name: str = DEFAULT_ENCODING,
        default_chunk_size: int = DEFAULT_CHUNK_SIZE,
        default_overlap: int = DEFAULT_OVERLAP,
    ):
        """Initialize TokenChunker with specified encoding and default parameters.
        
        Args:
            encoding_name: Tokenizer model encoding (default: "cl100k_base" for OpenAI GPT-4o/GPT-3.5/text-embedding-3).
            default_chunk_size: Target token capacity per chunk (default: 400).
            default_overlap: Number of overlapping tokens between consecutive chunks (default: 60, ~15%).
        """
        self.encoding_name = encoding_name
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Return total token count for the given text using the active tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def encode(self, text: str) -> List[int]:
        """Encode text to token ID sequence."""
        if not text:
            return []
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ID sequence back to string."""
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids)

    def token_chunks(
        self, text: str, size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into raw string chunks based strictly on token counts with controlled overlap.
        
        Args:
            text: Input document text.
            size: Chunk size in tokens (defaults to self.default_chunk_size).
            overlap: Overlap in tokens (defaults to self.default_overlap).
            
        Returns:
            List of decoded text chunks.
        """
        if not text:
            return []

        chunk_size = size if size is not None else self.default_chunk_size
        overlap_size = overlap if overlap is not None else self.default_overlap

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap_size < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap_size}")
        if overlap_size >= chunk_size:
            raise ValueError(
                f"overlap ({overlap_size}) must be strictly less than chunk_size ({chunk_size})"
            )

        tokens = self.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= chunk_size:
            return [text.strip()]

        step = chunk_size - overlap_size
        chunks: List[str] = []
        i = 0

        while i < total_tokens:
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            i += step
            # Avoid single token trailing chunks if remaining tokens already covered
            if i >= total_tokens:
                break

        return chunks

    def token_chunks_with_offsets(
        self, text: str, size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Split text into detailed chunk dictionaries tracking token indexes and approximate character spans.
        
        Args:
            text: Input document text.
            size: Chunk size in tokens.
            overlap: Overlap in tokens.
            
        Returns:
            List of dictionaries containing chunk text, token start/end, and token count.
        """
        if not text:
            return []

        chunk_size = size if size is not None else self.default_chunk_size
        overlap_size = overlap if overlap is not None else self.default_overlap

        if overlap_size >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        tokens = self.encode(text)
        total_tokens = len(tokens)
        step = chunk_size - overlap_size

        chunks_data: List[Dict[str, Any]] = []
        i = 0
        chunk_idx = 0

        while i < total_tokens:
            token_start = i
            token_end = min(i + chunk_size, total_tokens)
            chunk_tokens = tokens[token_start:token_end]
            chunk_text = self.decode(chunk_tokens)

            # Character offset approximation within source text
            char_len = len(chunk_text)

            chunks_data.append({
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "token_start": token_start,
                "token_end": token_end,
                "token_count": len(chunk_tokens),
                "char_length": char_len,
                "overlap_with_previous": overlap_size if chunk_idx > 0 else 0,
            })

            chunk_idx += 1
            i += step
            if i >= total_tokens:
                break

        return chunks_data

    def chunk_document(
        self,
        doc: Dict[str, Any],
        size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk a document dictionary (e.g., from DocumentLoader / TextCleaner) into tagged token-aware chunks.
        
        Preserves parent document metadata while injecting token metrics.
        """
        text = doc.get("text", "")
        chunks_info = self.token_chunks_with_offsets(text, size=size, overlap=overlap)
        
        tagged_chunks: List[Dict[str, Any]] = []
        for item in chunks_info:
            chunk_dict = {
                "text": item["text"],
                "metadata": {
                    "source": doc.get("source", "unknown"),
                    "file_type": doc.get("file_type", ".txt"),
                    "chunk_index": item["chunk_index"],
                    "token_start": item["token_start"],
                    "token_end": item["token_end"],
                    "token_count": item["token_count"],
                    "overlap_tokens": item["overlap_with_previous"],
                    "char_length": item["char_length"],
                    "doc_title": doc.get("title", doc.get("doc_title", None)),
                }
            }
            tagged_chunks.append(chunk_dict)

        return tagged_chunks

    def chunk_corpus(
        self,
        documents: List[Dict[str, Any]],
        size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process an entire document collection using token-aware chunking with overlap.
        
        Returns:
            Tuple of (all_tagged_chunks, corpus_stats_summary).
        """
        all_chunks: List[Dict[str, Any]] = []
        total_source_tokens = 0
        total_chunk_tokens = 0

        for doc in documents:
            doc_text = doc.get("text", "")
            doc_tokens = self.count_tokens(doc_text)
            total_source_tokens += doc_tokens

            doc_chunks = self.chunk_document(doc, size=size, overlap=overlap)
            all_chunks.extend(doc_chunks)

            for c in doc_chunks:
                total_chunk_tokens += c["metadata"]["token_count"]

        redundancy_tokens = max(0, total_chunk_tokens - total_source_tokens)
        redundancy_pct = (
            round((redundancy_tokens / total_source_tokens) * 100, 2)
            if total_source_tokens > 0
            else 0.0
        )

        stats = {
            "total_documents": len(documents),
            "total_chunks_created": len(all_chunks),
            "total_source_tokens": total_source_tokens,
            "total_chunked_tokens_stored": total_chunk_tokens,
            "overlap_token_redundancy": redundancy_tokens,
            "overlap_redundancy_pct": redundancy_pct,
            "target_chunk_size": size or self.default_chunk_size,
            "target_overlap": overlap if overlap is not None else self.default_overlap,
        }

        return all_chunks, stats

    def compare_overlap_effects(
        self, text: str, size: int = 400, overlap_values: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Simulate and compare how different overlap values affect chunk count, storage, and cost trade-offs."""
        if overlap_values is None:
            overlap_values = [0, 20, 40, 60, 100]

        total_tokens = self.count_tokens(text)
        results = []

        for ov in overlap_values:
            chunks = self.token_chunks(text, size=size, overlap=ov)
            total_stored_tokens = sum(self.count_tokens(c) for c in chunks)
            redundancy = total_stored_tokens - total_tokens
            redundancy_pct = (
                round((redundancy / total_tokens) * 100, 2) if total_tokens > 0 else 0.0
            )

            results.append({
                "chunk_size": size,
                "overlap": ov,
                "overlap_ratio_pct": round((ov / size) * 100, 1),
                "chunk_count": len(chunks),
                "total_stored_tokens": total_stored_tokens,
                "redundancy_tokens": redundancy,
                "redundancy_pct": redundancy_pct,
            })

        return results
