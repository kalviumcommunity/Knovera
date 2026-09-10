"""
src/top_k_retriever.py

Enterprise Similarity Search & Top-K Retrieval Engine.
Designed for Knovera RAG Assistant using ChromaDB Vector Store.

Key Capabilities:
1. Model-Aligned Query Embedding: Embeds user queries using the identical EmbeddingGenerator and model specification used for document chunks.
2. Top-K Vector Similarity Search: Queries the underlying HNSW index in ChromaDB to retrieve top-k nearest neighbor document chunks.
3. Grounded Score & Metadata Formatting: Binds cosine similarity scores, distance values, raw source text, and rich metadata (source document, chunk index, section, category).
4. Multi-K Comparative Analysis: Evaluates retrieval performance across varying k values (e.g., k=1, 3, 5) to quantify context expansion, score degradation, and noise trade-offs.
5. Model & Dimension Verification: Validates strict vector space parity between query vectors and indexed document collections.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class TopKRetriever:
    """
    Manages top-k semantic vector retrieval, score calculation, and multi-k analysis for RAG systems.
    """

    DEFAULT_COLLECTION_NAME = "knovera_top_k_chunks"

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        default_collection: str = DEFAULT_COLLECTION_NAME
    ):
        """
        Initialize the Top-K Retriever instance.
        
        Args:
            vector_db: VectorDatabase instance (creates default if None).
            embedding_generator: EmbeddingGenerator instance (creates default if None).
            default_collection: Default ChromaDB collection name.
        """
        load_dotenv()
        self.generator = embedding_generator or EmbeddingGenerator()
        self.vector_db = vector_db or VectorDatabase(embedding_generator=self.generator)
        self.default_collection = default_collection

    def verify_model_alignment(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validates that the query embedding generator's dimension matches the target collection's vector space.
        
        Returns:
            Dict[str, Any]: Model alignment diagnostic report.
        """
        target_collection = collection_name or self.default_collection
        collection_count = self.vector_db.count(target_collection)
        query_dim = self.generator.dimension
        expected_db_dim = self.vector_db.expected_dimension

        aligned = (query_dim == expected_db_dim)

        return {
            "model_name": self.generator.model_name,
            "query_dimension": query_dim,
            "collection_name": target_collection,
            "collection_record_count": collection_count,
            "collection_expected_dimension": expected_db_dim,
            "is_aligned": aligned,
            "status": "ALIGNED" if aligned else "DIMENSION_MISMATCH_WARNING"
        }

    def retrieve(
        self,
        query: Optional[str] = None,
        k: int = 3,
        collection_name: Optional[str] = None,
        query_vector: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the vector store for top-k most similar document chunks using query text or precomputed vector.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer > 0, got {k}")

        target_collection = collection_name or self.default_collection
        
        # Step 1: Embed query if query_vector not supplied directly
        if query_vector is None:
            if not query or not query.strip():
                logger.warning("Empty query string supplied to TopKRetriever.retrieve()")
                return []
            query_vectors = self.generator.embed([query.strip()])
            if not query_vectors or len(query_vectors) == 0:
                logger.error("Failed to generate embedding vector for user query.")
                return []
            q_vector = query_vectors[0]
        else:
            q_vector = query_vector

        # Step 2: Perform top-k similarity search in vector database
        raw_results = self.vector_db.query_similar(
            query_vector=q_vector,
            top_k=k,
            collection_name=target_collection
        )

        # Step 3: Format and enrich results with score and metadata
        formatted_results = []
        for idx, item in enumerate(raw_results, start=1):
            score = item.get("similarity", 0.0)
            distance = item.get("distance", 0.0)
            meta = item.get("metadata", {})
            
            # Ensure metadata keys are clean
            clean_metadata = {
                "source": meta.get("source", "unknown"),
                "chunk_index": int(meta.get("chunk_index", 0)) if str(meta.get("chunk_index", "0")).isdigit() else meta.get("chunk_index", 0),
                "section": meta.get("section", "General"),
                "doc_title": meta.get("doc_title", meta.get("source", "Document")),
                "category": meta.get("category", "Uncategorized")
            }
            # Include any custom metadata fields
            for key, val in meta.items():
                if key not in clean_metadata:
                    clean_metadata[key] = val

            formatted_results.append({
                "rank": idx,
                "score": round(float(score), 4),
                "distance": round(float(distance), 4),
                "text": item.get("text", ""),
                "metadata": clean_metadata,
                "id": item.get("id", f"chunk_{idx}")
            })

        return formatted_results

    def compare_k(
        self,
        query: str,
        k_values: List[int] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the same query across multiple k values to demonstrate context size expansion
        and score distribution changes.
        
        Args:
            query: User query string.
            k_values: List of k integers (default: [1, 3, 5]).
            collection_name: Target collection name.
            
        Returns:
            Dict[str, Any]: Multi-k comparison diagnostic payload.
        """
        if k_values is None:
            k_values = [1, 3, 5]

        target_collection = collection_name or self.default_collection
        comparison_runs = []

        start_time = time.time()

        for k in k_values:
            results = self.retrieve(query=query, k=k, collection_name=target_collection)
            
            scores = [r["score"] for r in results]
            total_chars = sum(len(r["text"]) for r in results)
            est_tokens = max(1, int(total_chars / 4.0)) # ~4 chars per token rule of thumb
            sources = list(dict.fromkeys(r["metadata"]["source"] for r in results))

            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            top_score = scores[0] if scores else 0.0
            min_score = scores[-1] if scores else 0.0

            comparison_runs.append({
                "k": k,
                "retrieved_count": len(results),
                "top_score": top_score,
                "min_score": min_score,
                "avg_score": avg_score,
                "total_context_characters": total_chars,
                "estimated_context_tokens": est_tokens,
                "unique_sources": sources,
                "results": results
            })

        duration = round(time.time() - start_time, 4)

        return {
            "query": query,
            "collection_name": target_collection,
            "embedding_model": self.generator.model_name,
            "k_values_evaluated": k_values,
            "execution_time_seconds": duration,
            "runs": comparison_runs
        }
