"""
src/reranker.py

Enterprise Chunk Re-Ranking Engine for Knovera RAG Assistant.
Implements a Two-Stage Retrieval Architecture:
  Stage 1 (Bi-Encoder Retrieval): Fast vector search over entire corpus (K_initial = 10-20).
  Stage 2 (Cross-Scoring Re-Ranking): Deep query-chunk relevance evaluation (K_final = 3-5).

Key Capabilities:
1. LLM & Algorithmic Cross-Scoring: Evaluates joint query-document relevance (0.0 - 10.0).
2. Rank Shift Tracking: Measures rank delta (initial_rank vs rerank_rank) for every chunk.
3. Cost & Latency Diagnostics: Quantifies token overhead, API cost, and latency trade-offs.
4. Two-Stage Pipeline Orchestration: Unifies VectorDatabase retrieval with re-ranking.
"""

import os
import sys
import re
import time
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator, cosine_similarity

# Optional OpenAI client for LLM-based re-ranking
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class ChunkReranker:
    """
    Two-stage retrieval and re-ranking orchestrator for precision RAG context assembly.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        llm_model: Optional[str] = None,
        default_collection: str = "knovera_knowledge_base"
    ):
        """
        Initializes the ChunkReranker.
        
        Args:
            vector_db: Vector database instance for Stage 1 candidate retrieval.
            embedding_generator: Generator for query embeddings and semantic scoring.
            llm_model: LLM model identifier for cross-encoder prompt scoring.
            default_collection: Target collection in ChromaDB.
        """
        load_dotenv()
        self.vector_db = vector_db or VectorDatabase()
        self.generator = embedding_generator or self.vector_db.generator or EmbeddingGenerator()
        self.default_collection = default_collection
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") if os.getenv("OPENROUTER_API_KEY") else None
        self.llm_model = llm_model or os.getenv("RERANKER_MODEL") or os.getenv("CHAT_MODEL") or "openai/gpt-4o-mini"
        
        self.client = None
        if HAS_OPENAI and self.api_key:
            try:
                headers = {
                    "HTTP-Referer": "https://knovera.ai",
                    "X-Title": "Knovera RAG Assistant Re-Ranker"
                }
                if self.base_url:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, default_headers=headers)
                else:
                    self.client = OpenAI(api_key=self.api_key, default_headers=headers)
                logger.info(f"Initialized LLM Re-Ranker client with model: {self.llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client for re-ranking: {e}. Using algorithmic cross-scorer.")
                self.client = None
        else:
            logger.info("No API key provided. Operating in deterministic cross-encoder scoring mode.")

    def score_chunk_llm(self, query: str, chunk_text: str) -> Optional[float]:
        """
        Scores query-chunk relevance using an LLM cross-encoder prompt (0.0 to 10.0 scale).
        """
        if not self.client:
            return None

        prompt = (
            "You are an expert RAG relevance re-ranker. "
            "Score how directly and accurately the following chunk text answers or provides required evidence for the query.\n"
            "Score on a scale from 0.0 to 10.0 (where 0 is completely irrelevant and 10 is the exact, complete answer).\n"
            "Respond ONLY with a single numeric float (e.g., 8.5).\n\n"
            f"Query: {query}\n"
            f"Chunk Text: {chunk_text}\n\n"
            "Score:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            raw_content = response.choices[0].message.content.strip()
            # Extract float
            match = re.search(r'[-+]?\d*\.\d+|\d+', raw_content)
            if match:
                val = float(match.group())
                return round(max(0.0, min(10.0, val)), 2)
        except Exception as e:
            logger.warning(f"LLM re-ranking failed: {e}. Falling back to cross-scoring.")
        return None

    def score_chunk_algorithmic(
        self,
        query: str,
        chunk_text: str,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """
        High-precision deterministic cross-scorer combining:
        1. Exact query keyword / phrase alignment (45% weight)
        2. Semantic embedding cosine alignment (40% weight)
        3. Information density and answer specificity (15% weight)
        
        Returns:
            float: Score bounded in [0.0, 10.0].
        """
        if not query or not chunk_text:
            return 0.0

        q_lower = query.lower()
        c_lower = chunk_text.lower()

        # 1. Lexical Token Alignment
        tokens = [t for t in re.findall(r'[a-zA-Z0-9_\-]+', q_lower) if len(t) > 2]
        stop_words = {"what", "which", "where", "when", "how", "why", "the", "and", "for", "are", "with", "this", "that"}
        keywords = [t for t in tokens if t not in stop_words]
        
        if keywords:
            matches = sum(1 for kw in keywords if kw in c_lower)
            lexical_ratio = matches / len(keywords)
            # Bonus for contiguous 2-gram matching
            bigrams = [f"{keywords[i]} {keywords[i+1]}" for i in range(len(keywords) - 1)]
            bigram_matches = sum(1 for bg in bigrams if bg in c_lower)
            bigram_bonus = (bigram_matches / len(bigrams)) if bigrams else 0.0
            lexical_score = min(1.0, 0.7 * lexical_ratio + 0.3 * bigram_bonus)
        else:
            lexical_score = 0.5

        # 2. Semantic Embedding Cosine Alignment
        q_emb = query_embedding if query_embedding is not None else self.generator.embed([query])[0]
        c_emb = self.generator.embed([chunk_text])[0]
        semantic_score = max(0.0, cosine_similarity(q_emb, c_emb))

        # 3. Specificity & Evidence Indicators
        # Rewards chunks containing explicit requirements, numbers, bullets, identifiers
        specificity_markers = ["require", "must", "mandatory", "step", "code", "link", "submit", "url", "artifact", "http", "guideline", "1.", "2."]
        spec_count = sum(1 for marker in specificity_markers if marker in c_lower)
        spec_score = min(1.0, spec_count / 4.0)

        # Combine into 0.0 - 10.0 scale
        combined = (0.45 * lexical_score) + (0.40 * semantic_score) + (0.15 * spec_score)
        return round(float(np_clip_scale(combined * 10.0, 0.0, 10.0)), 2)

    def score_chunk(
        self,
        query: str,
        chunk_text: str,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """
        Evaluates relevance score of a chunk for a given query (0.0 to 10.0 scale).
        """
        llm_score = self.score_chunk_llm(query, chunk_text)
        if llm_score is not None:
            return llm_score
        return self.score_chunk_algorithmic(query, chunk_text, query_embedding=query_embedding)

    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scores and re-ranks a candidate set of retrieved chunks.
        
        Args:
            query: User search query.
            candidates: List of candidate chunk dictionaries from Stage 1 retrieval.
            top_k: Number of top re-ranked chunks to return (K_final).
            query_embedding: Optional precomputed query embedding vector.
            
        Returns:
            List[Dict[str, Any]]: Reordered top_k chunks with rerank_score, rank deltas, and metadata.
        """
        if not candidates:
            return []

        q_emb = query_embedding if query_embedding is not None else self.generator.embed([query])[0]

        scored_candidates = []
        for idx, item in enumerate(candidates, start=1):
            text = item.get("text", "")
            r_score = self.score_chunk(query, text, query_embedding=q_emb)
            
            record = dict(item)
            record["initial_rank"] = item.get("rank", idx)
            record["vector_score"] = round(item.get("score", item.get("similarity", 0.0)), 4)
            record["rerank_score"] = r_score
            scored_candidates.append(record)

        # Sort descending by rerank_score, break ties with vector_score
        scored_candidates.sort(key=lambda x: (x["rerank_score"], x["vector_score"]), reverse=True)

        # Assign final re-ranked ranks and compute rank delta
        for final_idx, item in enumerate(scored_candidates, start=1):
            item["rerank_rank"] = final_idx
            # Positive delta means promoted (e.g. initial rank 4 -> final rank 1 = +3)
            item["rank_delta"] = item["initial_rank"] - final_idx

        return scored_candidates[:top_k]

    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Convenience alias for rerank_candidates."""
        return self.rerank_candidates(query=query, candidates=candidate_chunks, top_k=top_n)

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 10,
        final_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full Two-Stage Retrieval and Re-Ranking Pipeline.
        
        Stage 1: Retrieve initial_k candidates from vector database.
        Stage 2: Re-rank candidate pool and select final_k results.
        
        Returns:
            Dict[str, Any]: Pipeline execution summary containing initial candidates,
                            re-ranked selection, execution timing, and ordering comparison.
        """
        target_collection = collection_name or self.default_collection
        start_time = time.time()

        # --- Stage 1: Fast Vector Candidate Retrieval ---
        s1_start = time.time()
        q_emb = self.generator.embed([query])[0]
        raw_candidates = self.vector_db.query_similar(
            query_vector=q_emb,
            top_k=initial_k,
            where=metadata_filter,
            collection_name=target_collection
        )
        s1_duration = time.time() - s1_start

        # Standardize candidates
        candidates = []
        for idx, item in enumerate(raw_candidates, start=1):
            candidates.append({
                "id": item["id"],
                "score": round(item.get("similarity", 0.0), 4),
                "distance": round(item.get("distance", 0.0), 4),
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "source": item.get("metadata", {}).get("source", "unknown"),
                "section": item.get("metadata", {}).get("section", "General"),
                "rank": idx
            })

        # --- Stage 2: Deep Re-Ranking ---
        s2_start = time.time()
        all_reranked = self.rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=len(candidates),
            query_embedding=q_emb
        )
        final_selected = all_reranked[:final_k]
        s2_duration = time.time() - s2_start

        total_duration = time.time() - start_time

        # Cost Estimation (Tokens per candidate ~ 150 tokens)
        estimated_input_tokens = len(candidates) * 150
        # gpt-4o-mini pricing: $0.15 / 1M input tokens
        estimated_cost_usd = (estimated_input_tokens / 1_000_000) * 0.15

        return {
            "query": query,
            "initial_k": initial_k,
            "final_k": final_k,
            "metadata_filter": metadata_filter,
            "candidates_count": len(candidates),
            "stage_1_initial_top_k": candidates[:final_k],
            "stage_1_all_candidates": candidates,
            "stage_2_final_selected": final_selected,
            "stage_2_all_reranked": all_reranked,
            "metrics": {
                "stage_1_retrieval_latency_ms": round(s1_duration * 1000, 2),
                "stage_2_rerank_latency_ms": round(s2_duration * 1000, 2),
                "total_pipeline_latency_ms": round(total_duration * 1000, 2),
                "candidates_scored": len(candidates),
                "estimated_tokens_processed": estimated_input_tokens,
                "estimated_cost_usd": round(estimated_cost_usd, 6)
            }
        }


def np_clip_scale(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


# Alias for compatibility
LLMReranker = ChunkReranker
