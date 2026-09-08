"""
src/hybrid_retriever.py

Enterprise Metadata Filtering & Hybrid Retrieval Engine.
Designed for Knovera RAG Assistant using ChromaDB Vector Store.

Key Capabilities:
1. Metadata-Scoped Vector Retrieval: Restricts semantic search to targeted corpus subsets (source, section, doc_type, category, user_role, date).
2. Lexical & Exact-Keyword Matching: Computes token-level and phrase-level keyword relevance for exact IDs, error codes, names, and technical terms.
3. Weighted Hybrid Ranking: Blends continuous vector cosine similarity and discrete lexical keyword scores with configurable weight parameters (vector_weight, keyword_weight).
4. Multi-Mode Comparative Evaluation: Directly contrasts Unfiltered Vector, Filtered Vector, Unfiltered Hybrid, and Filtered Hybrid retrieval modes.
5. Precision & Relevancy Diagnostics: Quantifies precision gains by filtering out irrelevant cross-domain distractors.
"""

import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


def extract_search_tokens(text: str) -> List[str]:
    """
    Extracts alphanumeric keywords, identifiers, error codes, and terms from text.
    Preserves hyphenated identifiers like 'ERR-AUTH-902', 'SEC-POL-404', 'CS-101'.
    """
    if not text:
        return []
    # Match words and hyphenated/dotted identifiers
    tokens = re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())
    # Filter out single-character punctuation artifacts
    stop_words = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "are", "what", "how", "why"}
    return [t for t in tokens if len(t) > 1 and t not in stop_words]


def keyword_score(
    text: str,
    keywords: Union[List[str], str],
    case_sensitive: bool = False,
    exact_match: bool = False,
    method: str = "frequency"
) -> float:
    """
    Computes lexical keyword matching score for a text chunk against target keywords.
    
    Args:
        text: Chunk text content.
        keywords: Single keyword string or list of keyword terms.
        case_sensitive: If True, performs case-sensitive matching.
        exact_match: If True, requires exact word boundary matches.
        method: 'count' (raw match count), 'frequency' (normalized term occurrences),
                or 'presence' (fraction of keywords present).
                
    Returns:
        float: Calculated lexical score.
    """
    if not text or not keywords:
        return 0.0

    if isinstance(keywords, str):
        keyword_list = [keywords]
    else:
        keyword_list = list(keywords)

    if not keyword_list:
        return 0.0

    search_text = text if case_sensitive else text.lower()
    matches_per_keyword = {}

    for kw in keyword_list:
        pattern_str = kw if case_sensitive else kw.lower()
        if not pattern_str:
            continue
        if exact_match:
            # Word boundary regex
            escaped = re.escape(pattern_str)
            found = len(re.findall(r'\b' + escaped + r'\b', search_text))
        else:
            found = search_text.count(pattern_str)
        matches_per_keyword[pattern_str] = found

    total_matches = sum(matches_per_keyword.values())
    unique_present = sum(1 for cnt in matches_per_keyword.values() if cnt > 0)

    if method == "count":
        return float(total_matches)
    elif method == "presence":
        return float(unique_present / len(keyword_list))
    elif method == "frequency":
        # Density score normalized to reasonable [0.0, 1.0] scale with saturation
        # Rewarding both occurrence count and multi-keyword coverage
        coverage = unique_present / len(keyword_list)
        density_boost = min(1.0, total_matches / (len(keyword_list) * 2.0))
        return round(0.5 * coverage + 0.5 * density_boost, 4)
    else:
        return float(total_matches)


def hybrid_rank(
    vector_results: List[Dict[str, Any]],
    keywords: Optional[List[str]] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    normalize_vector_scores: bool = True
) -> List[Dict[str, Any]]:
    """
    Combines vector semantic similarity scores with lexical keyword scores to re-rank chunks.
    
    Formula:
        S_hybrid = (vector_weight * S_vector) + (keyword_weight * S_keyword)
        
    Args:
        vector_results: List of vector retrieval records containing 'score' or 'similarity' and 'text'.
        keywords: List of target exact keywords/phrases to score.
        vector_weight: Weight assigned to semantic vector similarity (default: 0.7).
        keyword_weight: Weight assigned to lexical keyword matching (default: 0.3).
        normalize_vector_scores: If True, ensures vector scores are within [0, 1].
        
    Returns:
        List[Dict[str, Any]]: Re-ranked results sorted in descending order of hybrid_score.
    """
    if not vector_results:
        return []

    # Normalize weights to sum to 1.0
    total_w = vector_weight + keyword_weight
    w_vec = vector_weight / total_w if total_w > 0 else 0.7
    w_kw = keyword_weight / total_w if total_w > 0 else 0.3

    kw_list = keywords or []

    ranked = []
    for item in vector_results:
        # Retrieve vector similarity score
        v_score = item.get("score", item.get("similarity", 0.0))
        if normalize_vector_scores:
            # Cosine similarity is in [-1, 1], map to [0, 1] if negative, though positive typical
            v_score_norm = max(0.0, min(1.0, (v_score + 1.0) / 2.0 if v_score < 0 else v_score))
        else:
            v_score_norm = max(0.0, v_score)

        text_content = item.get("text", "")
        lex_score = keyword_score(text_content, kw_list, method="frequency") if kw_list else 0.0
        raw_count = int(keyword_score(text_content, kw_list, method="count")) if kw_list else 0

        combined = (w_vec * v_score_norm) + (w_kw * lex_score)

        record_copy = dict(item)
        record_copy.update({
            "vector_score": round(v_score, 4),
            "keyword_score": round(lex_score, 4),
            "keyword_matches_count": raw_count,
            "hybrid_score": round(combined, 4)
        })
        ranked.append(record_copy)

    # Sort descending by hybrid_score, break ties with vector_score
    ranked.sort(key=lambda x: (x["hybrid_score"], x["vector_score"]), reverse=True)

    # Update rank indices
    for idx, r in enumerate(ranked, start=1):
        r["rank"] = idx

    return ranked


class HybridRetriever:
    """
    Enterprise hybrid retrieval manager that coordinates metadata filtering,
    vector nearest-neighbor search, and lexical keyword scoring.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        default_collection: str = "knovera_knowledge_base"
    ):
        """
        Initializes the HybridRetriever.
        
        Args:
            vector_db: VectorDatabase instance. If None, instantiates default VectorDatabase.
            embedding_generator: EmbeddingGenerator instance for query embedding.
            default_collection: Target collection name in ChromaDB.
        """
        load_dotenv()
        self.vector_db = vector_db or VectorDatabase()
        self.generator = embedding_generator or self.vector_db.generator or EmbeddingGenerator()
        self.default_collection = default_collection

    def embed_query(self, query: str) -> List[float]:
        """Generates a 1536-dimensional embedding vector for a single query string."""
        if hasattr(self.generator, "embed"):
            res = self.generator.embed([query])
            return res[0] if res else [0.0] * getattr(self.generator, "dimension", 1536)
        elif hasattr(self.generator, "generate_embedding"):
            return self.generator.generate_embedding(query)
        raise AttributeError("EmbeddingGenerator does not provide 'embed' or 'generate_embedding' method.")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search with optional metadata filtering.
        
        Args:
            query: User search query string.
            top_k: Number of nearest neighbor records to return.
            metadata_filter: Optional dictionary with ChromaDB metadata filter conditions
                             (e.g., {"section": "Account access"}, {"category": "Authentication"}).
            collection_name: Target ChromaDB collection name.
            
        Returns:
            List[Dict[str, Any]]: Retrieved records with score, id, text, metadata, and rank.
        """
        target_collection = collection_name or self.default_collection

        # Generate query vector embedding
        query_vector = self.embed_query(query)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        # Query vector store with optional metadata filter
        raw_results = self.vector_db.query_similar(
            query_vector=query_vector,
            top_k=top_k,
            where=metadata_filter,
            collection_name=target_collection
        )

        formatted = []
        for item in raw_results:
            formatted.append({
                "id": item["id"],
                "score": round(item.get("similarity", 0.0), 4),
                "distance": round(item.get("distance", 0.0), 4),
                "source": item.get("metadata", {}).get("source", "unknown"),
                "section": item.get("metadata", {}).get("section", "General"),
                "category": item.get("metadata", {}).get("category", "General"),
                "metadata": item.get("metadata", {}),
                "text": item.get("text", ""),
                "rank": item.get("rank", len(formatted) + 1)
            })

        return formatted

    def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        pre_fetch_multiplier: int = 3,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes end-to-end hybrid retrieval:
        1. Pre-fetches top candidate pool via vector search (with metadata filter if supplied).
        2. Applies lexical keyword scoring for exact terms / IDs / names.
        3. Fuses scores using weighted hybrid formula.
        4. Re-ranks and returns top_k results.
        
        Args:
            query: Semantic search query.
            keywords: Specific exact-match keywords (if None, extracted automatically from query).
            top_k: Final number of ranked records to return.
            metadata_filter: Optional metadata scoping filter.
            vector_weight: Weight for vector similarity [0.0 - 1.0].
            keyword_weight: Weight for keyword matching [0.0 - 1.0].
            pre_fetch_multiplier: Pre-fetch pool size multiplier (default: 3 * top_k).
            collection_name: Target collection.
            
        Returns:
            List[Dict[str, Any]]: Top-k hybrid re-ranked records.
        """
        # If keywords not explicitly provided, extract from query
        search_keywords = keywords if keywords is not None else extract_search_tokens(query)

        # Pre-fetch a wider candidate pool for re-ranking
        fetch_k = max(top_k * pre_fetch_multiplier, 10)
        candidates = self.retrieve(
            query=query,
            top_k=fetch_k,
            metadata_filter=metadata_filter,
            collection_name=collection_name
        )

        if not candidates:
            return []

        # Apply hybrid ranking
        ranked_pool = hybrid_rank(
            vector_results=candidates,
            keywords=search_keywords,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight
        )

        # Slice to top_k
        return ranked_pool[:top_k]

    def compare_retrieval_modes(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        keywords: Optional[List[str]] = None,
        top_k: int = 3,
        target_criteria: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compares all 4 retrieval modes for a given query:
        1. Unfiltered Vector Search
        2. Filtered Vector Search
        3. Unfiltered Hybrid Search
        4. Filtered Hybrid Search
        
        Calculates precision metrics based on target criteria (e.g., category == 'Authentication').
        """
        search_keywords = keywords if keywords is not None else extract_search_tokens(query)

        # 1. Unfiltered Vector Search
        unfiltered_vector = self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=None,
            collection_name=collection_name
        )

        # 2. Filtered Vector Search
        filtered_vector = self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            collection_name=collection_name
        )

        # 3. Unfiltered Hybrid Search
        unfiltered_hybrid = self.hybrid_search(
            query=query,
            keywords=search_keywords,
            top_k=top_k,
            metadata_filter=None,
            collection_name=collection_name
        )

        # 4. Filtered Hybrid Search
        filtered_hybrid = self.hybrid_search(
            query=query,
            keywords=search_keywords,
            top_k=top_k,
            metadata_filter=metadata_filter,
            collection_name=collection_name
        )

        # Helper to compute precision against target criteria
        def compute_precision(results: List[Dict[str, Any]]) -> float:
            if not results or not target_criteria:
                return 1.0 if results else 0.0
            matches = 0
            for r in results:
                meta = r.get("metadata", {})
                is_match = all(meta.get(k) == v for k, v in target_criteria.items())
                if is_match:
                    matches += 1
            return round(matches / len(results), 4)

        prec_unfiltered_vec = compute_precision(unfiltered_vector)
        prec_filtered_vec = compute_precision(filtered_vector)
        prec_unfiltered_hyb = compute_precision(unfiltered_hybrid)
        prec_filtered_hyb = compute_precision(filtered_hybrid)

        return {
            "query": query,
            "metadata_filter": metadata_filter,
            "keywords": search_keywords,
            "top_k": top_k,
            "target_criteria": target_criteria or metadata_filter,
            "results": {
                "unfiltered_vector": unfiltered_vector,
                "filtered_vector": filtered_vector,
                "unfiltered_hybrid": unfiltered_hybrid,
                "filtered_hybrid": filtered_hybrid
            },
            "precision_metrics": {
                "unfiltered_vector_precision": prec_unfiltered_vec,
                "filtered_vector_precision": prec_filtered_vec,
                "unfiltered_hybrid_precision": prec_unfiltered_hyb,
                "filtered_hybrid_precision": prec_filtered_hyb,
                "precision_improvement": round(prec_filtered_hyb - prec_unfiltered_vec, 4)
            }
        }
