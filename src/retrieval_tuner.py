"""
src/retrieval_tuner.py

Enterprise Retrieval Relevance Tuning & Experimentation Engine.
Designed for Knovera RAG Assistant using ChromaDB Vector Store.

Key Capabilities:
1. Ground Truth Test Query Specification: Manages query-expected_source tuples with optional expected chunk IDs and metadata tags.
2. Multi-Setting Retrieval Comparison: Evaluates combinations of top-k, metadata filters, score thresholds (min_score), vector/hybrid search, and weights.
3. Relevance Metrics Computation: Computes Hit Rate, Top-1 Hit Rate, Mean Reciprocal Rank (MRR), Average Score, and Noise Ratio.
4. Quantitative Best-Setting Selection: Evaluates performance across settings and identifies the optimal retrieval setting with data-backed justification.
5. Structured Audit Log & Report Generation: Formats evaluation results into Markdown tables and exports JSON/text audit artifacts.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.top_k_retriever import TopKRetriever
from src.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    """Represents a ground-truth test query with expected target document/source."""
    __test__ = False  # Prevent Pytest from treating this dataclass as a test class
    query: str
    expected_source: str
    expected_chunk_ids: Optional[List[str]] = field(default_factory=list)
    category: str = "General"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalSetting:
    """Defines a specific retrieval configuration to evaluate."""
    name: str
    k: int = 3
    metadata_filter: Optional[Dict[str, Any]] = None
    min_score: float = 0.0
    use_hybrid: bool = False
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationRow:
    """Holds detailed evaluation results for a single query under a specific setting."""
    query: str
    expected_source: str
    returned_sources: List[str]
    returned_chunk_ids: List[str]
    returned_scores: List[float]
    hit: bool
    top_1_hit: bool
    reciprocal_rank: float
    top_score: float
    kept_chunks_count: int
    raw_retrieved_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TuningSummary:
    """Aggregated evaluation metrics for a retrieval setting across all test queries."""
    setting_name: str
    setting_config: Dict[str, Any]
    total_queries: int
    hits: int
    hit_rate: float
    top_1_hits: int
    top_1_hit_rate: float
    mrr: float
    avg_top_score: float
    avg_retained_chunks: float
    details: List[EvaluationRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setting_name": self.setting_name,
            "setting_config": self.setting_config,
            "total_queries": self.total_queries,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "top_1_hits": self.top_1_hits,
            "top_1_hit_rate": round(self.top_1_hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "avg_top_score": round(self.avg_top_score, 4),
            "avg_retained_chunks": round(self.avg_retained_chunks, 2),
            "details": [d.to_dict() for d in self.details]
        }


class RetrievalTuner:
    """
    Manages retrieval experiment execution, metric calculation, setting comparison,
    and quantitative selection of optimal retrieval parameters.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        default_collection: str = "knovera_relevance_tuning_chunks"
    ):
        """
        Initialize RetrievalTuner with vector store and retriever components.
        """
        load_dotenv()
        self.generator = embedding_generator or EmbeddingGenerator()
        self.vector_db = vector_db or VectorDatabase(embedding_generator=self.generator)
        self.default_collection = default_collection

        self.top_k_retriever = TopKRetriever(
            vector_db=self.vector_db,
            embedding_generator=self.generator,
            default_collection=self.default_collection
        )
        self.hybrid_retriever = HybridRetriever(
            vector_db=self.vector_db,
            embedding_generator=self.generator,
            default_collection=self.default_collection
        )

    def retrieve_with_setting(
        self,
        query: str,
        setting: RetrievalSetting,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes query retrieval according to the parameters specified in RetrievalSetting.
        """
        target_collection = collection_name or self.default_collection

        if setting.use_hybrid:
            results = self.hybrid_retriever.hybrid_search(
                query=query,
                top_k=setting.k,
                metadata_filter=setting.metadata_filter,
                vector_weight=setting.vector_weight,
                keyword_weight=setting.keyword_weight,
                collection_name=target_collection
            )
        else:
            if setting.metadata_filter:
                results = self.hybrid_retriever.retrieve(
                    query=query,
                    top_k=setting.k,
                    metadata_filter=setting.metadata_filter,
                    collection_name=target_collection
                )
            else:
                results = self.top_k_retriever.retrieve(
                    query=query,
                    k=setting.k,
                    collection_name=target_collection
                )

        # Apply min_score score threshold filter
        kept_results = [r for r in results if r.get("score", 0.0) >= setting.min_score]
        return kept_results

    def evaluate_setting(
        self,
        setting: RetrievalSetting,
        test_queries: List[TestQuery],
        collection_name: Optional[str] = None
    ) -> TuningSummary:
        """
        Evaluates a single retrieval setting across a set of ground-truth test queries.
        """
        if not test_queries:
            raise ValueError("test_queries list cannot be empty.")

        target_collection = collection_name or self.default_collection
        rows: List[EvaluationRow] = []
        hits_count = 0
        top_1_hits_count = 0
        reciprocal_ranks_sum = 0.0
        top_scores_sum = 0.0
        retained_chunks_sum = 0

        for item in test_queries:
            # Raw retrieve without score threshold to get total retrieved count
            if setting.use_hybrid:
                raw_results = self.hybrid_retriever.hybrid_search(
                    query=item.query,
                    top_k=setting.k,
                    metadata_filter=setting.metadata_filter,
                    vector_weight=setting.vector_weight,
                    keyword_weight=setting.keyword_weight,
                    collection_name=target_collection
                )
            elif setting.metadata_filter:
                raw_results = self.hybrid_retriever.retrieve(
                    query=item.query,
                    top_k=setting.k,
                    metadata_filter=setting.metadata_filter,
                    collection_name=target_collection
                )
            else:
                raw_results = self.top_k_retriever.retrieve(
                    query=item.query,
                    k=setting.k,
                    collection_name=target_collection
                )

            # Filter results by min_score
            kept_results = [r for r in raw_results if r.get("score", 0.0) >= setting.min_score]

            sources = [r.get("metadata", {}).get("source", "") for r in kept_results]
            chunk_ids = [r.get("id", "") for r in kept_results]
            scores = [round(r.get("score", 0.0), 4) for r in kept_results]

            # Calculate hit metrics
            expected = item.expected_source
            hit = expected in sources
            top_1_hit = bool(sources and sources[0] == expected)

            # Reciprocal rank (1 / rank of first match)
            reciprocal_rank = 0.0
            if hit:
                match_index = sources.index(expected)
                reciprocal_rank = 1.0 / (match_index + 1)

            top_score = scores[0] if scores else 0.0

            if hit:
                hits_count += 1
            if top_1_hit:
                top_1_hits_count += 1

            reciprocal_ranks_sum += reciprocal_rank
            top_scores_sum += top_score
            retained_chunks_sum += len(kept_results)

            rows.append(EvaluationRow(
                query=item.query,
                expected_source=expected,
                returned_sources=sources,
                returned_chunk_ids=chunk_ids,
                returned_scores=scores,
                hit=hit,
                top_1_hit=top_1_hit,
                reciprocal_rank=round(reciprocal_rank, 4),
                top_score=round(top_score, 4),
                kept_chunks_count=len(kept_results),
                raw_retrieved_count=len(raw_results)
            ))

        total_q = len(test_queries)
        hit_rate = hits_count / total_q
        top_1_hit_rate = top_1_hits_count / total_q
        mrr = reciprocal_ranks_sum / total_q
        avg_top_score = top_scores_sum / total_q
        avg_retained = retained_chunks_sum / total_q

        return TuningSummary(
            setting_name=setting.name,
            setting_config=setting.to_dict(),
            total_queries=total_q,
            hits=hits_count,
            hit_rate=hit_rate,
            top_1_hits=top_1_hits_count,
            top_1_hit_rate=top_1_hit_rate,
            mrr=mrr,
            avg_top_score=avg_top_score,
            avg_retained_chunks=avg_retained,
            details=rows
        )

    def evaluate_all_settings(
        self,
        settings: List[RetrievalSetting],
        test_queries: List[TestQuery],
        collection_name: Optional[str] = None
    ) -> List[TuningSummary]:
        """
        Evaluates multiple retrieval settings against the test dataset.
        """
        summaries: List[TuningSummary] = []
        for setting in settings:
            summary = self.evaluate_setting(
                setting=setting,
                test_queries=test_queries,
                collection_name=collection_name
            )
            summaries.append(summary)
        return summaries

    def select_best_setting(
        self,
        summaries: List[TuningSummary]
    ) -> Tuple[TuningSummary, str]:
        """
        Selects the best performing retrieval setting using multi-criteria optimization:
        1. Maximize Hit Rate (Top priority).
        2. Maximize Top-1 Hit Rate / MRR (Secondary priority).
        3. Minimize unnecessary noise / context overhead (Prefer lower avg_retained_chunks among equal hit rates).
        
        Returns:
            Tuple[TuningSummary, str]: The winning summary and the detailed justification string.
        """
        if not summaries:
            raise ValueError("Summaries list cannot be empty.")

        # Sort candidate summaries by (Hit Rate DESC, Top-1 Hit Rate DESC, MRR DESC, Avg Retained ASC)
        sorted_summaries = sorted(
            summaries,
            key=lambda s: (s.hit_rate, s.top_1_hit_rate, s.mrr, -s.avg_retained_chunks),
            reverse=True
        )

        best = sorted_summaries[0]
        
        # Build quantitative justification text
        justification = (
            f"Setting '{best.setting_name}' achieved the optimal relevance performance with a Hit Rate of "
            f"{best.hit_rate * 100:.1f}% ({best.hits}/{best.total_queries} queries returning the expected source), "
            f"a Top-1 Hit Rate of {best.top_1_hit_rate * 100:.1f}%, and a Mean Reciprocal Rank (MRR) of {best.mrr:.4f}. "
            f"It balances precision and recall while returning an average of {best.avg_retained_chunks:.1f} high-confidence chunks "
            f"per query (avg top score: {best.avg_top_score:.4f})."
        )

        return best, justification

    def format_comparison_table(self, summaries: List[TuningSummary]) -> str:
        """
        Formats evaluation summaries into a clean GFM markdown comparison table.
        """
        lines = [
            "| Retrieval Setting | Hit Rate | Hits | Top-1 Hit Rate | MRR | Avg Top Score | Avg Retained Chunks | Config Overview |",
            "|---|---|---|---|---|---|---|---|"
        ]
        for s in summaries:
            cfg = s.setting_config
            cfg_desc = f"k={cfg.get('k')}, min_score={cfg.get('min_score')}, filter={cfg.get('metadata_filter') is not None}, hybrid={cfg.get('use_hybrid')}"
            line = (
                f"| **`{s.setting_name}`** | **`{s.hit_rate * 100:.1f}%`** | `{s.hits}/{s.total_queries}` | "
                f"`{s.top_1_hit_rate * 100:.1f}%` | `{s.mrr:.4f}` | `{s.avg_top_score:.4f}` | "
                f"`{s.avg_retained_chunks:.1f}` | {cfg_desc} |"
            )
            lines.append(line)
        return "\n".join(lines)

    def export_results(
        self,
        summaries: List[TuningSummary],
        winning_summary: TuningSummary,
        justification: str,
        json_path: str,
        txt_path: str
    ) -> None:
        """
        Exports experiment results and metrics to structured JSON and plain text audit files.
        """
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        payload = {
            "assignment": "3.34 Retrieval Relevance Tuning",
            "total_settings_compared": len(summaries),
            "chosen_setting": winning_summary.setting_name,
            "chosen_justification": justification,
            "summary_metrics": [s.to_dict() for s in summaries]
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        table_md = self.format_comparison_table(summaries)
        txt_lines = [
            "================================================================================",
            "KNOVERA RAG ASSISTANT — RETRIEVAL RELEVANCE TUNING EXPERIMENT LOG",
            "================================================================================",
            f"Total Settings Evaluated: {len(summaries)}",
            f"Chosen Optimal Setting:   {winning_summary.setting_name}",
            f"Justification Summary:    {justification}",
            "--------------------------------------------------------------------------------",
            "COMPARATIVE RESULTS TABLE:",
            table_md,
            "================================================================================",
            "\nDETAILED PER-SETTING QUERY BREAKDOWN:\n"
        ]

        for s in summaries:
            txt_lines.append(f"--- SETTING: {s.setting_name} (Hit Rate: {s.hit_rate*100:.1f}%) ---")
            for row in s.details:
                txt_lines.append(
                    f"  Query: '{row.query}'\n"
                    f"    Expected Source:  {row.expected_source}\n"
                    f"    Returned Sources: {row.returned_sources}\n"
                    f"    Hit: {row.hit} | Top-1 Hit: {row.top_1_hit} | Reciprocal Rank: {row.reciprocal_rank} | Top Score: {row.top_score}\n"
                )
            txt_lines.append("")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
