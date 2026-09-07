"""
src/retrieval_evaluator.py

Enterprise Retrieval Evaluation & Recall Testing Engine.
Designed for Knovera RAG Assistant using ChromaDB Vector Store.

Key Capabilities:
1. Labelled Query Set Management: Maps test queries to known relevant chunk IDs across domain categories.
2. Recall@k & Precision@k Metric Calculation: Computes exact Recall, Precision, F1-Score, MRR, and Hit Rate.
3. Multi-K Metric Comparison: Evaluates retrieval performance across varying top-k values (k=1, 3, 5, 10).
4. Automated Failure Inspection: Identifies low-scoring or failed queries (recall < 1.0) and diagnoses root causes.
5. Actionable Quality Diagnostics & Reporting: Generates structured GFM markdown tables, JSON metrics, and TXT audit logs.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Union
from dotenv import load_dotenv

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.top_k_retriever import TopKRetriever

logger = logging.getLogger(__name__)


@dataclass
class LabelledQuery:
    """Represents a test query paired with ground-truth relevant chunk IDs."""
    __test__ = False  # Prevent Pytest from treating this dataclass as a test class
    query: str
    relevant_chunk_ids: Union[Set[str], List[str]]
    category: str = "General"
    description: str = ""

    def __post_init__(self):
        if isinstance(self.relevant_chunk_ids, list):
            self.relevant_chunk_ids = set(self.relevant_chunk_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "relevant_chunk_ids": sorted(list(self.relevant_chunk_ids)),
            "category": self.category,
            "description": self.description
        }


@dataclass
class QueryEvaluationResult:
    """Detailed evaluation metrics for a single query."""
    query: str
    retrieved_ids: List[str]
    relevant_chunk_ids: List[str]
    hits: List[str]
    recall: float
    precision: float
    f1_score: float
    reciprocal_rank: float
    hit_at_k: bool
    scores: List[float]
    sources: List[str]
    failure_cause: Optional[str] = None
    category: str = "General"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "hits": self.hits,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1_score": round(self.f1_score, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "hit_at_k": self.hit_at_k,
            "scores": [round(s, 4) for s in self.scores],
            "sources": self.sources,
            "failure_cause": self.failure_cause,
            "category": self.category
        }


@dataclass
class AggregateMetrics:
    """Aggregated retrieval evaluation metrics across a labelled query dataset."""
    total_queries: int
    k: int
    avg_recall: float
    avg_precision: float
    avg_f1: float
    mrr: float
    hit_rate: float
    failed_queries_count: int
    passed_queries_count: int
    results: List[QueryEvaluationResult] = field(default_factory=list)
    failure_analysis: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "k": self.k,
            "avg_recall": round(self.avg_recall, 4),
            "avg_precision": round(self.avg_precision, 4),
            "avg_f1": round(self.avg_f1, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "failed_queries_count": self.failed_queries_count,
            "passed_queries_count": self.passed_queries_count,
            "results": [r.to_dict() for r in self.results],
            "failure_analysis": self.failure_analysis
        }


class RetrievalEvaluator:
    """
    Evaluates vector retrieval quality using labelled query benchmarks.
    Computes recall@k, precision@k, F1-Score, MRR, and diagnoses failure causes.
    """

    DEFAULT_COLLECTION_NAME = "knovera_eval_chunks"

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        default_collection: str = DEFAULT_COLLECTION_NAME
    ):
        """
        Initialize RetrievalEvaluator.
        """
        load_dotenv()
        self.generator = embedding_generator or EmbeddingGenerator()
        self.vector_db = vector_db or VectorDatabase(embedding_generator=self.generator)
        self.default_collection = default_collection

        self.retriever = TopKRetriever(
            vector_db=self.vector_db,
            embedding_generator=self.generator,
            default_collection=self.default_collection
        )

    def evaluate_query(
        self,
        query_item: Union[Dict[str, Any], LabelledQuery],
        k: int = 5,
        collection_name: Optional[str] = None
    ) -> QueryEvaluationResult:
        """
        Evaluates retrieval quality for a single query against target vector collection.

        Args:
            query_item: Dictionary or LabelledQuery instance.
            k: Top-k number of chunks to retrieve.
            collection_name: Vector store collection name.

        Returns:
            QueryEvaluationResult containing recall, precision, hits, and failure cause.
        """
        target_collection = collection_name or self.default_collection

        if isinstance(query_item, LabelledQuery):
            query_text = query_item.query
            relevant_set = set(query_item.relevant_chunk_ids)
            category = query_item.category
        elif isinstance(query_item, dict):
            query_text = query_item["query"]
            rel = query_item.get("relevant_chunk_ids", set())
            relevant_set = set(rel) if not isinstance(rel, set) else rel
            category = query_item.get("category", "General")
        else:
            raise TypeError("query_item must be a dict or LabelledQuery instance.")

        results = self.retriever.retrieve(query=query_text, k=k, collection_name=target_collection)

        retrieved_ids = [r["id"] for r in results]
        scores = [r.get("score", 0.0) for r in results]
        sources = [r.get("metadata", {}).get("source", "") for r in results]

        # Calculate hits
        hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_set]

        # Compute metric equations
        recall = len(hits) / len(relevant_set) if len(relevant_set) > 0 else 0.0
        precision = len(hits) / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
        
        if (precision + recall) > 0.0:
            f1_score = (2.0 * precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Reciprocal Rank calculation (1 / position of first hit, 1-indexed)
        reciprocal_rank = 0.0
        hit_at_k = len(hits) > 0
        for rank_idx, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_set:
                reciprocal_rank = 1.0 / rank_idx
                break

        # Diagnose failure cause if recall < 1.0
        failure_cause = None
        if recall < 1.0:
            failure_cause = self._diagnose_single_failure(
                k=k,
                relevant_set=relevant_set,
                retrieved_ids=retrieved_ids,
                hits=hits,
                scores=scores
            )

        return QueryEvaluationResult(
            query=query_text,
            retrieved_ids=retrieved_ids,
            relevant_chunk_ids=sorted(list(relevant_set)),
            hits=hits,
            recall=recall,
            precision=precision,
            f1_score=f1_score,
            reciprocal_rank=reciprocal_rank,
            hit_at_k=hit_at_k,
            scores=scores,
            sources=sources,
            failure_cause=failure_cause,
            category=category
        )

    def _diagnose_single_failure(
        self,
        k: int,
        relevant_set: Set[str],
        retrieved_ids: List[str],
        hits: List[str],
        scores: List[float]
    ) -> str:
        """Categorizes the likely cause of retrieval failure for a query."""
        if not hits:
            if scores and max(scores) < 0.4:
                return "LOW_SEMANTIC_SIMILARITY: Query vocabulary or embedding representation has weak match with indexed chunks."
            return "ZERO_HITS_EMBEDDING_MISMATCH: Target relevant chunk was not present in top-k results."
        
        if len(relevant_set) > k:
            return f"TOO_SMALL_K: Relevant target set size ({len(relevant_set)}) exceeds top-k cutoff ({k})."
        
        if len(hits) < len(relevant_set):
            return f"RANKING_NOISE_OVERCROWDING: Only {len(hits)} of {len(relevant_set)} relevant chunks ranked in top-{k} due to distractor noise."

        return "UNCATEGORIZED_RETRIEVAL_GAP"

    def evaluate_queries(
        self,
        labelled_queries: List[Union[Dict[str, Any], LabelledQuery]],
        k: int = 5,
        collection_name: Optional[str] = None
    ) -> AggregateMetrics:
        """
        Evaluates a batch of labelled queries and calculates aggregate metrics.

        Args:
            labelled_queries: List of LabelledQuery objects or dictionaries.
            k: Top-k cutoff integer (default: 5).
            collection_name: Target vector collection name.

        Returns:
            AggregateMetrics containing recall@k, precision@k, MRR, and failure breakdown.
        """
        if not labelled_queries:
            raise ValueError("labelled_queries list cannot be empty.")

        target_collection = collection_name or self.default_collection
        eval_results: List[QueryEvaluationResult] = []

        for item in labelled_queries:
            res = self.evaluate_query(query_item=item, k=k, collection_name=target_collection)
            eval_results.append(res)

        total_q = len(eval_results)
        avg_recall = sum(r.recall for r in eval_results) / total_q
        avg_precision = sum(r.precision for r in eval_results) / total_q
        avg_f1 = sum(r.f1_score for r in eval_results) / total_q
        mrr = sum(r.reciprocal_rank for r in eval_results) / total_q
        hit_rate = sum(1 for r in eval_results if r.hit_at_k) / total_q

        failed_count = sum(1 for r in eval_results if r.recall < 1.0)
        passed_count = total_q - failed_count

        aggregate = AggregateMetrics(
            total_queries=total_q,
            k=k,
            avg_recall=avg_recall,
            avg_precision=avg_precision,
            avg_f1=avg_f1,
            mrr=mrr,
            hit_rate=hit_rate,
            failed_queries_count=failed_count,
            passed_queries_count=passed_count,
            results=eval_results
        )

        aggregate.failure_analysis = self.inspect_failures(aggregate)
        return aggregate

    def evaluate_across_k(
        self,
        labelled_queries: List[Union[Dict[str, Any], LabelledQuery]],
        k_values: Optional[List[int]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[int, AggregateMetrics]:
        """
        Evaluates the same labelled query set across multiple top-k settings.

        Args:
            labelled_queries: Dataset of labelled queries.
            k_values: List of k integers (default: [1, 3, 5, 10]).
            collection_name: Target collection name.

        Returns:
            Dict[int, AggregateMetrics]: Mapping of k -> AggregateMetrics.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        target_collection = collection_name or self.default_collection
        k_results: Dict[int, AggregateMetrics] = {}

        for k in k_values:
            metrics = self.evaluate_queries(
                labelled_queries=labelled_queries,
                k=k,
                collection_name=target_collection
            )
            k_results[k] = metrics

        return k_results

    def inspect_failures(self, aggregate_metrics: AggregateMetrics) -> List[Dict[str, Any]]:
        """
        Extracts and analyzes failed or low-recall queries, recommending solutions.

        Args:
            aggregate_metrics: AggregateMetrics object containing query results.

        Returns:
            List[Dict[str, Any]]: Detailed failure diagnostics and actionable fixes.
        """
        failures = []
        for r in aggregate_metrics.results:
            if r.recall < 1.0:
                missing_chunks = [cid for cid in r.relevant_chunk_ids if cid not in r.hits]
                
                # Determine actionable recommendation based on cause
                recommendation = "Increase top-k cutoff or implement two-stage cross-encoder re-ranking."
                if "TOO_SMALL_K" in str(r.failure_cause):
                    recommendation = f"Increase top-k parameter from {aggregate_metrics.k} to at least {len(r.relevant_chunk_ids)}."
                elif "ZERO_HITS" in str(r.failure_cause) or "LOW_SEMANTIC" in str(r.failure_cause):
                    recommendation = "Implement hybrid search (vector + BM25 keyword matching) or query rewriting/expansion."
                elif "RANKING_NOISE" in str(r.failure_cause):
                    recommendation = "Apply metadata filtering to narrow search space, or adjust chunking size and overlap."

                failures.append({
                    "query": r.query,
                    "category": r.category,
                    "recall": round(r.recall, 4),
                    "precision": round(r.precision, 4),
                    "expected_chunk_ids": r.relevant_chunk_ids,
                    "retrieved_ids": r.retrieved_ids,
                    "hits": r.hits,
                    "missing_chunk_ids": missing_chunks,
                    "diagnosed_cause": r.failure_cause or "PARTIAL_RECALL_GAP",
                    "actionable_recommendation": recommendation
                })

        return failures

    def generate_summary_report(self, aggregate_metrics: AggregateMetrics) -> str:
        """
        Generates a formatted GitHub Flavored Markdown evaluation summary report.
        """
        m = aggregate_metrics
        lines = [
            f"### Retrieval Evaluation Summary (top-k={m.k})",
            "",
            "| Metric | Score / Value | Status / Benchmark Target |",
            "|---|---|---|",
            f"| **Total Evaluated Queries** | `{m.total_queries}` | 100% Executed |",
            f"| **Recall@{m.k} (Avg)** | **`{m.avg_recall * 100:.1f}%`** | Target > 80.0% |",
            f"| **Precision@{m.k} (Avg)** | `{m.avg_precision * 100:.1f}%` | Noise Control Indicator |",
            f"| **F1-Score (Avg)** | `{m.avg_f1:.4f}` | Balanced Metric |",
            f"| **MRR (Mean Reciprocal Rank)** | `{m.mrr:.4f}` | Rank Quality |",
            f"| **Hit Rate@{m.k}** | `{m.hit_rate * 100:.1f}%` | At least 1 match retrieved |",
            f"| **Passed Queries (Recall=1.0)** | `{m.passed_queries_count}/{m.total_queries}` | Perfect Recall |",
            f"| **Failed Queries (Recall<1.0)** | `{m.failed_queries_count}/{m.total_queries}` | Requires Analysis |",
            "",
            "#### Per-Query Evaluation Breakdown:",
            "| Category | Query | Recall | Precision | Hits | Expected Chunks | Retrieved Chunks |",
            "|---|---|---|---|---|---|---|"
        ]

        for r in m.results:
            rel_str = ", ".join(r.relevant_chunk_ids)
            ret_str = ", ".join(r.retrieved_ids) if r.retrieved_ids else "None"
            hits_str = ", ".join(r.hits) if r.hits else "None"
            line = (
                f"| {r.category} | `{r.query}` | `{r.recall*100:.0f}%` | `{r.precision*100:.0f}%` | "
                f"`{hits_str}` | `{rel_str}` | `{ret_str}` |"
            )
            lines.append(line)

        return "\n".join(lines)

    def export_results(
        self,
        aggregate_metrics: AggregateMetrics,
        json_path: str,
        txt_path: str
    ) -> None:
        """
        Exports evaluation metrics and failure inspection logs to JSON and TXT audit files.
        """
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        payload = {
            "assignment": "3.36 Retrieval Evaluation & Recall Testing",
            "k": aggregate_metrics.k,
            "total_queries": aggregate_metrics.total_queries,
            "summary_metrics": {
                "avg_recall": round(aggregate_metrics.avg_recall, 4),
                "avg_precision": round(aggregate_metrics.avg_precision, 4),
                "avg_f1": round(aggregate_metrics.avg_f1, 4),
                "mrr": round(aggregate_metrics.mrr, 4),
                "hit_rate": round(aggregate_metrics.hit_rate, 4),
                "passed_queries_count": aggregate_metrics.passed_queries_count,
                "failed_queries_count": aggregate_metrics.failed_queries_count
            },
            "failure_analysis": aggregate_metrics.failure_analysis,
            "detailed_results": [r.to_dict() for r in aggregate_metrics.results]
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary_md = self.generate_summary_report(aggregate_metrics)
        txt_lines = [
            "================================================================================",
            "KNOVERA RAG ASSISTANT — RETRIEVAL EVALUATION & RECALL TESTING AUDIT LOG",
            "================================================================================",
            f"Top-K Cutoff Evaluated: k={aggregate_metrics.k}",
            f"Total Queries Evaluated: {aggregate_metrics.total_queries}",
            f"Average Recall@{aggregate_metrics.k}:    {aggregate_metrics.avg_recall * 100:.2f}%",
            f"Average Precision@{aggregate_metrics.k}: {aggregate_metrics.avg_precision * 100:.2f}%",
            f"Mean Reciprocal Rank (MRR): {aggregate_metrics.mrr:.4f}",
            f"Hit Rate@{aggregate_metrics.k}:         {aggregate_metrics.hit_rate * 100:.2f}%",
            "--------------------------------------------------------------------------------",
            "SUMMARY REPORT (GFM Markdown):",
            summary_md,
            "================================================================================",
            "\nDETAILED FAILURE INSPECTION & DIAGNOSTICS:\n"
        ]

        if aggregate_metrics.failure_analysis:
            for idx, fail in enumerate(aggregate_metrics.failure_analysis, start=1):
                txt_lines.append(
                    f"--- FAILURE #{idx}: '{fail['query']}' (Category: {fail['category']}) ---\n"
                    f"  Recall@{aggregate_metrics.k}: {fail['recall'] * 100:.1f}%\n"
                    f"  Precision@{aggregate_metrics.k}: {fail['precision'] * 100:.1f}%\n"
                    f"  Expected Chunks: {fail['expected_chunk_ids']}\n"
                    f"  Retrieved Chunks: {fail['retrieved_ids']}\n"
                    f"  Hits: {fail['hits']}\n"
                    f"  Missing Chunks: {fail['missing_chunk_ids']}\n"
                    f"  Diagnosed Cause: {fail['diagnosed_cause']}\n"
                    f"  Actionable Recommendation: {fail['actionable_recommendation']}\n"
                )
        else:
            txt_lines.append("No retrieval failures detected! All test queries achieved 100% Recall.\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
