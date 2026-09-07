"""
src/embedding_quality_checker.py

Embedding Quality Assurance & Retrieval Sanity Testing Engine.
Designed for Knovera RAG Assistant.

Key Capabilities:
1. Known Relevance Verification: Evaluates query-chunk pairs against a corpus to verify that expected source chunks rank at the top.
2. Ranking Margin Analysis: Measures the cosine similarity gap between the top relevant chunk and unrelated distractors.
3. Edge Case & Failure Diagnostics: Identifies and explains surprising failure modes (e.g. semantic broadness vs specificity, negation blindspots, keyword overlap bias).
4. Vector Space Incompatibility Auditing: Demonstrates and flags model mismatch anomalies where query and document vectors occupy divergent latent spaces.
5. Structured Sanity Reporting: Outputs metrics including Hit@1, Mean Reciprocal Rank (MRR), and diagnostic logs.
"""

import os
import sys
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from src.embedding_generator import EmbeddingGenerator, cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingQualityChecker:
    """
    Automated testing and validation suite for text embeddings quality and retrieval ranking accuracy.
    """

    def __init__(
        self,
        generator: Optional[EmbeddingGenerator] = None,
        model_name: Optional[str] = None
    ):
        load_dotenv()
        self.generator = generator or EmbeddingGenerator(model_name=model_name)
        self.model_name = self.generator.model_name

    def rank_corpus_for_query(
        self,
        query: str,
        corpus_records: List[Dict[str, Any]],
        query_vector: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ranks all corpus records by cosine similarity against the query vector.
        """
        if not corpus_records:
            return []

        # Generate query vector if not explicitly provided
        q_vec = query_vector if query_vector is not None else self.generator.embed([query])[0]

        scored_records = []
        for record in corpus_records:
            doc_vec = record.get("embedding", [])
            if not doc_vec:
                continue
            sim = cosine_similarity(q_vec, doc_vec)
            scored_records.append({
                "id": record.get("id"),
                "text": record.get("text"),
                "metadata": record.get("metadata", {}),
                "score": round(sim, 4),
                "model": record.get("model", self.model_name)
            })

        # Sort descending by similarity score
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(scored_records, start=1):
            item["rank"] = idx

        return scored_records

    def evaluate_test_case(
        self,
        test_case: Dict[str, Any],
        corpus_records: List[Dict[str, Any]],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Runs retrieval ranking for a single test case and evaluates whether the target chunk is retrieved.
        
        Test case schema:
            - 'query': Search query string
            - 'expected_source': Target source filename (e.g. 'account-guide.md')
            - 'expected_id': Optional exact chunk ID
            - 'category': Optional domain label
            - 'description': Optional test rationale
        """
        query = test_case.get("query", "")
        expected_source = test_case.get("expected_source")
        expected_id = test_case.get("expected_id")

        ranked = self.rank_corpus_for_query(query, corpus_records)
        if not ranked:
            return {
                "query": query,
                "passed": False,
                "rank": None,
                "reciprocal_rank": 0.0,
                "top_source": None,
                "top_score": 0.0,
                "margin": 0.0,
                "note": "Corpus is empty or embedding failed."
            }

        top_match = ranked[0]
        top_source = top_match["metadata"].get("source")
        top_score = top_match["score"]

        # Find target rank
        target_rank = None
        target_score = None
        for item in ranked:
            match_source = expected_source and item["metadata"].get("source") == expected_source
            match_id = expected_id and item.get("id") == expected_id
            if match_source or match_id:
                target_rank = item["rank"]
                target_score = item["score"]
                break

        passed = (target_rank == 1)
        reciprocal_rank = 1.0 / target_rank if target_rank else 0.0

        # Calculate score margin between target and first distractor
        distractor_score = ranked[1]["score"] if len(ranked) > 1 and target_rank == 1 else (ranked[0]["score"] if target_rank != 1 else 0.0)
        margin = round((target_score or 0.0) - distractor_score, 4)

        return {
            "query": query,
            "category": test_case.get("category", "General"),
            "expected_source": expected_source,
            "expected_id": expected_id,
            "top_id": top_match["id"],
            "top_source": top_source,
            "top_score": top_score,
            "target_rank": target_rank,
            "target_score": target_score,
            "passed": passed,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "margin_vs_distractor": margin,
            "top_k_matches": [
                {
                    "rank": r["rank"],
                    "id": r["id"],
                    "source": r["metadata"].get("source"),
                    "section": r["metadata"].get("section"),
                    "score": r["score"],
                    "preview": r["text"][:75] + "..."
                }
                for r in ranked[:top_k]
            ],
            "note": test_case.get("description", "Standard relevance smoke test.")
        }

    def run_sanity_suite(
        self,
        test_cases: List[Dict[str, Any]],
        corpus_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes a complete sanity test suite across all provided test cases.
        Computes aggregate metrics: Total Tests, Passed, Failed, Hit@1 Rate, MRR, Mean Margin.
        """
        results = []
        for case in test_cases:
            res = self.evaluate_test_case(case, corpus_records)
            results.append(res)

        total_tests = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        failed_count = total_tests - passed_count
        hit_at_1 = passed_count / total_tests if total_tests > 0 else 0.0
        mrr = (sum(r["reciprocal_rank"] for r in results) / total_tests) if total_tests > 0 else 0.0

        passed_margins = [r["margin_vs_distractor"] for r in results if r["passed"]]
        avg_margin = round(sum(passed_margins) / len(passed_margins), 4) if passed_margins else 0.0

        return {
            "suite": "Embedding Retrieval Sanity Suite",
            "model": self.model_name,
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": failed_count,
            "hit_at_1_rate": round(hit_at_1 * 100, 2),
            "mrr": round(mrr, 4),
            "avg_positive_margin": avg_margin,
            "status": "PASSED" if failed_count == 0 else "SURPRISING_CASES_DETECTED",
            "results": results
        }

    def simulate_mismatched_model_ranking(
        self,
        query: str,
        corpus_records: List[Dict[str, Any]],
        expected_source: str
    ) -> Dict[str, Any]:
        """
        Demonstrates why embedding queries with a different model / random vector space
        completely corrupts similarity ranking, producing arbitrary scores and wrong matches.
        """
        import numpy as np
        # Generate arbitrary/incompatible vector (simulating mismatched model space)
        rng = np.random.RandomState(1337)
        incompatible_query_vec = rng.normal(0.0, 1.0, 1536)
        incompatible_query_vec = (incompatible_query_vec / np.linalg.norm(incompatible_query_vec)).tolist()

        mismatched_results = self.rank_corpus_for_query(query, corpus_records, query_vector=incompatible_query_vec)
        correct_results = self.rank_corpus_for_query(query, corpus_records)

        top_mismatched = mismatched_results[0] if mismatched_results else {}
        top_correct = correct_results[0] if correct_results else {}

        return {
            "query": query,
            "expected_source": expected_source,
            "correct_model": self.model_name,
            "correct_top_source": top_correct.get("metadata", {}).get("source"),
            "correct_top_score": top_correct.get("score"),
            "correct_passed": top_correct.get("metadata", {}).get("source") == expected_source,
            "mismatched_top_source": top_mismatched.get("metadata", {}).get("source"),
            "mismatched_top_score": top_mismatched.get("score"),
            "mismatched_passed": top_mismatched.get("metadata", {}).get("source") == expected_source,
            "diagnosis": "Mismatched model embeddings project texts into non-aligned geometric coordinates, destroying semantic cosine proximity."
        }
