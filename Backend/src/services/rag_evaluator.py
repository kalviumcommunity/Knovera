"""
src/rag_evaluator.py

RAG Evaluation & Answer Quality Scoring Engine for Knovera RAG Assistant.
Implements end-to-end evaluation of the RAG pipeline across three critical dimensions:
1. Correctness: Does the answer match expected ground-truth answer points?
2. Grounding: Are answer claims strictly supported by retrieved context (or safely refused when context is absent)?
3. Citation Accuracy: Do inline / payload citations accurately match expected source documents?

Also provides failure root-cause diagnosis, aggregate metric summarization, and reporting utilities.
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.rag_pipeline import RAGPipeline, retrieve_context, assemble_context, generate_answer
from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.rag_pipeline import RAGPipeline, retrieve_context, assemble_context, generate_answer

logger = logging.getLogger(__name__)

# Standard fallback markers for context refusal
REFUSAL_MARKERS = [
    "don't have enough",
    "cannot answer",
    "not enough information",
    "refuse",
    "no relevant context",
    "could not find relevant context",
    "insufficient context",
    "not available in the provided documentation"
]


def is_safe_refusal(text: str) -> bool:
    """Checks if text contains a safe refusal message."""
    if not text:
        return False
    text_clean = text.lower()
    return any(marker in text_clean for marker in REFUSAL_MARKERS)



def normalize_source_name(source: str) -> str:
    """
    Normalizes source filenames for flexible matching (e.g. 'submission-rubric.md' -> 'submission_rubric.md').
    
    Args:
        source: Raw source file name or path.
        
    Returns:
        str: Normalized basename string.
    """
    if not source:
        return ""
    basename = Path(source).name.lower()
    # Replace hyphens with underscores for uniform comparison
    normalized = basename.replace("-", "_")
    return normalized


def judge_expected_points(answer: str, expected_points: List[str]) -> float:
    """
    Scores correctness by verifying how many expected key points are present in the answer.
    
    Args:
        answer: Generated answer string.
        expected_points: List of expected factual points/concepts.
        
    Returns:
        float: Fraction of expected points matched (0.0 to 1.0).
    """
    if not expected_points:
        return 1.0
        
    if not answer or not answer.strip():
        return 0.0

    answer_clean = answer.lower()
    matched_count = 0

    for point in expected_points:
        point_clean = point.lower().strip()
        
        # Check direct substring match
        if point_clean in answer_clean:
            matched_count += 1
            continue

        # Check refusal equivalencies if point asks for refusal
        if any(term in point_clean for term in ["refuse", "say not enough", "not enough information"]):
            if any(marker in answer_clean for marker in REFUSAL_MARKERS):
                matched_count += 1
                continue

        # Split point into essential sub-keywords and check if all essential keywords are present
        keywords = [kw for kw in re.findall(r'\b\w+\b', point_clean) if len(kw) > 2]
        if keywords and all(kw in answer_clean for kw in keywords):
            matched_count += 1
            continue
            
        # Check partial keyword match (>= 75% of keywords)
        if keywords:
            hits = sum(1 for kw in keywords if kw in answer_clean)
            if hits / len(keywords) >= 0.75:
                matched_count += 1
                continue

    return round(matched_count / len(expected_points), 4)


def judge_grounding(
    answer: str,
    context: Union[str, List[Dict[str, Any]]],
    status: Optional[str] = None
) -> float:
    """
    Scores grounding by verifying whether answer claims are supported by retrieved context.
    If context is missing/empty:
      - Returns 1.0 if answer correctly refuses to answer.
      - Returns 0.0 if answer hallucinates unsupported facts.
      
    Args:
        answer: Generated answer string.
        context: Context text string or list of retrieved chunk dicts.
        status: Pipeline execution status string.
        
    Returns:
        float: Grounding score between 0.0 and 1.0.
    """
    if isinstance(context, list):
        context_text = " ".join([c.get("text", "") for c in context])
    else:
        context_text = str(context or "")

    answer_clean = answer.lower().strip()
    context_clean = context_text.lower().strip()

    # Case 1: Missing context or empty context
    if not context_clean or status in ["NO_CONTEXT_FOUND", "MISSING_CONTEXT_FALLBACK"]:
        # If the model correctly states missing context / refuses, grounding is 1.0 (safely grounded)
        if any(marker in answer_clean for marker in REFUSAL_MARKERS) or is_safe_refusal(answer):
            return 1.0
        else:
            # Answer made claims without supporting context -> Hallucination!
            return 0.0

    # Case 2: Context is present, check answer claims against context
    # If the answer is a valid refusal when context exists, check if context was actually irrelevant
    if any(marker in answer_clean for marker in REFUSAL_MARKERS):
        return 1.0

    # Extract sentences / key terms from answer
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    grounded_sentences = 0
    for sentence in sentences:
        s_clean = sentence.lower()
        # Extract meaningful terms (length > 3)
        terms = [t for t in re.findall(r'\b\w+\b', s_clean) if len(t) > 3 and t not in [
            "based", "provided", "documentation", "according", "context", "system", "require", "required", "this", "that"
        ]]
        if not terms:
            grounded_sentences += 1
            continue

        matching_terms = sum(1 for t in terms if t in context_clean)
        term_ratio = matching_terms / len(terms)

        if term_ratio >= 0.5:
            grounded_sentences += 1

    grounding_score = grounded_sentences / len(sentences)
    return round(grounding_score, 4)


def check_citations(
    citations: Union[Set[str], List[str]],
    expected_sources: Union[Set[str], List[str]]
) -> float:
    """
    Checks citation accuracy against expected ground-truth sources.
    
    Args:
        citations: List or set of source names cited in the RAG payload.
        expected_sources: List or set of expected ground-truth source names.
        
    Returns:
        float: Citation accuracy score (0.0 to 1.0).
    """
    citations_norm = {normalize_source_name(s) for s in citations if s}
    expected_norm = {normalize_source_name(s) for s in expected_sources if s}

    # Case 1: No sources expected (e.g. refusal / out of context query)
    if not expected_norm:
        if not citations_norm:
            return 1.0
        else:
            # Over-citing / hallucinatory citation when no sources support the answer
            return 0.0

    # Case 2: Expected sources present
    if not citations_norm:
        return 0.0

    # Calculate overlap recall: fraction of expected sources correctly cited
    correct_citations = citations_norm & expected_norm
    recall = len(correct_citations) / len(expected_norm)

    # Calculate precision: fraction of cited sources that were expected
    precision = len(correct_citations) / len(citations_norm)

    # Harmonic mean F1 score for citation accuracy
    if recall + precision == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def diagnose_failure_cause(
    correctness: float,
    grounding: float,
    citation_accuracy: float,
    num_retrieved: int = 0
) -> Optional[str]:
    """
    Diagnoses the root cause of a quality failure.
    
    Args:
        correctness: Correctness score.
        grounding: Grounding score.
        citation_accuracy: Citation accuracy score.
        num_retrieved: Number of retrieved chunks.
        
    Returns:
        Optional[str]: Diagnostic failure category string, or None if no failure.
    """
    if min(correctness, grounding, citation_accuracy) >= 1.0:
        return None

    if num_retrieved == 0 and correctness < 1.0:
        return "WEAK_RETRIEVAL_MISSING_CONTEXT"
        
    if correctness < 1.0 and grounding < 1.0:
        return "WEAK_RETRIEVAL_AND_HALLUCINATION"

    if correctness < 1.0:
        return "MISSING_EXPECTED_ANSWER_POINTS"

    if grounding < 1.0:
        return "UNSUPPORTED_DETAILS_OR_OVERGENERATION"

    if citation_accuracy < 1.0:
        return "BAD_CITATIONS_OR_MISSING_METADATA"

    return "UNSPECIFIED_QUALITY_DEGRADATION"


class RAGEvaluator:
    """
    End-to-End RAG System Evaluation Engine.
    Runs test sets through the RAG pipeline, calculates correctness, grounding,
    and citation accuracy, diagnoses failure modes, and summarizes quality.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        generator: Optional[EmbeddingGenerator] = None,
        collection_name: str = "knovera_knowledge_base",
        model_name: Optional[str] = None,
        use_api: bool = False
    ):
        """
        Initializes the RAG Evaluator.
        
        Args:
            vector_db: VectorDatabase instance.
            generator: EmbeddingGenerator instance.
            collection_name: Vector store collection name.
            model_name: Optional LLM model identifier.
            use_api: Whether to call live external LLM API.
        """
        self.generator = generator or EmbeddingGenerator()
        self.vector_db = vector_db or VectorDatabase(embedding_generator=self.generator)
        self.collection_name = collection_name
        self.model_name = model_name
        self.use_api = use_api
        self.pipeline = RAGPipeline(
            vector_db=self.vector_db,
            generator=self.generator,
            default_collection=self.collection_name,
            model_name=self.model_name,
            use_api=self.use_api
        )

    def answer_with_citations(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Runs the RAG pipeline on a question and extracts answer + cited sources.
        
        Args:
            question: User question string.
            k: Top-k chunks to retrieve.
            
        Returns:
            Dict[str, Any]: Payload containing 'answer', 'citations' (Set[str]), 'context', 'sources'.
        """
        res = self.pipeline.query(query=question, k=k)
        answer_text = res.get("answer", "")
        sources_list = res.get("sources", [])
        
        # Extract cited source filenames using inline citation markers [1], [2] if present
        citations = set()
        cited_indices = re.findall(r'\[(\d+)\]', answer_text)
        if cited_indices:
            for idx_str in cited_indices:
                idx = int(idx_str) - 1
                if 0 <= idx < len(sources_list):
                    src = sources_list[idx]
                    source_file = src.get("source") or src.get("doc_title") or src.get("id", "").split(":")[0]
                    if source_file:
                        citations.add(Path(source_file).name)
        else:
            for src in sources_list:
                source_file = src.get("source") or src.get("doc_title") or src.get("id", "").split(":")[0]
                if source_file:
                    citations.add(Path(source_file).name)

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "sources": sources_list,
            "context": res.get("context", ""),
            "num_retrieved": res.get("num_retrieved", 0),
            "status": res.get("status", "SUCCESS")
        }

    def score_answer(
        self,
        example: Dict[str, Any],
        result_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scores a single test set question on correctness, grounding, and citation accuracy.
        
        Args:
            example: Test example dict containing 'question', 'expected_points', 'expected_sources'.
            result_override: Optional pre-computed pipeline response dict.
            
        Returns:
            Dict[str, Any]: Scored answer payload with evaluation metrics.
        """
        question = example["question"]
        expected_points = example.get("expected_points", [])
        expected_sources = example.get("expected_sources", set())

        if result_override:
            result = result_override
        else:
            result = self.answer_with_citations(question)

        answer = result.get("answer", "")
        citations = result.get("citations", set())
        context = result.get("context", "")
        status = result.get("status", "SUCCESS")
        num_retrieved = result.get("num_retrieved", 0)

        # 1. Correctness
        correctness = judge_expected_points(answer=answer, expected_points=expected_points)

        # 2. Grounding
        grounding = judge_grounding(answer=answer, context=context, status=status)

        # 3. Citation accuracy
        citation_accuracy = check_citations(citations=citations, expected_sources=expected_sources)

        # 4. Diagnose failure cause
        failure_cause = diagnose_failure_cause(
            correctness=correctness,
            grounding=grounding,
            citation_accuracy=citation_accuracy,
            num_retrieved=num_retrieved
        )

        return {
            "question": question,
            "answer": answer,
            "correctness": correctness,
            "grounding": grounding,
            "citation_accuracy": citation_accuracy,
            "citations": sorted(list(citations)),
            "expected_points": expected_points,
            "expected_sources": sorted(list(expected_sources)),
            "num_retrieved": num_retrieved,
            "status": status,
            "failure_cause": failure_cause
        }

    def evaluate_test_set(
        self,
        test_set: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates a complete test set of examples and returns individual row scores + aggregate summary.
        
        Args:
            test_set: List of test example dicts.
            
        Returns:
            Dict[str, Any]: Payload with 'rows' and 'summary'.
        """
        rows = [self.score_answer(example) for example in test_set]
        summary = self.summarize_evaluations(rows)
        return {
            "rows": rows,
            "summary": summary
        }

    @staticmethod
    def summarize_evaluations(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarizes overall quality metrics and extracts notable failure cases.
        
        Args:
            rows: List of scored answer dictionaries.
            
        Returns:
            Dict[str, Any]: Aggregate summary dictionary.
        """
        if not rows:
            return {
                "questions": 0,
                "avg_correctness": 0.0,
                "avg_grounding": 0.0,
                "avg_citation_accuracy": 0.0,
                "overall_quality_score": 0.0,
                "weakest_dimension": "N/A",
                "failures": []
            }

        n = len(rows)
        avg_correctness = round(sum(r["correctness"] for r in rows) / n, 4)
        avg_grounding = round(sum(r["grounding"] for r in rows) / n, 4)
        avg_citation_accuracy = round(sum(r["citation_accuracy"] for r in rows) / n, 4)

        overall_score = round((avg_correctness + avg_grounding + avg_citation_accuracy) / 3.0, 4)

        # Identify weakest dimension
        dims = {
            "correctness": avg_correctness,
            "grounding": avg_grounding,
            "citation_accuracy": avg_citation_accuracy
        }
        weakest_dimension = min(dims, key=dims.get)

        failures = [
            r for r in rows
            if min(r["correctness"], r["grounding"], r["citation_accuracy"]) < 1.0
        ]

        return {
            "questions": n,
            "avg_correctness": avg_correctness,
            "avg_grounding": avg_grounding,
            "avg_citation_accuracy": avg_citation_accuracy,
            "overall_quality_score": overall_score,
            "weakest_dimension": weakest_dimension,
            "failures_count": len(failures),
            "failures": failures
        }

    @staticmethod
    def calculate_recall(retrieved: List[str], ground_truth: List[str]) -> float:
        """Calculate recall of retrieved documents."""
        if not ground_truth:
            return 1.0
        r_set = set(retrieved)
        gt_set = set(ground_truth)
        return len(r_set & gt_set) / len(gt_set)

    @staticmethod
    def calculate_precision_at_k(retrieved: List[str], ground_truth: List[str], k: int = 2) -> float:
        """Calculate precision at K."""
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        gt_set = set(ground_truth)
        return sum(1 for doc in top_k if doc in gt_set) / len(top_k)

    @staticmethod
    def calculate_mrr(retrieved: List[str], ground_truth: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        gt_set = set(ground_truth)
        for idx, doc in enumerate(retrieved, start=1):
            if doc in gt_set:
                return 1.0 / idx
        return 0.0
