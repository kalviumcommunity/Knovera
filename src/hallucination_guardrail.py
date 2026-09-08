"""
src/hallucination_guardrail.py

Hallucination Guardrail & Refusal Handling Engine for Knovera RAG Assistant.
Implements:
1. Weak Retrieval Detection: Evaluates context strength using top similarity score, supporting chunk counts, and average similarity thresholds.
2. Safe Refusal Handling: Intercepts weak or unsupported retrieval states before LLM invocation, returning standardized non-hallucinatory refusals.
3. Configurable Quality Thresholds: Supports custom relevance thresholds (default MIN_TOP_SCORE = 0.72, MIN_SUPPORTING_CHUNKS = 1).
4. Confident Grounded Generation: Allows strong supporting context to pass cleanly to grounded answer generation with citations.
5. Telemetry & Threshold Tuning: Tracks guardrail analytics and provides dynamic threshold evaluation routines.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure Knovera root is in sys.path
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.grounded_generator import generate_grounded_answer
from src.source_tracer import answer_with_citations

logger = logging.getLogger(__name__)

# Default Guardrail Quality Thresholds
MIN_TOP_SCORE: float = 0.72
MIN_SUPPORTING_CHUNKS: int = 1
MIN_AVG_SCORE: float = 0.60
STANDARD_SAFE_REFUSAL: str = "I don't have enough reliable context to answer that."


# ============================================================================
# RETRIEVAL STRENGTH EVALUATION
# ============================================================================

def retrieval_is_strong(
    chunks: List[Dict[str, Any]],
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
    min_avg_score: float = 0.0
) -> bool:
    """
    Evaluates whether the retrieved context is strong enough to support generation.
    
    Args:
        chunks: List of retrieved context chunk dictionaries containing 'score'.
        min_top_score: Minimum required similarity score for top chunk (default: 0.72).
        min_supporting_chunks: Minimum number of chunks matching min_top_score (default: 1).
        min_avg_score: Optional minimum average similarity score across all retrieved chunks.
        
    Returns:
        bool: True if retrieval context is strong, False otherwise.
    """
    if not chunks or len(chunks) == 0:
        return False
        
    # Filter strong chunks meeting top score threshold
    strong_chunks = [chunk for chunk in chunks if chunk.get("score", 0.0) >= min_top_score]
    
    if len(strong_chunks) < min_supporting_chunks:
        return False
        
    if min_avg_score > 0.0:
        scores = [chunk.get("score", 0.0) for chunk in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score < min_avg_score:
            return False
            
    return True


def evaluate_retrieval_quality(
    chunks: List[Dict[str, Any]],
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
    min_avg_score: float = MIN_AVG_SCORE
) -> Dict[str, Any]:
    """
    Performs detailed diagnostic evaluation of retrieval context quality.
    
    Args:
        chunks: List of retrieved context chunk dictionaries.
        min_top_score: Minimum required similarity score threshold.
        min_supporting_chunks: Minimum count of supporting chunks above threshold.
        min_avg_score: Minimum required average similarity score across chunks.
        
    Returns:
        Dict[str, Any]: Detailed evaluation metrics payload including status code and decision reason.
    """
    if not chunks or len(chunks) == 0:
        return {
            "is_strong": False,
            "total_chunks": 0,
            "top_score": 0.0,
            "strong_chunks_count": 0,
            "avg_score": 0.0,
            "refusal_reason": "NO_CHUNKS_RETRIEVED",
            "status_code": "refused_empty_context"
        }
        
    scores = [float(chunk.get("score", 0.0)) for chunk in chunks]
    top_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    strong_chunks = [chunk for chunk in chunks if float(chunk.get("score", 0.0)) >= min_top_score]
    strong_count = len(strong_chunks)
    
    if top_score < min_top_score:
        return {
            "is_strong": False,
            "total_chunks": len(chunks),
            "top_score": round(top_score, 4),
            "strong_chunks_count": strong_count,
            "avg_score": round(avg_score, 4),
            "refusal_reason": f"TOP_SCORE_BELOW_THRESHOLD ({top_score:.4f} < {min_top_score:.4f})",
            "status_code": "refused_low_similarity"
        }
        
    if strong_count < min_supporting_chunks:
        return {
            "is_strong": False,
            "total_chunks": len(chunks),
            "top_score": round(top_score, 4),
            "strong_chunks_count": strong_count,
            "avg_score": round(avg_score, 4),
            "refusal_reason": f"INSUFFICIENT_SUPPORTING_CHUNKS ({strong_count} < {min_supporting_chunks})",
            "status_code": "refused_insufficient_support"
        }
        
    if min_avg_score > 0.0 and avg_score < min_avg_score:
        return {
            "is_strong": False,
            "total_chunks": len(chunks),
            "top_score": round(top_score, 4),
            "strong_chunks_count": strong_count,
            "avg_score": round(avg_score, 4),
            "refusal_reason": f"LOW_AVERAGE_SIMILARITY ({avg_score:.4f} < {min_avg_score:.4f})",
            "status_code": "refused_weak_context"
        }
        
    return {
        "is_strong": True,
        "total_chunks": len(chunks),
        "top_score": round(top_score, 4),
        "strong_chunks_count": strong_count,
        "avg_score": round(avg_score, 4),
        "refusal_reason": "STRONG_CONTEXT",
        "status_code": "answered"
    }


# ============================================================================
# GUARDED ANSWER GENERATION
# ============================================================================

def guarded_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
    min_avg_score: float = MIN_AVG_SCORE,
    refusal_message: Optional[str] = None,
    use_api: bool = False,
    max_context_tokens: int = 5000,
    include_citations: bool = True
) -> Dict[str, Any]:
    """
    Executes guarded RAG answer synthesis. Checks retrieval context quality before passing to LLM.
    Returns safe refusal if context is weak; otherwise returns confident grounded answer.
    
    Args:
        question: User query string.
        chunks: List of retrieved context chunk dictionaries.
        model_name: Optional LLM model identifier.
        client: Optional OpenAI client.
        min_top_score: Similarity threshold for top chunk.
        min_supporting_chunks: Required count of supporting chunks.
        min_avg_score: Required average score threshold.
        refusal_message: Custom refusal answer text.
        use_api: Whether to invoke live LLM endpoint.
        max_context_tokens: Maximum token budget for context assembly.
        include_citations: Whether to compute source citations and citation map.
        
    Returns:
        Dict[str, Any]: Response dictionary with status, answer, sources, citations, and retrieval metrics.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace only.")
        
    # Evaluate retrieval quality first
    quality_eval = evaluate_retrieval_quality(
        chunks=chunks,
        min_top_score=min_top_score,
        min_supporting_chunks=min_supporting_chunks,
        min_avg_score=min_avg_score
    )
    
    if not quality_eval["is_strong"]:
        safe_msg = refusal_message or STANDARD_SAFE_REFUSAL
        logger.info(f"Guardrail triggered for query '{question[:30]}...': {quality_eval['refusal_reason']}")
        return {
            "question": question,
            "answer": safe_msg,
            "sources": [],
            "citations": {},
            "status": quality_eval["status_code"],
            "guardrail_triggered": True,
            "retrieval_metrics": quality_eval,
            "grounding_verified": True,
            "grounding_score": 1.0
        }
        
    # Retrieval context is strong -> Generate confident grounded response
    if include_citations:
        grounded_resp = answer_with_citations(
            question=question,
            chunks=chunks,
            use_api=use_api
        )
    else:
        grounded_resp = generate_grounded_answer(
            question=question,
            retrieved_chunks=chunks,
            model_name=model_name,
            client=client,
            max_context_tokens=max_context_tokens,
            use_api=use_api
        )
        
    # Merge guardrail telemetry into grounded output
    grounded_resp["guardrail_triggered"] = False
    grounded_resp["status"] = "answered"
    grounded_resp["retrieval_metrics"] = quality_eval
    
    return grounded_resp


# ============================================================================
# CLASS-BASED GUARDRAIL MANAGEMENT
# ============================================================================

class HallucinationGuardrail:
    """
    Object-oriented manager for Hallucination Guardrails & Refusal Handling.
    Maintains configurable quality thresholds, logs evaluation telemetry, and executes guarded generation.
    """
    def __init__(
        self,
        min_top_score: float = MIN_TOP_SCORE,
        min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
        min_avg_score: float = MIN_AVG_SCORE,
        refusal_message: str = STANDARD_SAFE_REFUSAL
    ):
        self.min_top_score = min_top_score
        self.min_supporting_chunks = min_supporting_chunks
        self.min_avg_score = min_avg_score
        self.refusal_message = refusal_message
        
        # Telemetry counters
        self.total_evaluated = 0
        self.total_refused = 0
        self.total_answered = 0
        self.refusal_reasons: Dict[str, int] = {}

    def is_strong(self, chunks: List[Dict[str, Any]]) -> bool:
        """Evaluates whether retrieved chunks meet active guardrail criteria."""
        return retrieval_is_strong(
            chunks,
            min_top_score=self.min_top_score,
            min_supporting_chunks=self.min_supporting_chunks,
            min_avg_score=self.min_avg_score
        )

    def evaluate(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates retrieval quality and updates telemetry metrics."""
        eval_res = evaluate_retrieval_quality(
            chunks,
            min_top_score=self.min_top_score,
            min_supporting_chunks=self.min_supporting_chunks,
            min_avg_score=self.min_avg_score
        )
        return eval_res

    def generate(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        client: Optional[Any] = None,
        use_api: bool = False
    ) -> Dict[str, Any]:
        """Executes guarded answer generation and updates telemetry counts."""
        self.total_evaluated += 1
        
        resp = guarded_answer(
            question=question,
            chunks=chunks,
            model_name=model_name,
            client=client,
            min_top_score=self.min_top_score,
            min_supporting_chunks=self.min_supporting_chunks,
            min_avg_score=self.min_avg_score,
            refusal_message=self.refusal_message,
            use_api=use_api
        )
        
        if resp.get("guardrail_triggered", False):
            self.total_refused += 1
            reason = resp.get("retrieval_metrics", {}).get("refusal_reason", "UNKNOWN")
            self.refusal_reasons[reason] = self.refusal_reasons.get(reason, 0) + 1
        else:
            self.total_answered += 1
            
        return resp

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns summarized guardrail execution metrics."""
        refusal_rate = (self.total_refused / self.total_evaluated) if self.total_evaluated > 0 else 0.0
        return {
            "total_evaluated": self.total_evaluated,
            "total_refused": self.total_refused,
            "total_answered": self.total_answered,
            "refusal_rate": round(refusal_rate, 4),
            "refusal_reasons_breakdown": self.refusal_reasons,
            "thresholds": {
                "min_top_score": self.min_top_score,
                "min_supporting_chunks": self.min_supporting_chunks,
                "min_avg_score": self.min_avg_score
            }
        }


# ============================================================================
# THRESHOLD TUNING & BENCHMARKING HELPER
# ============================================================================

def tune_guardrail_thresholds(
    evaluation_queries: List[Dict[str, Any]],
    candidate_thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluates candidate similarity score thresholds against a ground-truth dataset.
    Computes precision, recall, and F1 score for identifying answerable vs unanswerable queries.
    
    Args:
        evaluation_queries: List of dicts with 'query', 'retrieved_chunks', and 'is_answerable' (bool).
        candidate_thresholds: List of similarity score thresholds to test.
        
    Returns:
        Dict[str, Any]: Optimal threshold recommendation and detailed candidate scores.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.50, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]
        
    results = []
    best_f1 = -1.0
    best_threshold = 0.72
    
    for score_thresh in candidate_thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for item in evaluation_queries:
            chunks = item.get("retrieved_chunks", [])
            is_answerable_truth = item.get("is_answerable", True)
            
            predicted_strong = retrieval_is_strong(chunks, min_top_score=score_thresh)
            
            if predicted_strong and is_answerable_truth:
                tp += 1
            elif predicted_strong and not is_answerable_truth:
                fp += 1
            elif not predicted_strong and not is_answerable_truth:
                tn += 1
            else:
                fn += 1
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        entry = {
            "threshold": score_thresh,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        }
        results.append(entry)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = score_thresh
            
    return {
        "best_threshold": best_threshold,
        "best_f1_score": round(best_f1, 4),
        "candidate_evaluations": results
    }
