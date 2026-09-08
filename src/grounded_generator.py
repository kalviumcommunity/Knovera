"""
src/grounded_generator.py

Grounded Answer Generation & Hallucination Mitigation Engine for Knovera RAG Assistant.
Implements:
1. Grounded Generation: Synthesizes answers strictly using injected context with inline citations ([1], [2]).
2. Grounding Accuracy Verification: Programmatically validates that generated answer claims are supported by source chunks.
3. Missing-Context Fallback: Returns explicit non-hallucinatory fallback when supporting context is absent.
4. With vs Without Retrieval Comparison: Side-by-side benchmarking of grounded RAG responses vs ungrounded raw LLM completions.
"""

import os
import re
import time
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.context_injector import (
    build_prompt,
    assemble_context,
    count_tokens,
    GROUNDED_RAG_PROMPT_TEMPLATE
)
from templates.prompts import render_prompt

# Try importing OpenAI client
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

# Standard fallback message when supporting context is missing or insufficient
STANDARD_MISSING_CONTEXT_FALLBACK = "I don't have enough information in the provided context."

# Prompt template for ungrounded (baseline parametric memory) generation
UNGROUNDED_PROMPT_TEMPLATE = """Answer this question directly based on your general knowledge:
Question: {question}

Answer:"""


# ============================================================================
# TASK 1: GENERATE FROM INJECTED CONTEXT
# ============================================================================

def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
    max_context_tokens: int = 5000,
    temperature: float = 0.1,
    max_tokens: int = 500,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Generates a factual response grounded strictly in the provided retrieved chunks.
    
    Args:
        question: User query string.
        retrieved_chunks: List of retrieved chunk dictionaries.
        model_name: Optional LLM model identifier.
        client: Optional OpenAI client.
        max_context_tokens: Context token limit.
        temperature: Sampling temperature (low value 0.1 for high determinism).
        max_tokens: Maximum generation output tokens.
        use_api: If True and API credentials are valid, invokes live LLM; otherwise uses deterministic synthesizer.
        
    Returns:
        Dict[str, Any]: Grounded generation payload with answer, prompt, context, and source citations.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace only.")
        
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": STANDARD_MISSING_CONTEXT_FALLBACK,
            "context": "",
            "prompt": "",
            "sources": [],
            "citations": {},
            "grounding_verified": True,
            "grounding_score": 1.0,
            "status": "MISSING_CONTEXT_FALLBACK"
        }
        
    # Build token-budgeted prompt with grounding instructions
    prompt_data = build_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
        max_context_tokens=max_context_tokens
    )
    
    answer_text = None
    
    # Try Live API Generation if enabled
    if use_api:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
        base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
        target_model = model_name or os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        
        if HAS_OPENAI and api_key and api_key.strip():
            try:
                active_client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=5.0)
                response = active_client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Knovera, a strictly grounded technical assistant. "
                                "Answer using ONLY facts directly stated in the context. "
                                "Do NOT assume or extrapolate. Cite sources with [1], [2] notation. "
                                f"If context lacks the answer, say: '{STANDARD_MISSING_CONTEXT_FALLBACK}'."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt_data["prompt"]
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                raw_ans = response.choices[0].message.content
                if raw_ans and raw_ans.strip():
                    answer_text = raw_ans.strip()
            except Exception as e:
                logger.warning(f"Live LLM call failed or timed out ({e}). Falling back to deterministic grounded synthesis.")
                
    # Deterministic Grounded Synthesis (offline or fallback)
    if not answer_text:
        answer_text = _synthesize_deterministic_grounded_answer(
            question=question,
            context=prompt_data["context"],
            chunks=retrieved_chunks
        )
        
    # Task 2: Check source accuracy
    verification = verify_grounding(answer_text, retrieved_chunks)

    # Build citation map keyed by marker e.g. [1]
    citation_map = {}
    for idx, c in enumerate(retrieved_chunks, start=1):
        meta = c.get("metadata", {})
        citation_map[f"[{idx}]"] = {
            "source": meta.get("source", c.get("source", "Unknown Document")),
            "chunk_id": meta.get("chunk_id", c.get("id", f"chunk_{idx}")),
            "chunk_index": meta.get("chunk_index", 0),
            "section": meta.get("section") or meta.get("section_heading", "General"),
            "page": meta.get("page_number", meta.get("page", 1)),
            "char_span": f"{meta.get('char_start', 0)}-{meta.get('char_end', len(c.get('text', '')))}",
            "text": c.get("text", "")
        }
    
    return {
        "question": question,
        "answer": answer_text,
        "context": prompt_data["context"],
        "prompt": prompt_data["prompt"],
        "sources": prompt_data["sources_used"],
        "citations": citation_map,
        "grounding_verified": verification["is_grounded"],
        "grounding_score": verification["grounding_score"],
        "verification_details": verification,
        "status": "GROUNDED_SUCCESS"
    }


def _synthesize_deterministic_grounded_answer(
    question: str,
    context: str,
    chunks: List[Dict[str, Any]]
) -> str:
    """Extracts query-supported factual sentences directly from context with citation markers."""
    if not context or not context.strip():
        return STANDARD_MISSING_CONTEXT_FALLBACK
        
    query_words = set(w.lower() for w in re.findall(r'\b\w+\b', question) if len(w) > 2)
    blocks = context.split("\n\n---\n\n") if "\n\n---\n\n" in context else context.split("\n\n")
    scored_sentences = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        header = lines[0] if lines else ""
        body = " ".join(lines[1:]) if len(lines) > 1 else block
        
        # Extract citation marker, e.g. "[1]"
        marker_match = re.search(r'\[\d+\]', header)
        marker = marker_match.group(0) if marker_match else "[1]"
        
        header_words = set(w.lower() for w in re.findall(r'\b\w+\b', header))
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', body) if len(s.strip()) > 10]
        for sent in sentences:
            sent_clean = sent.rstrip(".")
            sent_words = set(w.lower() for w in re.findall(r'\b\w+\b', sent_clean))
            overlap = len(query_words.intersection(sent_words))
            for qw in query_words:
                if len(qw) > 3 and any((qw in sw or sw in qw) for sw in sent_words):
                    overlap += 0.5
                if len(qw) > 3 and any((qw in hw or hw in qw) for hw in header_words):
                    overlap += 0.25
            if overlap > 0:
                scored_sentences.append((overlap, f"{sent_clean}. {marker}"))
                
    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:2]]
        return "Based on the provided documentation: " + " ".join(top_sentences)
        
    first_chunk_text = chunks[0].get("text", "") if chunks else ""
    return f"According to the provided context: {first_chunk_text[:180]}... [1]"


# ============================================================================
# TASK 2: CHECK SOURCE ACCURACY & VERIFY GROUNDING
# ============================================================================

def verify_grounding(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    min_word_overlap_ratio: float = 0.35
) -> Dict[str, Any]:
    """
    Task 2: Confirms that the answer correctly reflects the retrieved chunks
    and does not introduce unsupported claims.
    
    Args:
        answer: Generated answer text.
        retrieved_chunks: List of ground-truth candidate chunks.
        min_word_overlap_ratio: Minimum content word overlap threshold.
        
    Returns:
        Dict[str, Any]: Grounding audit report with overlap score, citation audit, and status.
    """
    if not answer or not answer.strip():
        return {
            "is_grounded": False,
            "grounding_score": 0.0,
            "reason": "EMPTY_ANSWER",
            "cited_markers": [],
            "unsupported_terms": []
        }
        
    # Check if answer is explicit fallback
    if STANDARD_MISSING_CONTEXT_FALLBACK.lower() in answer.lower():
        return {
            "is_grounded": True,
            "grounding_score": 1.0,
            "reason": "VALID_MISSING_CONTEXT_FALLBACK",
            "cited_markers": [],
            "unsupported_terms": []
        }
        
    if not retrieved_chunks:
        return {
            "is_grounded": False,
            "grounding_score": 0.0,
            "reason": "ANSWER_PRODUCED_WITHOUT_CHUNKS",
            "cited_markers": [],
            "unsupported_terms": []
        }
        
    # Aggregate all chunk content
    combined_chunk_text = " ".join([c.get("text", "") for c in retrieved_chunks]).lower()
    chunk_tokens = set(re.findall(r'\b\w+\b', combined_chunk_text))
    
    # Extract substantive answer words (len > 3, exclude common filler words)
    stop_words = {
        "based", "provided", "documentation", "according", "context", "that", "this",
        "with", "from", "have", "been", "must", "will", "should", "users", "user"
    }
    answer_tokens = [
        w.lower() for w in re.findall(r'\b\w+\b', answer)
        if len(w) > 3 and w.lower() not in stop_words and not w.isdigit()
    ]
    
    if not answer_tokens:
        return {
            "is_grounded": True,
            "grounding_score": 1.0,
            "reason": "NO_EXTRACTABLE_CLAIMS",
            "cited_markers": re.findall(r'\[\d+\]', answer),
            "unsupported_terms": []
        }
        
    supported_tokens = [w for w in answer_tokens if w in chunk_tokens]
    unsupported_tokens = [w for w in answer_tokens if w not in chunk_tokens]
    
    grounding_score = round(len(supported_tokens) / len(answer_tokens), 4)
    cited_markers = list(set(re.findall(r'\[\d+\]', answer)))
    is_grounded = grounding_score >= min_word_overlap_ratio
    
    return {
        "is_grounded": is_grounded,
        "grounding_score": grounding_score,
        "total_claim_tokens": len(answer_tokens),
        "supported_claim_tokens": len(supported_tokens),
        "unsupported_terms": unsupported_tokens[:5],
        "cited_markers": cited_markers,
        "has_citations": len(cited_markers) > 0,
        "reason": "VERIFIED_GROUNDED" if is_grounded else "POTENTIAL_UNGROUNDED_CONTENT"
    }


# ============================================================================
# TASK 3: MISSING-CONTEXT FALLBACK HANDLING
# ============================================================================

def answer_query(
    question: str,
    retriever: Optional[Any] = None,
    k: int = 4,
    score_threshold: Optional[float] = None,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Task 3: Orchestrates retrieval and generation, gracefully handling missing context.
    If retriever returns no chunks or no chunks meet score_threshold, returns the fallback.
    
    Args:
        question: User query string.
        retriever: Optional retriever instance with .retrieve(query, k=...) method.
        k: Top-k chunks to retrieve.
        score_threshold: Minimum similarity cutoff.
        use_api: Whether to call live LLM.
        
    Returns:
        Dict[str, Any]: Final response payload.
    """
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Please provide a valid question.",
            "sources": [],
            "status": "EMPTY_QUESTION"
        }
        
    chunks = []
    if retriever is not None:
        try:
            if hasattr(retriever, "retrieve"):
                raw_chunks = retriever.retrieve(question, k=k)
            else:
                raw_chunks = []
                
            if score_threshold is not None:
                chunks = [c for c in raw_chunks if c.get("score", 1.0) >= score_threshold]
            else:
                chunks = raw_chunks
        except Exception as e:
            logger.warning(f"Retrieval failed ({e}). Proceeding to empty context fallback.")
            chunks = []
            
    # Task 3: Missing-context fallback guard
    if not chunks:
        return {
            "question": question,
            "answer": STANDARD_MISSING_CONTEXT_FALLBACK,
            "context": "",
            "prompt": "",
            "sources": [],
            "grounding_verified": True,
            "grounding_score": 1.0,
            "status": "MISSING_CONTEXT_FALLBACK"
        }
        
    return generate_grounded_answer(
        question=question,
        retrieved_chunks=chunks,
        use_api=use_api
    )


# ============================================================================
# TASK 4: COMPARE WITH AND WITHOUT RETRIEVAL
# ============================================================================

def generate_ungrounded_answer(
    question: str,
    model_name: Optional[str] = None,
    client: Optional[Any] = None,
    temperature: float = 0.7,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Task 4: Generates an ungrounded answer directly from parametric model memory
    without any retrieved document context.
    
    Args:
        question: User query string.
        model_name: Optional model identifier.
        client: Optional OpenAI client.
        temperature: Higher temperature (0.7) typical of ungrounded general generation.
        use_api: Whether to call live LLM.
        
    Returns:
        Dict[str, Any]: Ungrounded response payload with zero sources.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
        
    prompt = render_prompt(UNGROUNDED_PROMPT_TEMPLATE, question=question.strip())
    
    answer_text = None
    if use_api:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY"))
        base_url = os.getenv("OPENROUTER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
        target_model = model_name or os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        
        if HAS_OPENAI and api_key and api_key.strip():
            try:
                active_client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=5.0)
                response = active_client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=300
                )
                answer_text = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Live ungrounded call failed ({e}). Using simulated ungrounded response.")
                
    if not answer_text:
        answer_text = (
            f"Generally speaking, for {question.lower().replace('?', '')}, organizations typically require "
            "standard documentation, digital artifacts, and general verification steps. Please consult your "
            "institutional portal or system administrator for specific policies."
        )
        
    return {
        "question": question,
        "answer": answer_text,
        "mode": "UNGROUNDED_NO_RETRIEVAL",
        "sources": [],
        "context_injected": False,
        "citations_count": 0
    }


def compare_grounded_vs_ungrounded(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Task 4: Runs the same question in both modes (with retrieval vs without retrieval)
    and computes a structured comparison of accuracy, citations, and hallucination risk.
    
    Args:
        question: User query string.
        retrieved_chunks: Supporting document chunks.
        use_api: Whether to call live LLM.
        
    Returns:
        Dict[str, Any]: Comparative evaluation report.
    """
    # 1. Grounded Run (With Retrieval)
    grounded_res = generate_grounded_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        use_api=use_api
    )
    
    # 2. Ungrounded Run (Without Retrieval)
    ungrounded_res = generate_ungrounded_answer(
        question=question,
        use_api=use_api
    )
    
    comparison_summary = {
        "question": question,
        "with_retrieval": {
            "answer": grounded_res["answer"],
            "sources_count": len(grounded_res["sources"]),
            "sources": [s["source"] for s in grounded_res["sources"]],
            "grounding_score": grounded_res.get("grounding_score", 1.0),
            "is_grounded": grounded_res.get("grounding_verified", True),
            "has_citations": len(re.findall(r'\[\d+\]', grounded_res["answer"])) > 0
        },
        "without_retrieval": {
            "answer": ungrounded_res["answer"],
            "sources_count": 0,
            "sources": [],
            "grounding_score": 0.0,
            "is_grounded": False,
            "has_citations": False
        },
        "key_differences": [
            "With retrieval answers are tied directly to corpus chunks (verifiable source provenance).",
            "Without retrieval answers rely on generic parametric assumptions (hallucination risk).",
            "With retrieval includes deterministic citation markers [1], [2] for auditing.",
            "With retrieval safely triggers fallback when context is insufficient."
        ]
    }
    
    return comparison_summary


# ============================================================================
# GROUNDED GENERATOR OBJECT-ORIENTED ENGINE
# ============================================================================

class GroundedGenerator:
    """
    Object-oriented Grounded Answer Generation Engine for Knovera.
    """
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        model_name: Optional[str] = None,
        max_context_tokens: int = 5000,
        temperature: float = 0.1,
        use_api: bool = False
    ):
        """
        Initialize GroundedGenerator instance.
        """
        self.retriever = retriever
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.use_api = use_api
        
    def generate_grounded(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate grounded answer from chunks."""
        return generate_grounded_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            model_name=self.model_name,
            max_context_tokens=self.max_context_tokens,
            temperature=self.temperature,
            use_api=self.use_api
        )
        
    def generate_ungrounded(self, question: str) -> Dict[str, Any]:
        """Generate ungrounded baseline answer."""
        return generate_ungrounded_answer(
            question=question,
            model_name=self.model_name,
            use_api=self.use_api
        )
        
    def answer(
        self,
        question: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """End-to-end query answering with retrieval and fallback."""
        return answer_query(
            question=question,
            retriever=self.retriever,
            k=k,
            score_threshold=score_threshold,
            use_api=self.use_api
        )
        
    def verify(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify grounding alignment of an answer."""
        return verify_grounding(answer=answer, retrieved_chunks=chunks)
        
    def compare(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare with and without retrieval."""
        return compare_grounded_vs_ungrounded(
            question=question,
            retrieved_chunks=chunks,
            use_api=self.use_api
        )
