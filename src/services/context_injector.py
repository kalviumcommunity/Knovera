"""
src/context_injector.py

Context Injection & Prompt Augmentation Engine for Knovera RAG Assistant.
Implements:
1. Grounded Chunk Formatting: Formats retrieved chunks with explicit indexed source markers (`[1] source.md#chunk_0`).
2. Token Budget Enforcement: Dynamically tracks and enforces token budgets using tiktoken to prevent context window overflow.
3. Grounding Prompt Augmentation: Embeds assembled context into strict "answer-only-from-context" prompt templates.
4. Source Attribution Tracing: Retains metadata mapping of included chunks vs dropped/truncated chunks when exceeding budget.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import tiktoken
from dotenv import load_dotenv

# Ensure Knovera root is in sys.path when run directly
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from templates.prompts import render_prompt

logger = logging.getLogger(__name__)

# Default Token Budget Allocations (for standard 8K context windows like GPT-4 / GPT-3.5)
DEFAULT_MAX_CONTEXT_TOKENS = 5000
DEFAULT_RESERVED_ANSWER_TOKENS = 1500
DEFAULT_INSTRUCTION_QUESTION_TOKENS = 800
DEFAULT_SAFETY_BUFFER_TOKENS = 700

# Canonical Grounded Prompt Template with Strict Context Boundaries and Fallback Rules
GROUNDED_RAG_PROMPT_TEMPLATE = """You are a grounded assistant for Knovera.
Answer the question using ONLY the provided context below.
If the answer is not in the context, or if the context is insufficient, explicitly say:
"I don't have enough information in the provided context."
Do not extrapolate or speculate beyond the provided evidence.
When possible, cite sources using the citation markers like [1] or [2].

Context:
{context}

Question:
{question}

Answer:"""


def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Returns a tiktoken encoding instance with fallback."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Computes exact BPE token count for a text string using tiktoken.
    
    Args:
        text: Input string.
        encoding_name: BPE encoding name (default: 'cl100k_base').
        
    Returns:
        int: Number of tokens.
    """
    if not text:
        return 0
    enc = get_tokenizer(encoding_name)
    return len(enc.encode(text))


# ============================================================================
# TASK 1 & 3: FORMAT RETRIEVED CHUNKS & SOURCE MARKERS
# ============================================================================

def format_chunk(
    index: int,
    chunk: Dict[str, Any],
    style: str = "standard"
) -> str:
    """
    Formats an individual retrieved chunk with unambiguous source markers for citation.
    
    Styles:
        - "standard": `[1] source.md#chunk_0`
        - "detailed": `[1] Source: source.md#chunk_0 | Section: Authentication | Category: Guides`
        - "compact": `[1] source.md`
        
    Args:
        index: 1-based chunk position in retrieval ranking.
        chunk: Chunk dictionary containing 'text' and 'metadata'.
        style: Marker styling format.
        
    Returns:
        str: Formatted chunk string with header marker and raw text.
    """
    meta = chunk.get("metadata", {})
    source = chunk.get("source") or meta.get("source") or meta.get("doc_title") or f"document_{index}"
    chunk_index = meta.get("chunk_index", 0)
    section = meta.get("section") or meta.get("section_heading", "")
    category = meta.get("category", "")
    text = chunk.get("text", "").strip()
    
    if style == "detailed":
        header_parts = [f"[{index}] Source: {source}#{chunk_index}"]
        if section:
            header_parts.append(f"Section: '{section}'")
        if category:
            header_parts.append(f"Category: {category}")
        header = " | ".join(header_parts)
    elif style == "compact":
        header = f"[{index}] {source}"
    else:  # standard
        header = f"[{index}] {source}#{chunk_index}"
        
    return f"{header}\n{text}"


# ============================================================================
# TASK 2: TOKEN BUDGET ENFORCEMENT & CONTEXT ASSEMBLY
# ============================================================================

def assemble_context(
    chunks: List[Dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    separator: str = "\n\n---\n\n",
    chunk_style: str = "standard",
    encoding_name: str = "cl100k_base"
) -> Tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assembles retrieved chunks into a single context string while strictly enforcing
    the allocated token budget.
    
    Iterates through chunks in priority order (assumed sorted by retrieval score),
    appends each chunk if it fits within max_tokens, and terminates when the budget
    would be exceeded.
    
    Args:
        chunks: List of retrieved chunk dictionaries.
        max_tokens: Maximum allowed token count for the assembled context.
        separator: Delimiter string between successive chunks.
        chunk_style: Style for chunk formatting ("standard", "detailed", "compact").
        encoding_name: Tokenizer model encoding.
        
    Returns:
        Tuple of:
            - context_str (str): Assembled context text block.
            - used_tokens (int): Total tokens utilized by the assembled context.
            - included_chunks (List[Dict]): List of chunk objects successfully included.
            - dropped_chunks (List[Dict]): List of chunks omitted due to token budget limits.
    """
    if not chunks:
        return "", 0, [], []
        
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer > 0, got {max_tokens}")
        
    selected_formatted = []
    included_chunks = []
    dropped_chunks = []
    used_tokens = 0
    separator_tokens = count_tokens(separator, encoding_name=encoding_name)
    
    for index, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(index, chunk, style=chunk_style)
        chunk_token_count = count_tokens(formatted, encoding_name=encoding_name)
        
        # Calculate marginal tokens needed (including separator if not the first chunk)
        marginal_tokens = chunk_token_count + (separator_tokens if selected_formatted else 0)
        
        if used_tokens + marginal_tokens > max_tokens:
            logger.info(
                f"Token budget reached: Chunk #{index} ({chunk_token_count} tokens) "
                f"exceeds remaining budget ({max_tokens - used_tokens} tokens). Dropping remaining."
            )
            dropped_chunks.append({
                "index": index,
                "chunk": chunk,
                "token_count": chunk_token_count,
                "reason": "EXCEEDED_TOKEN_BUDGET"
            })
            continue
            
        selected_formatted.append(formatted)
        included_chunks.append(chunk)
        used_tokens += marginal_tokens
        
    context_str = separator.join(selected_formatted)
    
    # Recalculate exact total tokens on final string to ensure 100% precision
    actual_tokens = count_tokens(context_str, encoding_name=encoding_name)
    
    return context_str, actual_tokens, included_chunks, dropped_chunks


# ============================================================================
# TASK 4: BUILD GROUNDED AUGMENTED PROMPT
# ============================================================================

def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    prompt_template: str = GROUNDED_RAG_PROMPT_TEMPLATE,
    chunk_style: str = "standard",
    encoding_name: str = "cl100k_base"
) -> Dict[str, Any]:
    """
    Builds a complete, grounded, token-budgeted augmented prompt for downstream LLM generation.
    
    Args:
        question: User natural language question.
        retrieved_chunks: List of retrieved chunk dictionaries.
        max_context_tokens: Maximum token budget for injected context.
        prompt_template: Prompt template containing {context} and {question} variables.
        chunk_style: Formatting style for source markers.
        encoding_name: Tokenizer model encoding.
        
    Returns:
        Dict[str, Any]: Comprehensive prompt payload containing:
            - 'prompt': Full rendered prompt string with grounding instructions.
            - 'context': Assembled context string.
            - 'context_tokens': Number of tokens used by context.
            - 'question_tokens': Number of tokens used by question.
            - 'total_prompt_tokens': Total tokens of the entire prompt payload.
            - 'max_context_tokens': Configured context token budget.
            - 'sources_used': List of metadata dictionaries for included chunks.
            - 'chunks_included_count': Number of chunks included.
            - 'chunks_dropped_count': Number of chunks omitted due to budget.
            - 'dropped_chunks': Audit details of omitted chunks.
    """
    if not question or not question.strip():
        raise ValueError("Question string cannot be empty or whitespace only.")
        
    context, context_tokens, included_chunks, dropped_chunks = assemble_context(
        chunks=retrieved_chunks,
        max_tokens=max_context_tokens,
        chunk_style=chunk_style,
        encoding_name=encoding_name
    )
    
    # Render prompt safely using template
    prompt_text = render_prompt(
        prompt_template,
        context=context if context else "(No relevant context retrieved)",
        question=question.strip()
    )
    
    total_tokens = count_tokens(prompt_text, encoding_name=encoding_name)
    question_tokens = count_tokens(question, encoding_name=encoding_name)
    
    sources_used = []
    for idx, chunk in enumerate(included_chunks, start=1):
        meta = dict(chunk.get("metadata", {}))
        meta["marker"] = f"[{idx}]"
        meta["id"] = chunk.get("id")
        meta["source"] = meta.get("source", chunk.get("source", "Unknown"))
        sources_used.append(meta)
        
    return {
        "prompt": prompt_text,
        "context": context,
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
        "total_prompt_tokens": total_tokens,
        "max_context_tokens": max_context_tokens,
        "sources_used": sources_used,
        "chunks_included_count": len(included_chunks),
        "chunks_dropped_count": len(dropped_chunks),
        "dropped_chunks": dropped_chunks
    }


# ============================================================================
# CONTEXT INJECTOR OBJECT-ORIENTED ENGINE
# ============================================================================

class ContextInjector:
    """
    Manages context injection, token budget allocation, and prompt augmentation
    for Knovera RAG applications.
    """
    
    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        model_context_window: int = 8000,
        reserved_answer_tokens: int = DEFAULT_RESERVED_ANSWER_TOKENS,
        encoding_name: str = "cl100k_base",
        chunk_style: str = "standard"
    ):
        """
        Initialize ContextInjector with explicit token budget limits.
        
        Args:
            max_context_tokens: Upper bound for context tokens (default: 5000).
            model_context_window: Full model context window limit (default: 8000).
            reserved_answer_tokens: Tokens reserved for model generation output (default: 1500).
            encoding_name: BPE encoding name (default: "cl100k_base").
            chunk_style: Source marker style ("standard", "detailed", "compact").
        """
        self.max_context_tokens = max_context_tokens
        self.model_context_window = model_context_window
        self.reserved_answer_tokens = reserved_answer_tokens
        self.encoding_name = encoding_name
        self.chunk_style = chunk_style
        self.tokenizer = get_tokenizer(encoding_name)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens of a string."""
        return count_tokens(text, self.encoding_name)
        
    def format_chunk(self, index: int, chunk: Dict[str, Any]) -> str:
        """Format a single chunk with source marker."""
        return format_chunk(index, chunk, style=self.chunk_style)
        
    def assemble(self, chunks: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> Tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assemble chunks within token budget."""
        limit = max_tokens if max_tokens is not None else self.max_context_tokens
        return assemble_context(
            chunks=chunks,
            max_tokens=limit,
            chunk_style=self.chunk_style,
            encoding_name=self.encoding_name
        )
        
    def augment_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_tokens: Optional[int] = None,
        prompt_template: str = GROUNDED_RAG_PROMPT_TEMPLATE
    ) -> Dict[str, Any]:
        """Builds augmented prompt with complete budget telemetry."""
        limit = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        return build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            max_context_tokens=limit,
            prompt_template=prompt_template,
            chunk_style=self.chunk_style,
            encoding_name=self.encoding_name
        )


def assemble_augmented_context(chunks: List[Dict[str, Any]], max_tokens: int = 5000) -> str:
    """Convenience function returning the assembled context string."""
    ctx_str, _, _, _ = assemble_context(chunks, max_tokens=max_tokens)
    return ctx_str
