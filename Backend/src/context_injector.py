"""
src/context_injector.py

Backward-compatible re-export from src.services.context_injector.
"""

from src.services.context_injector import (
    ContextInjector,
    assemble_context,
    assemble_augmented_context,
    format_chunk,
    build_prompt,
    count_tokens,
    GROUNDED_RAG_PROMPT_TEMPLATE
)

__all__ = [
    "ContextInjector",
    "assemble_context",
    "assemble_augmented_context",
    "format_chunk",
    "build_prompt",
    "count_tokens",
    "GROUNDED_RAG_PROMPT_TEMPLATE"
]
