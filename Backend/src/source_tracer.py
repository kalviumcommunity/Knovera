"""
src/source_tracer.py

Backward-compatible re-export from src.services.source_tracer.
"""

from src.services.source_tracer import (
    SourceTracer,
    build_citation_map,
    build_cited_prompt,
    answer_with_citations,
    extract_citations,
    verify_citation_grounding
)

__all__ = [
    "SourceTracer",
    "build_citation_map",
    "build_cited_prompt",
    "answer_with_citations",
    "extract_citations",
    "verify_citation_grounding"
]
