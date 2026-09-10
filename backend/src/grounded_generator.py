"""
src/grounded_generator.py

Backward-compatible re-export of GroundedGenerator from src.services.grounded_generator.
"""

from src.services.grounded_generator import (
    GroundedGenerator,
    GroundedAnswerGenerator,
    generate_grounded_answer,
    generate_ungrounded_answer,
    verify_grounding,
    compare_grounded_vs_ungrounded,
    answer_query,
    STANDARD_MISSING_CONTEXT_FALLBACK
)

__all__ = [
    "GroundedGenerator",
    "GroundedAnswerGenerator",
    "generate_grounded_answer",
    "generate_ungrounded_answer",
    "verify_grounding",
    "compare_grounded_vs_ungrounded",
    "answer_query",
    "STANDARD_MISSING_CONTEXT_FALLBACK"
]
