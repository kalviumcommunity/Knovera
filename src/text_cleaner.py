"""
src/text_cleaner.py

Backward-compatible re-export of TextCleaner from src.services.text_cleaner.
"""

from src.services.text_cleaner import TextCleaner, clean_text, normalize_whitespace

__all__ = ["TextCleaner", "clean_text", "normalize_whitespace"]
