"""
src/history_manager.py

Backward-compatible re-export from src.services.history_manager.
"""

from src.services.history_manager import HistoryManager, rewrite_followup

__all__ = ["HistoryManager", "rewrite_followup"]
