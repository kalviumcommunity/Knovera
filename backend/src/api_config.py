"""
src/api_config.py

Backward-compatible alias re-exporting APIConfig and get_config from src.config.
"""

from src.config import APIConfig, get_config

__all__ = ["APIConfig", "get_config"]
