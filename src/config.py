"""
src/config.py

Centralized Configuration Management for Knovera RAG Backend.
Loads configuration settings from environment variables and .env files with
robust type validation, default fallbacks, safe secret masking, and CORS origins.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Automatically load .env file from project root
_root_dir = Path(__file__).resolve().parent.parent
_env_path = _root_dir / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)
else:
    load_dotenv(override=False)


class APIConfig:
    """
    Centralized configuration management class for Knovera RAG Backend API.
    Reads environment variables dynamically with fallback defaults.
    """

    def __init__(self):
        self.reload()

    def reload(self) -> None:
        """Reload configuration from environment variables."""
        # API Keys & Secrets
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.active_api_key: str = self.openrouter_api_key or self.openai_api_key

        # API Base URLs
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

        # Model Names
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL",
            os.getenv("EMBED_MODEL", "text-embedding-3-small")
        ).strip()
        self.chat_model: str = os.getenv(
            "CHAT_MODEL",
            os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        ).strip()

        # Vector Database Settings
        self.vector_db_url: str = os.getenv(
            "VECTOR_DB_URL",
            os.getenv("CHROMA_PERSIST_DIR", str(_root_dir / "data" / "chroma_db"))
        ).strip()
        self.collection_name: str = os.getenv("COLLECTION_NAME", "knovera_knowledge_base").strip()

        # Server Settings
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0").strip()
        try:
            self.api_port: int = int(os.getenv("API_PORT", "8000"))
        except ValueError:
            self.api_port = 8000

        self.app_env: str = os.getenv("APP_ENV", "development").strip().lower()
        self.api_title: str = os.getenv("API_TITLE", "Knovera RAG Backend API").strip()
        self.api_version: str = os.getenv("API_VERSION", "1.0.0").strip()

        # CORS Allowed Origins (comma-separated or list)
        cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://localhost:8000,*")
        self.cors_origins: List[str] = [orig.strip() for orig in cors_raw.split(",") if orig.strip()]

        # RAG Retrieval & Guardrail Parameters
        try:
            self.top_k: int = int(os.getenv("TOP_K", "4"))
        except ValueError:
            self.top_k = 4

        try:
            self.min_top_score: float = float(os.getenv("MIN_TOP_SCORE", "0.72"))
        except ValueError:
            self.min_top_score = 0.72

        # Upload Settings
        self.upload_dir: str = os.getenv("UPLOAD_DIR", str(_root_dir / "uploads")).strip()
        try:
            self.max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
        except ValueError:
            self.max_upload_size_mb = 10

        self.use_live_api: bool = os.getenv("USE_LIVE_API", "false").lower() in ("true", "1", "yes")

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Args:
            mask_secrets: If True, masks API keys for safe logging and client inspection.
            
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        def mask_key(key: str) -> str:
            if not key:
                return "Not Configured"
            if len(key) <= 8:
                return "***"
            return f"{key[:4]}...{key[-4:]}"

        return {
            "app_env": self.app_env,
            "api_title": self.api_title,
            "api_version": self.api_version,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "cors_origins": self.cors_origins,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "vector_db_url": self.vector_db_url,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "min_top_score": self.min_top_score,
            "use_live_api": self.use_live_api,
            "has_api_key": bool(self.active_api_key),
            "api_key_masked": mask_key(self.active_api_key) if mask_secrets else self.active_api_key
        }


# Global singleton instance
_config_instance: Optional[APIConfig] = None


def get_config(force_reload: bool = False) -> APIConfig:
    """
    Retrieve or initialize the centralized API configuration singleton.
    
    Args:
        force_reload: If True, reloads all settings from environment variables.
        
    Returns:
        APIConfig: Active configuration instance.
    """
    global _config_instance
    if _config_instance is None or force_reload:
        _config_instance = APIConfig()
    return _config_instance
