"""
src/api/routes/health.py

Health, status, and system configuration endpoints for Knovera RAG Backend.
"""

import datetime
import logging
from fastapi import APIRouter, Depends
from src.config import get_config, APIConfig
from src.models.schemas import HealthResponse, ConfigResponse
from src.services.vector_store import VectorDatabase
from src.services.embedding_generator import EmbeddingGenerator

logger = logging.getLogger("knovera.api.health")
router = APIRouter(tags=["Monitoring & System"])


def get_vector_db(config: APIConfig = Depends(get_config)) -> VectorDatabase:
    generator = EmbeddingGenerator(
        model_name=config.embedding_model,
        api_key=config.active_api_key or None,
        base_url=config.openrouter_base_url if config.openrouter_api_key else config.openai_base_url
    )
    return VectorDatabase(
        persist_dir=config.vector_db_url,
        embedding_generator=generator
    )


@router.get("/health", response_model=HealthResponse)
def health_check(
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> HealthResponse:
    """
    Health & readiness check endpoint.
    Verifies vector store availability and returns active model configurations.
    """
    vector_db_ok = False
    total_chunks = 0
    try:
        if vector_db.is_reachable():
            total_chunks = vector_db.count(config.collection_name)
            vector_db_ok = total_chunks >= 0
    except Exception as e:
        logger.error(f"Healthcheck vector DB probe failed: {e}")
        vector_db_ok = False

    overall_status = "ok" if vector_db_ok else "degraded"

    return HealthResponse(
        status=overall_status,
        service="knovera-rag-backend-api",
        version=config.api_version,
        environment=config.app_env,
        collection_name=config.collection_name,
        total_indexed_chunks=total_chunks,
        embedding_model=config.embedding_model,
        chat_model=config.chat_model,
        vector_db_available=vector_db_ok,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@router.get("/config", response_model=ConfigResponse)
def get_system_config(config: APIConfig = Depends(get_config)) -> ConfigResponse:
    """Returns sanitized runtime configuration with masked credentials and CORS settings."""
    cfg_dict = config.to_dict(mask_secrets=True)
    return ConfigResponse(
        environment=cfg_dict["app_env"],
        api_host=cfg_dict["api_host"],
        api_port=cfg_dict["api_port"],
        cors_origins=cfg_dict["cors_origins"],
        embedding_model=cfg_dict["embedding_model"],
        chat_model=cfg_dict["chat_model"],
        vector_db_url=cfg_dict["vector_db_url"],
        collection_name=cfg_dict["collection_name"],
        top_k=cfg_dict["top_k"],
        min_top_score=cfg_dict["min_top_score"],
        use_live_api=cfg_dict["use_live_api"],
        api_key_status=cfg_dict["api_key_masked"]
    )
