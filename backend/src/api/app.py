"""
src/api/app.py

FastAPI Application Factory for Knovera RAG Backend.
Configures CORS for Next.js and web clients, request processing time middleware,
structured exception handlers, and attaches all API routers.
"""

import time
import datetime
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from src.config import get_config, APIConfig
from src.api.routes.health import router as health_router
from src.api.routes.query import router as query_router
from src.api.routes.documents import router as documents_router
from src.api.routes.evaluation import router as eval_router
from src.api.routes.guardrails import router as guardrails_router
from src.api.routes.logs import router as logs_router
from src.api.routes.chunks import router as chunks_router
from src.api.routes.conversations import router as conversations_router
from src.api.routes.dashboard import router as dashboard_router

logger = logging.getLogger("knovera.api")


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """
    Creates and configures the FastAPI application instance.
    
    Args:
        config: Optional pre-configured APIConfig instance.
        
    Returns:
        FastAPI: Configured web application instance.
    """
    cfg = config or get_config()

    app = FastAPI(
        title=cfg.api_title,
        version=cfg.api_version,
        description=(
            "Knovera RAG Backend API: High-performance, production-ready REST service "
            "for grounded Question Answering with strict hallucination guardrails and source citations."
        ),
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # ------------------------------------------------------------------------
    # CORS MIDDLEWARE FOR NEXT.JS & WEB CLIENTS
    # ------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins if cfg.cors_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------------
    # PROCESS TIME MIDDLEWARE
    # ------------------------------------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(process_time)
        return response

    # ------------------------------------------------------------------------
    # CUSTOM EXCEPTION HANDLERS
    # ------------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        error_messages = []
        for err in errors:
            loc = " -> ".join([str(p) for p in err.get("loc", [])])
            msg = err.get("msg", "Validation error")
            error_messages.append(f"{loc}: {msg}")
            
        detail_msg = "; ".join(error_messages)
        logger.warning(f"Validation failure on {request.url.path}: {detail_msg}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": detail_msg,
                "error_code": "VALIDATION_ERROR",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"Value error on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_code": "BAD_REQUEST",
                "status_code": status.HTTP_400_BAD_REQUEST,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "RAG service failed to process request due to an internal server error.",
                "error_code": "INTERNAL_SERVER_ERROR",
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        )

    # ------------------------------------------------------------------------
    # ROOT NAVIGATION & METADATA
    # ------------------------------------------------------------------------
    @app.get("/", tags=["General"])
    @app.get("/api", tags=["General"])
    def root_endpoint() -> Dict[str, Any]:
        """Root endpoint providing service metadata, routes, and documentation links."""
        return {
            "name": cfg.api_title,
            "version": cfg.api_version,
            "environment": cfg.app_env,
            "status": "online",
            "documentation": "/docs",
            "endpoints": {
                "health": "GET /api/health (or /health)",
                "config": "GET /api/config (or /config)",
                "query": "POST /api/query (or /query)",
                "chat": "POST /api/chat",
                "documents_upload": "POST /api/documents (or /documents)",
                "documents_list": "GET /api/documents",
                "evaluate": "POST /api/evaluate"
            }
        }

    # Mount modular routers under both /api and root level for maximum flexibility
    app.include_router(health_router, prefix="/api")
    app.include_router(health_router)
    app.include_router(query_router, prefix="/api")
    app.include_router(query_router)
    app.include_router(documents_router, prefix="/api")
    app.include_router(documents_router)
    app.include_router(eval_router, prefix="/api")
    app.include_router(eval_router)

    # Mount enterprise database-backed routers
    app.include_router(guardrails_router)
    app.include_router(logs_router)
    app.include_router(chunks_router)
    app.include_router(conversations_router)
    app.include_router(dashboard_router)

    return app
