"""
main.py

Knovera RAG Backend Application Entrypoint.
Initializes the unified FastAPI application and provides the ASGI server entry point
for production deployments and local Next.js frontend development.

Run locally:
    python main.py
    # or
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import uvicorn
from src.config import get_config
from src.api.app import create_app

# Instantiate FastAPI application for ASGI servers
config = get_config()
app = create_app(config)


def run_server():
    """Runs the Knovera backend server via Uvicorn."""
    print("=" * 70)
    print("  KNOVERA RAG BACKEND SERVICE")
    print("=" * 70)
    print(f"  * Environment:  {config.app_env}")
    print(f"  * API Docs:     http://localhost:{config.api_port}/docs")
    print(f"  * Health Check: http://localhost:{config.api_port}/api/health")
    print(f"  * CORS Origins: {', '.join(config.cors_origins)}")
    print("=" * 70)
    
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=(config.app_env == "development")
    )


if __name__ == "__main__":
    run_server()
