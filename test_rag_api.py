"""
test_rag_api.py

Unit and Integration Test Suite for Knovera RAG Backend API Service (Assignment 3.44).
Validates:
1. Pydantic request and response models & input validation rules (length, empty, types).
2. Endpoint routing: /query, /health, /config, and root /.
3. Structured JSON response contract: answer, sources (source, chunk_id, score), status, latency.
4. Hallucination guardrail refusal state propagation through API responses.
5. Error handling and HTTP status codes (HTTP 400, HTTP 422, HTTP 500).
6. Dynamic environment configuration loading and secrets masking.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api_config import APIConfig, get_config
from src.rag_api import create_app, RAGService, QueryRequest, QueryResponse, Source


class TestRAGBackendAPI(unittest.TestCase):
    """Test suite for Knovera FastAPI RAG backend service."""

    def setUp(self):
        # Create a mock or test-configured RAG service
        self.mock_rag_service = MagicMock(spec=RAGService)
        self.mock_rag_service.config = get_config(force_reload=True)
        self.mock_rag_service.vector_db = MagicMock()
        self.mock_rag_service.vector_db.is_reachable.return_value = True
        self.mock_rag_service.vector_db.count.return_value = 12

        # Initialize test app and client
        self.app = create_app(service=self.mock_rag_service)
        self.client = TestClient(self.app)

    # ------------------------------------------------------------------------
    # 1. ROOT & HEALTH ENDPOINTS
    # ------------------------------------------------------------------------

    def test_root_endpoint(self):
        """Root endpoint should return 200 and list available endpoints."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("name", data)
        self.assertIn("version", data)
        self.assertEqual(data["status"], "online")
        self.assertIn("query", data["endpoints"])

    def test_health_endpoint_healthy(self):
        """Health endpoint should verify vector DB and return 200 with status 'ok'."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["vector_db_available"])
        self.assertEqual(data["service"], "knovera-rag-backend-api")
        self.assertIn("embedding_model", data)
        self.assertIn("chat_model", data)

    def test_health_endpoint_degraded(self):
        """Health endpoint should report degraded status if vector store fails."""
        self.mock_rag_service.vector_db.is_reachable.side_effect = Exception("DB Connection Lost")
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["vector_db_available"])

    def test_config_endpoint_masks_secrets(self):
        """Config endpoint should expose runtime settings while masking API keys."""
        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("embedding_model", data)
        self.assertIn("vector_db_url", data)
        self.assertIn("api_key_status", data)
        # Ensure raw secrets are not exposed
        self.assertNotIn("sk-proj-", data["api_key_status"])

    # ------------------------------------------------------------------------
    # 2. QUERY ENDPOINT: VALIDATION & ERROR HANDLING
    # ------------------------------------------------------------------------

    def test_query_validation_too_short(self):
        """Questions under 3 characters should be rejected with HTTP 422."""
        payload = {"question": "Hi"}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")
        self.assertIn("at least 3", data["detail"])

    def test_query_validation_whitespace_only(self):
        """Whitespace-only questions should be rejected with HTTP 422."""
        payload = {"question": "     "}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")

    def test_query_validation_too_long(self):
        """Questions exceeding 1000 characters should be rejected with HTTP 422."""
        long_question = "What is the policy? " * 60  # > 1000 chars
        self.assertGreater(len(long_question), 1000)
        payload = {"question": long_question}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error_code"], "VALIDATION_ERROR")

    def test_query_validation_missing_question(self):
        """Empty request body should trigger HTTP 422 validation error."""
        response = self.client.post("/query", json={})
        self.assertEqual(response.status_code, 422)

    def test_query_validation_invalid_k_parameter(self):
        """Invalid k value (< 1 or > 20) should trigger HTTP 422."""
        response = self.client.post("/query", json={"question": "What is Knovera?", "k": 0})
        self.assertEqual(response.status_code, 422)

        response = self.client.post("/query", json={"question": "What is Knovera?", "k": 50})
        self.assertEqual(response.status_code, 422)

    def test_query_value_error_handling(self):
        """ValueError raised by service should return HTTP 400 Bad Request."""
        self.mock_rag_service.execute_query.side_effect = ValueError("Invalid search filter format")
        response = self.client.post("/query", json={"question": "What is Knovera?"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Invalid search filter format", data["detail"])

    def test_query_unhandled_server_error(self):
        """Unhandled internal exceptions should return HTTP 500."""
        self.mock_rag_service.execute_query.side_effect = RuntimeError("Chroma index corrupted")
        response = self.client.post("/query", json={"question": "What is Knovera?"})
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("RAG service failed", data["detail"])

    # ------------------------------------------------------------------------
    # 3. QUERY ENDPOINT: STRUCTURED JSON RESPONSE & SOURCES
    # ------------------------------------------------------------------------

    def test_query_successful_answer_with_sources(self):
        """Valid query should return 200, grounded answer, sources list, and status 'answered'."""
        self.mock_rag_service.execute_query.return_value = {
            "query": "What evidence is required for project submission?",
            "answer": "The submission requires a PR link, sample output, and a video explanation. [1]",
            "sources": [
                {
                    "source": "submission-rubric.md",
                    "chunk_id": "submission-rubric.md:2",
                    "score": 0.8425,
                    "rank": 1,
                    "section": "Submission Requirements",
                    "doc_title": "Project Submission Rubric"
                }
            ],
            "status": "answered",
            "latency_ms": 150.5,
            "stage_latencies_ms": {"embed_ms": 50.0, "retrieve_ms": 20.0, "assemble_ms": 0.5, "generate_ms": 80.0}
        }

        payload = {
            "question": "What evidence is required for project submission?",
            "k": 3,
            "score_threshold": 0.60
        }
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query"], payload["question"])
        self.assertIn("PR link", data["answer"])
        self.assertEqual(data["status"], "answered")
        self.assertEqual(len(data["sources"]), 1)
        
        # Verify source shape
        source = data["sources"][0]
        self.assertEqual(source["source"], "submission-rubric.md")
        self.assertEqual(source["chunk_id"], "submission-rubric.md:2")
        self.assertAlmostEqual(source["score"], 0.8425, places=4)
        self.assertEqual(source["rank"], 1)
        self.assertEqual(source["section"], "Submission Requirements")
        self.assertIn("timestamp", data)
        self.assertIn("X-Response-Time-Ms", response.headers)

    def test_query_refusal_when_no_context_found(self):
        """Query with unindexed or out-of-domain topic returns refusal status and appropriate message."""
        self.mock_rag_service.execute_query.return_value = {
            "query": "What is the secret recipe for Martian space pie?",
            "answer": "I could not find relevant context for that question.",
            "sources": [],
            "status": "refused_empty_context",
            "latency_ms": 45.2,
            "stage_latencies_ms": {"embed_ms": 30.0, "retrieve_ms": 15.0}
        }

        response = self.client.post("/query", json={"question": "What is the secret recipe for Martian space pie?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "refused_empty_context")
        self.assertEqual(len(data["sources"]), 0)
        self.assertIn("I could not find relevant context", data["answer"])

    # ------------------------------------------------------------------------
    # 4. ENVIRONMENT CONFIGURATION LOADING
    # ------------------------------------------------------------------------

    def test_api_config_environment_loading(self):
        """APIConfig should dynamically load environment variables."""
        with patch.dict(os.environ, {
            "EMBEDDING_MODEL": "custom-embed-model-v2",
            "CHAT_MODEL": "gpt-4o-custom",
            "COLLECTION_NAME": "test_eval_collection",
            "API_PORT": "9090",
            "TOP_K": "8",
            "MIN_TOP_SCORE": "0.80",
            "USE_LIVE_API": "true"
        }):
            config = APIConfig()
            self.assertEqual(config.embedding_model, "custom-embed-model-v2")
            self.assertEqual(config.chat_model, "gpt-4o-custom")
            self.assertEqual(config.collection_name, "test_eval_collection")
            self.assertEqual(config.api_port, 9090)
            self.assertEqual(config.top_k, 8)
            self.assertEqual(config.min_top_score, 0.80)
            self.assertTrue(config.use_live_api)


if __name__ == "__main__":
    unittest.main()
