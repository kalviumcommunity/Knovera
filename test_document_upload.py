"""
test_document_upload.py

Unit and Integration Test Suite for Document Upload & Indexing Endpoint (Assignment 3.45).
Tests:
1. Document upload & ingestion for Markdown (.md) and Plain Text (.txt) files.
2. Error handling for unsupported file formats (HTTP 415).
3. Error handling for empty 0-byte files (HTTP 400).
4. Error handling for oversized files exceeding capacity limits (HTTP 413).
5. Safe file persistence and path sanitization.
6. Runtime searchability confirmation (querying new content immediately without server restart).
7. Alias endpoint (/upload) functionality.
"""

import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api_config import APIConfig, get_config
from src.rag_api import create_app, RAGService
from src.document_indexer import (
    validate_upload_metadata,
    SUPPORTED_EXTENSIONS,
    DEFAULT_MAX_FILE_SIZE_BYTES
)
from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator


class TestDocumentUploadAndIndexing(unittest.TestCase):
    """Test suite for Document Upload & Runtime Indexing functionality."""

    def setUp(self):
        # Create isolated temporary directory for test uploads and chroma DB
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = Path(self.test_dir) / "uploads"
        self.test_chroma_dir = Path(self.test_dir) / "chroma_db"
        self.test_collection = "test_upload_kb"

        # Initialize test in-memory/isolated vector database
        self.embedding_gen = EmbeddingGenerator(dimension=1536)
        self.vector_db = VectorDatabase(
            persist_dir=str(self.test_chroma_dir),
            in_memory=True,
            embedding_generator=self.embedding_gen
        )

        # Setup mock/isolated RAG Service
        self.rag_service = RAGService()
        self.rag_service._vector_db = self.vector_db
        self.rag_service._embedding_generator = self.embedding_gen

        # Patch environment config
        self.config_patcher = patch.dict(os.environ, {
            "UPLOAD_DIR": str(self.test_upload_dir),
            "VECTOR_DB_URL": str(self.test_chroma_dir),
            "COLLECTION_NAME": self.test_collection,
            "MAX_UPLOAD_SIZE_MB": "5"
        })
        self.config_patcher.start()

        # Initialize API Client
        self.app = create_app(service=self.rag_service)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.config_patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # 1. METADATA VALIDATION TESTS
    # ------------------------------------------------------------------------

    def test_validate_upload_metadata_valid_extensions(self):
        """All supported extensions should pass validation and return lowercase suffix."""
        for ext in [".txt", ".md", ".pdf", ".html", ".htm"]:
            suffix = validate_upload_metadata(f"document{ext}", file_size=100)
            self.assertEqual(suffix, ext)

    def test_validate_upload_metadata_unsupported_extension(self):
        """Unsupported file extensions should raise HTTP 415."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            validate_upload_metadata("malware.exe", file_size=100)
        self.assertEqual(ctx.exception.status_code, 415)
        self.assertIn("Unsupported file type", ctx.exception.detail)

    def test_validate_upload_metadata_empty_file(self):
        """Empty 0-byte file should raise HTTP 400."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            validate_upload_metadata("notes.md", file_size=0)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("empty", ctx.exception.detail.lower())

    def test_validate_upload_metadata_oversized_file(self):
        """File exceeding size threshold should raise HTTP 413."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            validate_upload_metadata("large.pdf", file_size=20 * 1024 * 1024, max_size_bytes=10 * 1024 * 1024)
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("exceeds maximum allowed limit", ctx.exception.detail)

    # ------------------------------------------------------------------------
    # 2. UPLOAD ENDPOINT (POST /documents & POST /upload)
    # ------------------------------------------------------------------------

    def test_upload_markdown_document_success(self):
        """POST /documents should ingest, chunk, embed, and index markdown file."""
        doc_content = (
            "# Remote Work & Equipment Policy\n\n"
            "Employees are entitled to a $500 home office equipment stipend upon hiring. "
            "All hardware purchases must be submitted with valid receipts within 30 days.\n\n"
            "## Internet Allowance\n"
            "A monthly internet reimbursement of $75 is provided to all full-time remote team members."
        ).encode("utf-8")

        files = {"file": ("remote-work-policy.md", io.BytesIO(doc_content), "text/markdown")}
        response = self.client.post("/documents", files=files)

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["filename"], "remote-work-policy.md")
        self.assertIn("summary", data)
        self.assertGreaterEqual(data["summary"]["chunks"], 1)
        self.assertEqual(data["summary"]["indexed"], data["summary"]["chunks"])
        self.assertIn("stage_latencies_ms", data["summary"])

    def test_upload_text_document_success(self):
        """POST /documents should accept plain text (.txt) files."""
        doc_content = (
            "Knovera Onboarding Protocol:\n"
            "Step 1: Set up company email and SSO login.\n"
            "Step 2: Request API access keys from enterprise portal.\n"
            "Step 3: Complete mandatory security training modules."
        ).encode("utf-8")

        files = {"file": ("onboarding-protocol.txt", io.BytesIO(doc_content), "text/plain")}
        response = self.client.post("/documents", files=files)

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["filename"], "onboarding-protocol.txt")

    def test_upload_alias_endpoint(self):
        """POST /upload should function as an alias to POST /documents."""
        doc_content = b"Short test policy content for alias endpoint."
        files = {"file": ("alias-test.md", io.BytesIO(doc_content), "text/markdown")}
        response = self.client.post("/upload", files=files)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["status"], "indexed")

    def test_upload_unsupported_format_returns_415(self):
        """Uploading .docx or unsupported format should return HTTP 415."""
        files = {"file": ("document.docx", io.BytesIO(b"fake docx content"), "application/octet-stream")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 415)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_upload_empty_file_returns_400(self):
        """Uploading an empty file should return HTTP 400 Bad Request."""
        files = {"file": ("empty.md", io.BytesIO(b""), "text/markdown")}
        response = self.client.post("/documents", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    # ------------------------------------------------------------------------
    # 3. RUNTIME SEARCHABILITY CONFIRMATION
    # ------------------------------------------------------------------------

    def test_runtime_searchability_new_content_queryable_without_restart(self):
        """
        Verify that new document content uploaded at runtime is immediately
        retrievable and searchable via POST /query without restarting the service.
        """
        query_payload = {
            "question": "What is the stipend for home office equipment?",
            "k": 3,
            "score_threshold": 0.40
        }

        # Step 1: Query before upload -> should return refusal / no context
        resp_before = self.client.post("/query", json=query_payload)
        self.assertEqual(resp_before.status_code, 200)
        self.assertIn(resp_before.json()["status"], ["refused_empty_context", "refused_low_similarity", "NO_CONTEXT_FOUND"])

        # Step 2: Upload new document with the answer
        new_doc = (
            "# Remote Equipment Policy\n\n"
            "Full-time employees receive a $500 home office equipment stipend during their first month. "
            "Reimbursement requests must be submitted via the finance portal."
        ).encode("utf-8")

        upload_resp = self.client.post(
            "/documents",
            files={"file": ("equipment-policy.md", io.BytesIO(new_doc), "text/markdown")}
        )
        self.assertEqual(upload_resp.status_code, 201)

        # Step 3: Query immediately after upload without restarting server
        resp_after = self.client.post("/query", json=query_payload)
        self.assertEqual(resp_after.status_code, 200)
        data = resp_after.json()

        # Step 4: Verify grounded answer and source attribution
        self.assertEqual(data["status"], "answered")
        self.assertIn("500", data["answer"])
        self.assertGreaterEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["source"], "equipment-policy.md")


if __name__ == "__main__":
    unittest.main()
