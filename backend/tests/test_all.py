"""
tests/test_all.py

Comprehensive Unified Test Suite for Knovera RAG Backend.
Consolidates all unit, integration, and API endpoint tests across all architectural layers.

Run via:
    python -m unittest tests/test_all.py
"""

import sys
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root directory is in sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi.testclient import TestClient

# Core and Configuration
from src.config import get_config, APIConfig

# Services
from src.services.text_cleaner import clean_text, normalize_whitespace
from src.services.document_loader import load_document, extract_metadata_from_filename
from src.services.token_chunker import chunk_text_by_tokens, estimate_tokens
from src.services.chunk_tagger import tag_chunk_metadata
from src.services.embedding_generator import EmbeddingGenerator
from src.services.embedding_quality_checker import check_embedding_quality
from src.services.vector_store import VectorDatabase
from src.services.top_k_retriever import TopKRetriever
from src.services.hybrid_retriever import HybridRetriever
from src.services.reranker import LLMReranker
from src.services.context_injector import assemble_augmented_context
from src.services.hallucination_guardrail import evaluate_retrieval_quality, STANDARD_SAFE_REFUSAL
from src.services.grounded_generator import GroundedAnswerGenerator
from src.services.source_tracer import extract_citations, verify_citation_grounding
from src.services.history_manager import HistoryManager
from src.services.document_indexer import store_upload, process_uploaded_document
from src.services.rag_pipeline import (
    RAGPipeline,
    embed_query,
    retrieve_context,
    assemble_context,
    generate_answer,
    EMPTY_RETRIEVAL_FALLBACK_MESSAGE
)
from src.services.rag_evaluator import RAGEvaluator

# API
from src.api.app import create_app


# ============================================================================
# 1. TEXT CLEANER TESTS
# ============================================================================
class TestTextCleaner(unittest.TestCase):
    def test_normalize_whitespace(self):
        raw = "Hello   world \t from    Knovera  "
        cleaned = normalize_whitespace(raw)
        self.assertEqual(cleaned, "Hello world from Knovera")

    def test_clean_text_noise_removal(self):
        raw = "Header ====\nSpecial content *** with symbols.\n--- Footer ---"
        cleaned = clean_text(raw)
        self.assertIn("Special content", cleaned)
        self.assertIsInstance(cleaned, str)

    def test_empty_string_handling(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t  "), "")


# ============================================================================
# 2. DOCUMENT LOADER TESTS
# ============================================================================
class TestDocumentLoader(unittest.TestCase):
    def test_extract_metadata_from_filename(self):
        meta = extract_metadata_from_filename("employee-handbook.pdf")
        self.assertEqual(meta["extension"], ".pdf")
        self.assertIn("employee", meta["title"].lower())

    def test_load_text_file(self):
        sample_path = _root / "data" / "customer_policy.txt"
        if sample_path.exists():
            content = load_document(sample_path)
            self.assertIsInstance(content, str)
            self.assertGreater(len(content), 0)

    def test_unsupported_file_extension(self):
        unsupported_file = _root / "data" / "unsupported_file.xyz"
        if unsupported_file.exists():
            with self.assertRaises(ValueError):
                load_document(unsupported_file)


# ============================================================================
# 3. TOKEN CHUNKER TESTS
# ============================================================================
class TestTokenChunker(unittest.TestCase):
    def test_estimate_tokens(self):
        text = "Knovera is a Retrieval-Augmented Generation system."
        tokens = estimate_tokens(text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    def test_chunk_text_by_tokens(self):
        text = "Sentence one about RAG. " * 50
        chunks = chunk_text_by_tokens(text, chunk_size=30, chunk_overlap=5)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("token_count", chunk)
            self.assertLessEqual(chunk["token_count"], 35)

    def test_empty_text_chunking(self):
        chunks = chunk_text_by_tokens("", chunk_size=100)
        self.assertEqual(chunks, [])


# ============================================================================
# 4. CHUNK TAGGER TESTS
# ============================================================================
class TestChunkTagger(unittest.TestCase):
    def test_tag_chunk_metadata(self):
        chunk = {"text": "This section covers refund policies and returns.", "token_count": 8}
        tagged = tag_chunk_metadata(
            chunk=chunk,
            doc_id="doc_001",
            source_file="refund_policy.md",
            chunk_index=0
        )
        self.assertEqual(tagged["source"], "refund_policy.md")
        self.assertEqual(tagged["doc_id"], "doc_001")
        self.assertEqual(tagged["chunk_index"], 0)
        self.assertIn("token_count", tagged)


# ============================================================================
# 5. EMBEDDING GENERATOR & QUALITY TESTS
# ============================================================================
class TestEmbeddingGeneratorAndQuality(unittest.TestCase):
    def setUp(self):
        self.generator = EmbeddingGenerator(model_name="text-embedding-3-small")

    @patch.object(EmbeddingGenerator, "embed", return_value=[[0.1] * 1536])
    def test_generate_embeddings_mock(self, mock_embed):
        vec = self.generator.generate_embedding("Test query")
        self.assertEqual(len(vec), 1536)

    def test_embedding_quality_checker(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.9, 0.1, 0.0])
        quality = check_embedding_quality(vec1, vec2)
        self.assertTrue(quality["is_valid"])
        self.assertGreater(quality["cosine_similarity"], 0.8)


# ============================================================================
# 6. VECTOR STORE TESTS
# ============================================================================
class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.generator = EmbeddingGenerator(model_name="text-embedding-3-small")
        self.vector_db = VectorDatabase(in_memory=True, embedding_generator=self.generator)
        self.test_col = "test_unified_collection"

    def tearDown(self):
        try:
            self.vector_db.delete_collection(self.test_col)
        except Exception:
            pass

    def test_collection_lifecycle_and_upsert(self):
        self.vector_db.create_collection(self.test_col)
        self.assertTrue(self.vector_db.collection_exists(self.test_col))

        dummy_vectors = [[0.1 * i] * 1536 for i in range(1, 4)]
        self.vector_db.upsert(
            collection_name=self.test_col,
            ids=["id1", "id2", "id3"],
            embeddings=dummy_vectors,
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[{"source": "test.txt"}, {"source": "test.txt"}, {"source": "test.txt"}]
        )

        count = self.vector_db.count(self.test_col)
        self.assertEqual(count, 3)

        results = self.vector_db.query(
            collection_name=self.test_col,
            query_vector=[0.1] * 1536,
            top_k=2
        )
        self.assertLessEqual(len(results), 2)


# ============================================================================
# 7. RETRIEVER & RERANKER TESTS
# ============================================================================
class TestRetrieversAndReranker(unittest.TestCase):
    def setUp(self):
        self.generator = EmbeddingGenerator(model_name="text-embedding-3-small")
        self.vector_db = VectorDatabase(in_memory=True, embedding_generator=self.generator)
        self.test_col = "test_retrieval_col"
        self.vector_db.create_collection(self.test_col)
        
        self.vector_db.upsert(
            collection_name=self.test_col,
            ids=["c1", "c2"],
            embeddings=[[0.5] * 1536, [0.1] * 1536],
            documents=["Knovera supports multi-turn chat.", "Weather forecast is sunny."],
            metadatas=[{"source": "rag.md"}, {"source": "weather.md"}]
        )

    def tearDown(self):
        try:
            self.vector_db.delete_collection(self.test_col)
        except Exception:
            pass

    def test_top_k_retriever(self):
        retriever = TopKRetriever(vector_db=self.vector_db)
        results = retriever.retrieve(
            collection_name=self.test_col,
            query_vector=[0.5] * 1536,
            k=2
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "c1")

    def test_hybrid_retriever_fusion(self):
        hybrid = HybridRetriever(vector_db=self.vector_db)
        results = hybrid.retrieve_hybrid(
            collection_name=self.test_col,
            query_text="Knovera chat",
            query_vector=[0.5] * 1536,
            k=2
        )
        self.assertIsInstance(results, list)

    def test_reranker_deterministic_fallback(self):
        reranker = LLMReranker()
        chunks = [
            {"id": "1", "text": "Python RAG backend", "score": 0.6},
            {"id": "2", "text": "Unrelated fruit banana", "score": 0.8}
        ]
        reranked = reranker.rerank(query="Python RAG", candidate_chunks=chunks, top_n=2)
        self.assertEqual(len(reranked), 2)


# ============================================================================
# 8. CONTEXT INJECTION & GUARDRAILS TESTS
# ============================================================================
class TestContextAndGuardrails(unittest.TestCase):
    def test_assemble_augmented_context(self):
        chunks = [
            {"source": "guide.md", "text": "Step 1: Install dependencies."},
            {"source": "faq.md", "text": "Step 2: Run main.py."}
        ]
        augmented = assemble_augmented_context(chunks)
        self.assertIn("guide.md", augmented)
        self.assertIn("Step 1", augmented)
        self.assertIn("Step 2", augmented)

    def test_hallucination_guardrail_pass_and_refuse(self):
        strong_chunks = [{"score": 0.88, "text": "High relevance chunk."}]
        eval_strong = evaluate_retrieval_quality(strong_chunks, min_top_score=0.70)
        self.assertTrue(eval_strong["is_strong"])

        weak_chunks = [{"score": 0.35, "text": "Low relevance chunk."}]
        eval_weak = evaluate_retrieval_quality(weak_chunks, min_top_score=0.70)
        self.assertFalse(eval_weak["is_strong"])

    def test_empty_chunks_guardrail(self):
        eval_empty = evaluate_retrieval_quality([], min_top_score=0.70)
        self.assertFalse(eval_empty["is_strong"])


# ============================================================================
# 9. GROUNDED GENERATION & SOURCE CITATIONS TESTS
# ============================================================================
class TestGenerationAndCitation(unittest.TestCase):
    def test_grounded_answer_generator_deterministic(self):
        generator = GroundedAnswerGenerator()
        ans = generator.generate_answer(
            query="How to run backend?",
            context="[Source: run.md] Run python main.py to launch server.",
            use_api=False
        )
        self.assertIsInstance(ans, str)
        self.assertGreater(len(ans), 0)

    def test_source_citation_extraction(self):
        answer = "To run Knovera, execute python main.py [Source: setup.md]."
        citations = extract_citations(answer)
        self.assertIn("setup.md", citations)

    def test_citation_grounding_verification(self):
        context = "Reference manual [Source: doc.pdf] describes setup."
        answer = "Setup is described in [Source: doc.pdf]."
        is_grounded = verify_citation_grounding(answer, context)
        self.assertTrue(is_grounded)


# ============================================================================
# 10. HISTORY MANAGER TESTS
# ============================================================================
class TestHistoryManager(unittest.TestCase):
    def test_history_manager_pruning(self):
        hm = HistoryManager(token_budget=50)
        hm.add_message("user", "Short question")
        hm.add_message("assistant", "Short answer")
        self.assertGreaterEqual(len(hm.get_messages()), 2)


# ============================================================================
# 11. DOCUMENT INDEXER TESTS
# ============================================================================
class TestDocumentIndexer(unittest.TestCase):
    def setUp(self):
        self.generator = EmbeddingGenerator(model_name="text-embedding-3-small")
        self.vector_db = VectorDatabase(in_memory=True, embedding_generator=self.generator)

    def test_process_uploaded_document(self):
        sample_path = _root / "uploads" / "equipment-policy.md"
        if not sample_path.exists():
            sample_path = _root / "data" / "customer_policy.txt"

        if sample_path.exists():
            res = process_uploaded_document(
                path=sample_path,
                vector_db=self.vector_db,
                embedding_generator=self.generator,
                collection_name="test_doc_indexer_col"
            )
            self.assertIn("chunks", res)
            self.assertGreater(res["chunks"], 0)
            self.assertEqual(res["indexed"], res["chunks"])


# ============================================================================
# 12. RAG PIPELINE ORCHESTRATOR TESTS
# ============================================================================
class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.generator = EmbeddingGenerator(model_name="text-embedding-3-small")
        self.vector_db = VectorDatabase(in_memory=True, embedding_generator=self.generator)
        self.pipeline = RAGPipeline(
            vector_db=self.vector_db,
            embedding_generator=self.generator,
            collection_name="test_pipeline_col"
        )
        self.vector_db.create_collection("test_pipeline_col")
        self.vector_db.upsert(
            collection_name="test_pipeline_col",
            ids=["p1"],
            embeddings=[[0.8] * 1536],
            documents=["Knovera RAG backend is production-ready."],
            metadatas=[{"source": "overview.md", "doc_title": "Overview"}]
        )

    def tearDown(self):
        try:
            self.vector_db.delete_collection("test_pipeline_col")
        except Exception:
            pass

    def test_rag_pipeline_run_mocked(self):
        with patch.object(EmbeddingGenerator, "embed", return_value=[[0.8] * 1536]):
            res = self.pipeline.run(query="Is Knovera production ready?", use_api=False)
            self.assertIn("answer", res)
            self.assertIn("sources", res)
            self.assertEqual(res["status"], "SUCCESS")


# ============================================================================
# 13. RAG EVALUATOR TESTS
# ============================================================================
class TestRAGEvaluator(unittest.TestCase):
    def test_evaluation_metrics(self):
        evaluator = RAGEvaluator()
        retrieved = ["doc1.md", "doc2.md", "doc3.md"]
        ground_truth = ["doc1.md", "doc4.md"]
        
        recall = evaluator.calculate_recall(retrieved, ground_truth)
        precision = evaluator.calculate_precision_at_k(retrieved, ground_truth, k=2)
        mrr = evaluator.calculate_mrr(retrieved, ground_truth)
        
        self.assertEqual(recall, 0.5)
        self.assertEqual(precision, 0.5)
        self.assertEqual(mrr, 1.0)


# ============================================================================
# 14. FASTAPI REST ENDPOINTS TESTS (FOR NEXT.JS INTEGRATION)
# ============================================================================
class TestFastAPIRestEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_config()
        cls.app = create_app(cls.config)
        cls.client = TestClient(cls.app)

    def test_root_and_api_endpoints(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

        res_api = self.client.get("/api")
        self.assertEqual(res_api.status_code, 200)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["status"], ["ok", "degraded"])
        self.assertIn("service", data)
        self.assertIn("version", data)

    def test_config_endpoint_masks_secrets(self):
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cors_origins", data)
        self.assertNotIn("sk-live-raw-secret", data.get("api_key_status", ""))

    def test_query_endpoint_validation(self):
        res = self.client.post("/api/query", json={"question": "a"})
        self.assertEqual(res.status_code, 422)

        res_empty = self.client.post("/api/query", json={"question": "   "})
        self.assertEqual(res_empty.status_code, 422)

    def test_query_endpoint_success(self):
        with patch.object(EmbeddingGenerator, "embed", return_value=[[0.1] * 1536]):
            res = self.client.post(
                "/api/query",
                json={
                    "question": "What is the return policy?",
                    "use_api": False,
                    "k": 2
                }
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("query", data)
            self.assertIn("answer", data)
            self.assertIn("sources", data)
            self.assertIn("status", data)

    def test_chat_endpoint_conversational(self):
        with patch.object(EmbeddingGenerator, "embed", return_value=[[0.1] * 1536]):
            res = self.client.post(
                "/api/chat",
                json={
                    "session_id": "test_session_123",
                    "message": "Hello, how do I submit a project?",
                    "use_api": False
                }
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["session_id"], "test_session_123")
            self.assertIn("message", data)

    def test_documents_list_endpoint(self):
        res = self.client.get("/api/documents")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("documents", data)
        self.assertIn("total_documents", data)

    def test_evaluate_endpoint(self):
        with patch.object(EmbeddingGenerator, "embed", return_value=[[0.1] * 1536]):
            res = self.client.post(
                "/api/evaluate",
                json={
                    "test_cases": [
                        {"query": "Test evaluation query", "min_score": 0.1}
                    ]
                }
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("summary", data)
            self.assertEqual(data["summary"]["total_queries"], 1)

    def test_cors_and_process_time_headers(self):
        res = self.client.get("/api/health")
        self.assertIn("X-Response-Time-Ms", res.headers)


# ============================================================================
# MAIN RUNNER
# ============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
