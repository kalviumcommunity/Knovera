"""
test_rag_pipeline.py

Comprehensive test suite for Assignment 3.37: RAG Pipeline Architecture & Flow Design.
Validates:
1. Stage 1: embed_query (vector generation, dimension, validation)
2. Stage 2: retrieve_context (top-k semantic search, score thresholds, metadata filtering)
3. Stage 3: assemble_context (citation formatting, source indexing, multi-chunk concatenation)
4. Stage 4: generate_answer (grounded synthesis, empty context handling, prompt rendering)
5. Orchestration: answer_query & RAGPipeline (end-to-end execution, fallback on empty retrieval, latency tracking)
"""

import os
import unittest
from typing import List, Dict, Any

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.rag_pipeline import (
    embed_query,
    retrieve_context,
    assemble_context,
    generate_answer,
    answer_query,
    RAGPipeline,
    EMPTY_RETRIEVAL_FALLBACK_MESSAGE
)


class TestRAGPipelineArchitecture(unittest.TestCase):
    """Unit and Integration tests for RAG Pipeline Architecture & Flow."""

    @classmethod
    def setUpClass(cls):
        """Initializes an in-memory test vector database loaded with sample corpus chunks."""
        cls.generator = EmbeddingGenerator()
        cls.test_collection = "test_rag_flow_collection"
        
        # Sample realistic multi-domain corpus
        cls.sample_chunks = [
            {
                "id": "submission-rubric.md:0",
                "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough demonstrating all test executions and architecture flows.",
                "metadata": {
                    "source": "submission-rubric.md",
                    "chunk_index": 0,
                    "section": "Evidence Requirements",
                    "doc_title": "Project Submission Rubric",
                    "category": "Academics"
                }
            },
            {
                "id": "account-guide.md:0",
                "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and verifying their registered email address.",
                "metadata": {
                    "source": "account-guide.md",
                    "chunk_index": 0,
                    "section": "Authentication",
                    "doc_title": "Account & Security Guide",
                    "category": "Authentication"
                }
            },
            {
                "id": "campus-guide.md:0",
                "text": "Campus cafeteria hours and lunch menu: The cafeteria operates from 11:30 AM to 2:30 PM offering hot artisan wraps, fresh garden salads, and rotating daily chef specials.",
                "metadata": {
                    "source": "campus-guide.md",
                    "chunk_index": 0,
                    "section": "Dining Facilities",
                    "doc_title": "Campus Amenities Guide",
                    "category": "Campus Life"
                }
            },
            {
                "id": "grading-policy.md:0",
                "text": "Assignment grading criteria: Learners receive scores based on test suite completeness (40%), modular code separation (30%), and documentation depth (30%). Minimum pass is 60%.",
                "metadata": {
                    "source": "grading-policy.md",
                    "chunk_index": 0,
                    "section": "Evaluation Rubrics",
                    "doc_title": "Academic Grading Policy",
                    "category": "Academics"
                }
            }
        ]

        # Embed and index into in-memory ChromaDB
        embedded_records = cls.generator.embed_chunks(cls.sample_chunks)
        cls.vector_db = VectorDatabase(in_memory=True, embedding_generator=cls.generator)
        indexer = CorpusIndexer(vector_db=cls.vector_db, default_collection=cls.test_collection)
        indexer.index_corpus(embedded_records, collection_name=cls.test_collection)

    # ------------------------------------------------------------------------
    # STAGE 1 TESTS: EMBED QUERY
    # ------------------------------------------------------------------------
    def test_stage1_embed_query_valid(self):
        """Validates that embed_query returns a non-empty vector of expected dimension."""
        query = "What evidence is required for project submission?"
        vector = embed_query(query, generator=self.generator)
        
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), self.generator.dimension)
        self.assertTrue(all(isinstance(x, (float, int)) for x in vector))

    def test_stage1_embed_query_empty_error(self):
        """Validates that embed_query raises ValueError when given an empty query."""
        with self.assertRaises(ValueError):
            embed_query("", generator=self.generator)
        with self.assertRaises(ValueError):
            embed_query("   ", generator=self.generator)

    # ------------------------------------------------------------------------
    # STAGE 2 TESTS: RETRIEVE CONTEXT
    # ------------------------------------------------------------------------
    def test_stage2_retrieve_context_top_k(self):
        """Validates top-k retrieval fetches relevant chunks with scores and metadata."""
        query = "What evidence is required for project submission?"
        vector = embed_query(query, generator=self.generator)
        
        results = retrieve_context(
            query_vector=vector,
            vector_db=self.vector_db,
            collection_name=self.test_collection,
            k=2
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertIn("submission-rubric.md", results[0]["source"])
        self.assertIn("video walkthrough", results[0]["text"])
        self.assertIn("metadata", results[0])
        self.assertGreaterEqual(results[0]["score"], 0.0)

    def test_stage2_retrieve_context_score_threshold_filter(self):
        """Validates score threshold filters out low-scoring chunks."""
        query = "What evidence is required for project submission?"
        vector = embed_query(query, generator=self.generator)
        
        # High threshold that allows top match
        results = retrieve_context(
            query_vector=vector,
            vector_db=self.vector_db,
            collection_name=self.test_collection,
            k=4,
            score_threshold=0.85
        )
        for r in results:
            self.assertGreaterEqual(r["score"], 0.85)

    def test_stage2_retrieve_context_invalid_k(self):
        """Validates error raised on non-positive k value."""
        vector = [0.1] * self.generator.dimension
        with self.assertRaises(ValueError):
            retrieve_context(vector, vector_db=self.vector_db, collection_name=self.test_collection, k=0)

    # ------------------------------------------------------------------------
    # STAGE 3 TESTS: ASSEMBLE CONTEXT
    # ------------------------------------------------------------------------
    def test_stage3_assemble_context_formatting(self):
        """Validates context assembly structures chunks with indexed headers and source citations."""
        mock_chunks = [
            {
                "source": "submission-rubric.md",
                "text": "Upload repository GitHub links and video walkthrough.",
                "metadata": {"section": "Evidence"}
            },
            {
                "source": "grading-policy.md",
                "text": "Learners receive scores based on test completeness.",
                "metadata": {"section": "Grading"}
            }
        ]
        
        context = assemble_context(mock_chunks)
        self.assertIn("[1] Source: submission-rubric.md | Section: Evidence", context)
        self.assertIn("Upload repository GitHub links and video walkthrough.", context)
        self.assertIn("[2] Source: grading-policy.md | Section: Grading", context)
        self.assertIn("\n\n", context)

    def test_stage3_assemble_context_empty(self):
        """Validates assemble_context returns empty string when given empty chunk list."""
        self.assertEqual(assemble_context([]), "")

    # ------------------------------------------------------------------------
    # STAGE 4 TESTS: GENERATE ANSWER
    # ------------------------------------------------------------------------
    def test_stage4_generate_answer_grounded(self):
        """Validates generate_answer generates factual answer containing context evidence."""
        query = "What evidence is required for project submission?"
        context = "[1] Source: submission-rubric.md\nProject submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough."
        
        answer = generate_answer(query=query, context=context, use_api=False)
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 20)
        self.assertTrue("github" in answer.lower() or "video" in answer.lower() or "walkthrough" in answer.lower())

    def test_stage4_generate_answer_empty_context_fallback(self):
        """Validates generate_answer returns fallback when context is empty."""
        answer = generate_answer("What is the refund policy?", "", use_api=False)
        self.assertEqual(answer, EMPTY_RETRIEVAL_FALLBACK_MESSAGE)

    # ------------------------------------------------------------------------
    # STAGE 5 TESTS: END-TO-END ORCHESTRATOR & FALLBACK
    # ------------------------------------------------------------------------
    def test_stage5_answer_query_end_to_end_success(self):
        """Validates complete query-to-answer pipeline run on sample query."""
        query = "What evidence is required for project submission?"
        result = answer_query(
            query=query,
            k=2,
            collection_name=self.test_collection,
            vector_db=self.vector_db,
            generator=self.generator,
            use_api=False
        )
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["query"], query)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 10)
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["source"], "submission-rubric.md")
        self.assertIn("stage_latencies_ms", result)
        self.assertIn("embed_ms", result["stage_latencies_ms"])
        self.assertIn("retrieve_ms", result["stage_latencies_ms"])
        self.assertIn("assemble_ms", result["stage_latencies_ms"])
        self.assertIn("generate_ms", result["stage_latencies_ms"])

    def test_stage5_answer_query_empty_retrieval_fallback(self):
        """Validates pipeline handles empty retrieval state without hallucinating."""
        query = "How do I build a fusion reactor on Mars?"
        result = answer_query(
            query=query,
            k=2,
            score_threshold=0.99999,
            collection_name=self.test_collection,
            vector_db=self.vector_db,
            generator=self.generator,
            use_api=False
        )
        
        self.assertEqual(result["status"], "NO_CONTEXT_FOUND")
        self.assertEqual(result["answer"], EMPTY_RETRIEVAL_FALLBACK_MESSAGE)
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["context"], "")
        self.assertEqual(result["num_retrieved"], 0)

    def test_stage5_rag_pipeline_class_interface(self):
        """Validates object-oriented RAGPipeline wrapper executes seamlessly."""
        pipeline = RAGPipeline(
            vector_db=self.vector_db,
            generator=self.generator,
            default_collection=self.test_collection,
            default_k=2,
            use_api=False
        )
        
        # Test stage methods
        vec = pipeline.embed("When does cafeteria serve lunch?")
        chunks = pipeline.retrieve(vec, k=1)
        self.assertEqual(len(chunks), 1)
        self.assertIn("campus-guide.md", chunks[0]["source"])
        
        ctx = pipeline.assemble(chunks)
        self.assertIn("campus-guide.md", ctx)
        
        ans = pipeline.generate("When does cafeteria serve lunch?", ctx)
        self.assertTrue(len(ans) > 10)
        
        # Test complete query method
        result = pipeline.query("When does cafeteria serve lunch?")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("campus-guide.md", result["sources"][0]["source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
