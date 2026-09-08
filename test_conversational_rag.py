"""
test_conversational_rag.py

Comprehensive Test Suite for Assignment 3.42: Conversational RAG & Follow-Up Context.

Validates:
1. Task 1: History tracking across multi-turn interactions.
2. Task 2: Follow-up query rewriting (pronoun resolution, reference expansion).
3. Task 3: Context retrieval precision using rewritten standalone queries vs. raw queries.
4. Task 4: End-to-end multi-turn conversational RAG flow.
5. Task 5: Token budget management and rolling history trimming.
"""

import os
import sys
import unittest
from typing import List, Dict, Any

# Ensure parent root is in sys.path
_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.history_manager import HistoryManager, rewrite_followup
from src.rag_pipeline import RAGPipeline, conversational_answer, embed_query, retrieve_context


class TestConversationalRAG(unittest.TestCase):
    """Unit and Integration test cases for Conversational RAG & Follow-Up Context."""

    @classmethod
    def setUpClass(cls):
        """Initializes in-memory vector store loaded with test corpus."""
        cls.generator = EmbeddingGenerator()
        cls.test_collection = "test_conversational_rag_kb"
        
        cls.test_chunks = [
            {
                "id": "submission_rubric.md:0",
                "text": "Project submission evidence requirement: Final project submissions require a GitHub pull request link, a sample output JSON file, and a 3-5 minute video explanation demonstrating working tests and pipeline execution.",
                "metadata": {
                    "source": "submission_rubric.md",
                    "section": "Submission Evidence",
                    "doc_title": "Project Submission Guidelines",
                    "category": "Academics"
                }
            },
            {
                "id": "video_policy.md:0",
                "text": "Video explanation requirements: The video explanation must be recorded as a screen-share walkthrough (3-5 minutes), uploaded to Google Drive with permission set to 'Anyone with the link can view', and verified in an incognito private tab before submission.",
                "metadata": {
                    "source": "video_policy.md",
                    "section": "Video Standards",
                    "doc_title": "Video Recording Guidelines",
                    "category": "Media"
                }
            },
            {
                "id": "sprint_schedule.md:0",
                "text": "Sprint 2 deliverables and scope: Sprint 2 covers Conversational RAG, multi-turn history tracking, follow-up query rewriting, and source attribution. All evidence guidelines including PR link and video explanation fully apply to Sprint 2.",
                "metadata": {
                    "source": "sprint_schedule.md",
                    "section": "Sprint 2 Policy",
                    "doc_title": "Sprint Roadmap & Scope",
                    "category": "Curriculum"
                }
            }
        ]

        embedded_records = cls.generator.embed_chunks(cls.test_chunks)
        cls.vector_db = VectorDatabase(in_memory=True, embedding_generator=cls.generator)
        indexer = CorpusIndexer(vector_db=cls.vector_db, default_collection=cls.test_collection)
        indexer.index_corpus(embedded_records, collection_name=cls.test_collection)

    # ------------------------------------------------------------------------
    # TASK 1 TESTS: CONVERSATION HISTORY TRACKING
    # ------------------------------------------------------------------------
    def test_task1_track_dialogue_history(self):
        """Validates that HistoryManager correctly records multi-turn user/assistant exchanges."""
        history = HistoryManager(system_message="System prompt", token_budget=1000)
        history.add_user_message("What evidence is required for project submission?")
        history.add_assistant_message("The submission requires a PR link and video walkthrough.")
        
        status = history.get_status()
        self.assertEqual(status["message_count"], 3)  # System + 1 turn (user + assistant)
        self.assertEqual(status["turn_count"], 1)
        self.assertTrue(status["token_count"] > 0)

    def test_task1_formatted_history_extraction(self):
        """Validates get_formatted_history formats dialogue turns cleanly for query rewriting."""
        history = HistoryManager(system_message="System prompt", token_budget=1000)
        history.add_user_message("What evidence is required for project submission?")
        history.add_assistant_message("The submission requires a PR link and video walkthrough.")
        
        formatted = history.get_formatted_history()
        self.assertIn('user: "What evidence is required for project submission?"', formatted)
        self.assertIn('assistant: "The submission requires a PR link and video walkthrough."', formatted)

    # ------------------------------------------------------------------------
    # TASK 2 TESTS: FOLLOW-UP QUERY REWRITING
    # ------------------------------------------------------------------------
    def test_task2_rewrite_followup_video(self):
        """Validates rewriting ambiguous follow-up 'What about the video?' into standalone query."""
        history = HistoryManager(system_message="System prompt", token_budget=1000)
        history.add_user_message("What evidence is required for project submission?")
        history.add_assistant_message("The submission requires a PR link and video walkthrough.")
        
        rewritten = rewrite_followup(history, "What about the video?", use_api=False)
        self.assertIsInstance(rewritten, str)
        self.assertIn("video", rewritten.lower())
        self.assertIn("submission", rewritten.lower())
        self.assertNotIn("What about the video?", rewritten)

    def test_task2_rewrite_followup_sprint_scope(self):
        """Validates rewriting follow-up 'Does it apply to Sprint 2?' into standalone query."""
        history = HistoryManager(system_message="System prompt", token_budget=1000)
        history.add_user_message("What evidence is required for project submission?")
        history.add_assistant_message("The submission requires a PR link and video walkthrough.")
        
        rewritten = rewrite_followup(history, "Does it apply to Sprint 2?", use_api=False)
        self.assertIsInstance(rewritten, str)
        self.assertIn("sprint 2", rewritten.lower())
        self.assertTrue("apply" in rewritten.lower() or "submission" in rewritten.lower())

    def test_task2_standalone_question_unchanged(self):
        """Validates that an explicit standalone query without ambiguous pronouns remains clean."""
        history = HistoryManager(system_message="System prompt", token_budget=1000)
        history.add_user_message("What evidence is required for project submission?")
        history.add_assistant_message("The submission requires a PR link and video walkthrough.")
        
        explicit_q = "What is the passing score threshold?"
        rewritten = rewrite_followup(history, explicit_q, use_api=False)
        self.assertIn("passing score", rewritten.lower())

    # ------------------------------------------------------------------------
    # TASK 3 TESTS: RETRIEVAL WITH REWRITTEN QUERY
    # ------------------------------------------------------------------------
    def test_task3_retrieval_with_rewritten_query(self):
        """Validates retrieval using rewritten query successfully fetches video policy chunk."""
        history = HistoryManager(system_message="System prompt", token_budget=1000)
        history.add_user_message("What evidence is required for project submission?")
        history.add_assistant_message("The submission requires a PR link and video walkthrough.")
        
        standalone_q = rewrite_followup(history, "What about the video?", use_api=False)
        q_vec = embed_query(standalone_q, generator=self.generator)
        
        chunks = retrieve_context(
            query_vector=q_vec,
            vector_db=self.vector_db,
            collection_name=self.test_collection,
            k=2
        )
        
        self.assertGreaterEqual(len(chunks), 1)
        # Verify retrieved chunks contain video explanation content
        retrieved_texts = [c["text"].lower() for c in chunks]
        self.assertTrue(any("video" in t for t in retrieved_texts))

    # ------------------------------------------------------------------------
    # TASK 4 TESTS: DEMONSTRATE MULTI-TURN CONVERSATIONAL RAG
    # ------------------------------------------------------------------------
    def test_task4_multi_turn_conversational_flow(self):
        """Validates a complete 3-turn conversational RAG dialogue."""
        pipeline = RAGPipeline(
            vector_db=self.vector_db,
            generator=self.generator,
            default_collection=self.test_collection,
            default_k=2,
            use_api=False
        )
        history = HistoryManager(system_message="System prompt", token_budget=2000)
        
        # Turn 1
        res1 = pipeline.conversational_query("What evidence is required for project submission?", history)
        self.assertEqual(res1["status"], "SUCCESS")
        self.assertIn("submission_rubric.md", res1["sources"][0]["source"])
        
        # Turn 2
        res2 = pipeline.conversational_query("What about the video?", history)
        self.assertEqual(res2["status"], "SUCCESS")
        self.assertIn("video", res2["rewritten_query"].lower())
        self.assertTrue(len(res2["sources"]) > 0)
        
        # Turn 3
        res3 = pipeline.conversational_query("Does it apply to Sprint 2?", history)
        self.assertEqual(res3["status"], "SUCCESS")
        self.assertIn("sprint 2", res3["rewritten_query"].lower())
        
        # Verify history accumulated turns
        self.assertEqual(history.turn_count, 3)

    # ------------------------------------------------------------------------
    # TASK 5 TESTS: TOKEN BUDGETING & TRIMMING
    # ------------------------------------------------------------------------
    def test_task5_token_budget_trimming(self):
        """Validates that history manager trims oldest turns when exceeding budget."""
        history = HistoryManager(system_message="System prompt", token_budget=300)
        
        # Add multiple turns until budget forces trimming
        for i in range(5):
            history.add_user_message(f"Turn {i}: What are the specific detailed guidelines for project submission evidence?")
            history.add_assistant_message(f"Turn {i}: The project submission evidence guidelines require GitHub PR links, video walkthroughs, and verified output files.")
            if history.should_trim():
                history.trim_to_budget()
                
        status = history.get_status()
        self.assertTrue(status["within_budget"] or status["message_count"] <= 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
