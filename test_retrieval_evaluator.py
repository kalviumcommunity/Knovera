"""
test_retrieval_evaluator.py

Comprehensive Pytest suite for Assignment 3.36: Retrieval Evaluation & Recall Testing.
Validates labelled query set handling, recall@k and precision@k metrics calculation,
multi-k comparison, failure inspection, and audit report generation.
"""

import os
import json
import pytest

from src.vector_store import VectorDatabase
from src.embedding_generator import EmbeddingGenerator
from src.corpus_indexer import CorpusIndexer
from src.retrieval_evaluator import (
    LabelledQuery,
    QueryEvaluationResult,
    AggregateMetrics,
    RetrievalEvaluator
)


@pytest.fixture
def sample_labelled_queries():
    """Provides standard benchmark labelled queries with ground-truth chunk IDs."""
    return [
        LabelledQuery(
            query="How can a learner reset their password?",
            relevant_chunk_ids={"account-guide.md:0", "account-guide.md:1"},
            category="Authentication"
        ),
        LabelledQuery(
            query="What evidence is required for project submission?",
            relevant_chunk_ids={"submission-rubric.md:0"},
            category="Academics"
        ),
        LabelledQuery(
            query="When does the cafeteria menu change?",
            relevant_chunk_ids={"campus-guide.md:0"},
            category="Campus Life"
        )
    ]


@pytest.fixture
def indexed_eval_vector_db():
    """Populates an in-memory ChromaDB vector store with clean test document chunks."""
    generator = EmbeddingGenerator()
    chunks = [
        {
            "id": "account-guide.md:0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "category": "Authentication"}
        },
        {
            "id": "account-guide.md:1",
            "text": "Password recovery email dispatch: Learners receive a single-use secure link to recover their registered account password.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "category": "Authentication"}
        },
        {
            "id": "submission-rubric.md:0",
            "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 0, "category": "Academics"}
        },
        {
            "id": "campus-guide.md:0",
            "text": "Campus cafeteria hours and daily lunch menu rotation: Operating from 11:30 AM to 2:30 PM offering fresh garden salads.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "category": "Campus Life"}
        },
        {
            "id": "distractor.md:0",
            "text": "Software engineering design patterns and clean architectural abstractions for Python development.",
            "metadata": {"source": "distractor.md", "chunk_index": 0, "category": "General"}
        }
    ]

    records = generator.embed_chunks(chunks)
    vdb = VectorDatabase(in_memory=True)
    indexer = CorpusIndexer(vector_db=vdb, default_collection="test_eval_collection")
    indexer.index_corpus(records, reset_collection=True)

    return vdb, "test_eval_collection"


def test_labelled_query_and_metrics_dataclasses():
    """Validates dataclass instantiation and dictionary serialization."""
    item = LabelledQuery(
        query="How to reset password?",
        relevant_chunk_ids=["account.md:0", "account.md:1"],
        category="Auth"
    )
    assert item.query == "How to reset password?"
    assert isinstance(item.relevant_chunk_ids, set)
    assert "account.md:0" in item.relevant_chunk_ids
    assert item.to_dict()["category"] == "Auth"

    result = QueryEvaluationResult(
        query="Test query",
        retrieved_ids=["chunk_1", "chunk_2"],
        relevant_chunk_ids=["chunk_1"],
        hits=["chunk_1"],
        recall=1.0,
        precision=0.5,
        f1_score=0.6667,
        reciprocal_rank=1.0,
        hit_at_k=True,
        scores=[0.9, 0.4],
        sources=["doc1", "doc2"]
    )
    assert result.recall == 1.0
    assert result.precision == 0.5
    assert result.to_dict()["hit_at_k"] is True


def test_single_query_recall_precision_evaluation(indexed_eval_vector_db):
    """Tests evaluation of a single query and verifies recall and precision calculations."""
    vdb, collection_name = indexed_eval_vector_db
    evaluator = RetrievalEvaluator(vector_db=vdb, default_collection=collection_name)

    query_item = {
        "query": "How can a learner reset their password?",
        "relevant_chunk_ids": {"account-guide.md:0", "account-guide.md:1"},
        "category": "Authentication"
    }

    result = evaluator.evaluate_query(query_item=query_item, k=5, collection_name=collection_name)

    assert result.query == query_item["query"]
    assert len(result.retrieved_ids) > 0
    assert "account-guide.md:0" in result.hits or "account-guide.md:1" in result.hits
    assert result.recall > 0.0
    assert 0.0 <= result.precision <= 1.0
    assert result.hit_at_k is True
    assert result.reciprocal_rank > 0.0


def test_batch_query_evaluation_aggregate_metrics(indexed_eval_vector_db, sample_labelled_queries):
    """Tests evaluate_queries across batch dataset and verifies summary metrics."""
    vdb, collection_name = indexed_eval_vector_db
    evaluator = RetrievalEvaluator(vector_db=vdb, default_collection=collection_name)

    metrics = evaluator.evaluate_queries(
        labelled_queries=sample_labelled_queries,
        k=5,
        collection_name=collection_name
    )

    assert isinstance(metrics, AggregateMetrics)
    assert metrics.total_queries == len(sample_labelled_queries)
    assert metrics.k == 5
    assert 0.0 <= metrics.avg_recall <= 1.0
    assert 0.0 <= metrics.avg_precision <= 1.0
    assert metrics.hit_rate > 0.0
    assert metrics.passed_queries_count + metrics.failed_queries_count == len(sample_labelled_queries)
    assert len(metrics.results) == len(sample_labelled_queries)


def test_evaluate_across_k_progression(indexed_eval_vector_db, sample_labelled_queries):
    """Verifies that recall increases or stays stable as top-k cutoff increases."""
    vdb, collection_name = indexed_eval_vector_db
    evaluator = RetrievalEvaluator(vector_db=vdb, default_collection=collection_name)

    k_results = evaluator.evaluate_across_k(
        labelled_queries=sample_labelled_queries,
        k_values=[1, 3, 5],
        collection_name=collection_name
    )

    assert set(k_results.keys()) == {1, 3, 5}
    # Recall at k=5 should be greater than or equal to recall at k=1
    assert k_results[5].avg_recall >= k_results[1].avg_recall


def test_failure_inspection_and_diagnosis(indexed_eval_vector_db):
    """Tests failure cause diagnosis when top-k is smaller than relevant chunk set."""
    vdb, collection_name = indexed_eval_vector_db
    evaluator = RetrievalEvaluator(vector_db=vdb, default_collection=collection_name)

    # Query with 2 expected relevant chunks, evaluated at k=1
    multi_chunk_query = LabelledQuery(
        query="How can a learner reset their password?",
        relevant_chunk_ids={"account-guide.md:0", "account-guide.md:1"}
    )

    result_k1 = evaluator.evaluate_query(query_item=multi_chunk_query, k=1, collection_name=collection_name)
    assert result_k1.recall < 1.0  # Max recall at k=1 for 2 relevant chunks is 0.5
    assert result_k1.failure_cause is not None
    assert "TOO_SMALL_K" in result_k1.failure_cause

    metrics_k1 = evaluator.evaluate_queries([multi_chunk_query], k=1, collection_name=collection_name)
    failures = metrics_k1.failure_analysis
    assert len(failures) == 1
    assert failures[0]["query"] == multi_chunk_query.query
    assert "actionable_recommendation" in failures[0]


def test_export_results(tmp_path, indexed_eval_vector_db, sample_labelled_queries):
    """Tests JSON and TXT export functionality."""
    vdb, collection_name = indexed_eval_vector_db
    evaluator = RetrievalEvaluator(vector_db=vdb, default_collection=collection_name)

    metrics = evaluator.evaluate_queries(sample_labelled_queries, k=5, collection_name=collection_name)

    json_file = str(tmp_path / "eval_results.json")
    txt_file = str(tmp_path / "eval_audit.log")

    evaluator.export_results(metrics, json_file, txt_file)

    assert os.path.exists(json_file)
    assert os.path.exists(txt_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["assignment"] == "3.36 Retrieval Evaluation & Recall Testing"
        assert data["total_queries"] == len(sample_labelled_queries)
        assert "summary_metrics" in data

    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "KNOVERA RAG ASSISTANT — RETRIEVAL EVALUATION & RECALL TESTING AUDIT LOG" in content
        assert "Average Recall@5" in content
