"""
test_retrieval_relevance_tuning.py

Comprehensive Pytest suite for Assignment 3.34: Retrieval Relevance Tuning.
Validates test query definitions, retrieval setting configurations, evaluation metrics calculation,
score threshold filtering, metadata filtering, and quantitative best setting selection.
"""

import os
import json
import pytest
from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.retrieval_tuner import (
    TestQuery,
    RetrievalSetting,
    EvaluationRow,
    TuningSummary,
    RetrievalTuner
)


@pytest.fixture
def sample_test_queries():
    """Provides standard ground-truth test queries for relevance tuning."""
    return [
        TestQuery(
            query="How can a learner reset their password?",
            expected_source="account-guide.md",
            category="Authentication"
        ),
        TestQuery(
            query="When does the cafeteria menu change?",
            expected_source="campus-guide.md",
            category="Campus Life"
        ),
        TestQuery(
            query="What evidence is required for project submission?",
            expected_source="submission-rubric.md",
            category="Academics"
        )
    ]


@pytest.fixture
def indexed_vector_db():
    """Populates an in-memory ChromaDB vector store with diverse test document chunks."""
    generator = EmbeddingGenerator()
    chunks = [
        {
            "id": "account-guide.md:0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Authentication"}
        },
        {
            "id": "account-guide.md:1",
            "text": "Learners can recover access using their registered email address. A one-time secure verification link is dispatched immediately.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "doc_type": "guide", "category": "Authentication"}
        },
        {
            "id": "campus-guide.md:0",
            "text": "Campus cafeteria hours and lunch menu: Operating from 11:30 AM to 2:30 PM offering fresh garden salads and daily specials.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "doc_type": "guide", "category": "Campus Life"}
        },
        {
            "id": "submission-rubric.md:0",
            "text": "Project submission evidence requirement: Learners must upload repository GitHub links and a 3-minute video walkthrough.",
            "metadata": {"source": "submission-rubric.md", "chunk_index": 0, "doc_type": "rubric", "category": "Academics"}
        },
        {
            "id": "service-policy.md:0",
            "text": "Customer service SLA and refund eligibility details for enterprise subscription tiers.",
            "metadata": {"source": "service-policy.md", "chunk_index": 0, "doc_type": "policy", "category": "Billing"}
        }
    ]

    records = generator.embed_chunks(chunks)
    vdb = VectorDatabase(in_memory=True)
    indexer = CorpusIndexer(vector_db=vdb, default_collection="test_tuning_collection")
    indexer.index_corpus(records, reset_collection=True)

    return vdb, "test_tuning_collection"


def test_query_and_setting_dataclasses():
    """Validates dataclass construction and dictionary serialization."""
    query = TestQuery(query="Test query?", expected_source="test.md", category="Test")
    assert query.query == "Test query?"
    assert query.expected_source == "test.md"
    assert query.to_dict()["category"] == "Test"

    setting = RetrievalSetting(name="baseline_k3", k=3, min_score=0.5)
    assert setting.name == "baseline_k3"
    assert setting.k == 3
    assert setting.min_score == 0.5
    assert setting.to_dict()["k"] == 3


def test_single_setting_evaluation(indexed_vector_db, sample_test_queries):
    """Tests evaluation of a single retrieval setting and verifies hit rate computation."""
    vdb, collection_name = indexed_vector_db
    tuner = RetrievalTuner(vector_db=vdb, default_collection=collection_name)

    setting = RetrievalSetting(name="baseline_k3", k=3, min_score=0.0)
    summary = tuner.evaluate_setting(setting, sample_test_queries, collection_name=collection_name)

    assert summary.setting_name == "baseline_k3"
    assert summary.total_queries == len(sample_test_queries)
    assert summary.hits >= 0
    assert 0.0 <= summary.hit_rate <= 1.0
    assert len(summary.details) == len(sample_test_queries)


def test_score_threshold_filtering(indexed_vector_db, sample_test_queries):
    """Verifies min_score filters out low confidence chunks."""
    vdb, collection_name = indexed_vector_db
    tuner = RetrievalTuner(vector_db=vdb, default_collection=collection_name)

    permissive_setting = RetrievalSetting(name="permissive", k=5, min_score=0.0)
    strict_setting = RetrievalSetting(name="strict", k=5, min_score=0.999)

    perm_summary = tuner.evaluate_setting(permissive_setting, sample_test_queries, collection_name=collection_name)
    strict_summary = tuner.evaluate_setting(strict_setting, sample_test_queries, collection_name=collection_name)

    assert perm_summary.avg_retained_chunks >= strict_summary.avg_retained_chunks


def test_multi_setting_comparison(indexed_vector_db, sample_test_queries):
    """Tests evaluate_all_settings across multiple retrieval settings."""
    vdb, collection_name = indexed_vector_db
    tuner = RetrievalTuner(vector_db=vdb, default_collection=collection_name)

    settings = [
        RetrievalSetting(name="baseline_k3", k=3, min_score=0.0),
        RetrievalSetting(name="filtered_k3", k=3, metadata_filter={"doc_type": "guide"}, min_score=0.0),
        RetrievalSetting(name="hybrid_k3", k=3, use_hybrid=True, vector_weight=0.7, keyword_weight=0.3)
    ]

    summaries = tuner.evaluate_all_settings(settings, sample_test_queries, collection_name=collection_name)
    assert len(summaries) == 3
    assert summaries[0].setting_name == "baseline_k3"
    assert summaries[1].setting_name == "filtered_k3"
    assert summaries[2].setting_name == "hybrid_k3"


def test_best_setting_selection(indexed_vector_db, sample_test_queries):
    """Validates select_best_setting identifies top-performing summary with justification."""
    vdb, collection_name = indexed_vector_db
    tuner = RetrievalTuner(vector_db=vdb, default_collection=collection_name)

    settings = [
        RetrievalSetting(name="low_k1", k=1, min_score=0.0),
        RetrievalSetting(name="optimal_k3", k=3, min_score=0.0)
    ]

    summaries = tuner.evaluate_all_settings(settings, sample_test_queries, collection_name=collection_name)
    best_summary, justification = tuner.select_best_setting(summaries)

    assert isinstance(best_summary, TuningSummary)
    assert len(justification) > 20
    assert best_summary.hit_rate >= min(s.hit_rate for s in summaries)


def test_export_results(tmp_path, indexed_vector_db, sample_test_queries):
    """Tests JSON and TXT artifact generation."""
    vdb, collection_name = indexed_vector_db
    tuner = RetrievalTuner(vector_db=vdb, default_collection=collection_name)

    setting = RetrievalSetting(name="baseline_k3", k=3, min_score=0.0)
    summary = tuner.evaluate_setting(setting, sample_test_queries, collection_name=collection_name)
    best_summary, justification = tuner.select_best_setting([summary])

    json_file = str(tmp_path / "results.json")
    txt_file = str(tmp_path / "results.txt")

    tuner.export_results([summary], best_summary, justification, json_file, txt_file)

    assert os.path.exists(json_file)
    assert os.path.exists(txt_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["chosen_setting"] == "baseline_k3"
        assert len(data["summary_metrics"]) == 1
