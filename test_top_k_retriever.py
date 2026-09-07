"""
test_top_k_retriever.py

Comprehensive Pytest suite for Assignment 3.32: Similarity Search & Top-K Retrieval.
Validates model alignment, top-k retrieval, score calculation, metadata payload integrity,
and multi-k comparative analysis.
"""

import pytest
import chromadb
from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.top_k_retriever import TopKRetriever


@pytest.fixture
def sample_embedded_chunks():
    """Provides a sample multi-domain embedded corpus for retrieval tests."""
    generator = EmbeddingGenerator()
    chunks = [
        {
            "id": "account-guide.md:0",
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Recovery", "doc_title": "Account Admin Guide", "category": "Authentication"}
        },
        {
            "id": "account-guide.md:1",
            "text": "Learners can recover access using their registered email address. A one-time secure verification link is dispatched immediately.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Verification", "doc_title": "Account Admin Guide", "category": "Authentication"}
        },
        {
            "id": "account-guide.md:2",
            "text": "Two-factor authentication (2FA) is mandatory for enterprise admin consoles. Register TOTP authenticator app or hardware key.",
            "metadata": {"source": "account-guide.md", "chunk_index": 2, "section": "Multi-Factor Security", "doc_title": "Account Admin Guide", "category": "Authentication"}
        },
        {
            "id": "service-policy.md:0",
            "text": "Knovera Customer Service & Refund Policy: Enterprise clients are eligible for full subscription refunds within 14 days of provisioning.",
            "metadata": {"source": "service-policy.md", "chunk_index": 0, "section": "Refund Eligibility", "doc_title": "Service Policy Guide", "category": "Billing"}
        },
        {
            "id": "campus-guide.md:0",
            "text": "Campus cafeteria hours and lunch menu: Operating from 11:30 AM to 2:30 PM offering fresh garden salads and daily specials.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 0, "section": "Dining Hours", "doc_title": "Campus Guide", "category": "Campus Life"}
        }
    ]

    records = generator.embed_chunks(chunks)
    return records


@pytest.fixture
def indexed_vector_db(sample_embedded_chunks):
    """Initializes ephemeral vector store and populates sample collection."""
    vdb = VectorDatabase(in_memory=True)
    indexer = CorpusIndexer(vector_db=vdb, default_collection="test_top_k_collection")
    indexer.index_corpus(sample_embedded_chunks, reset_collection=True)
    return vdb, "test_top_k_collection"


def test_top_k_retriever_initialization():
    """Tests default initialization of TopKRetriever."""
    retriever = TopKRetriever()
    assert retriever.generator is not None
    assert retriever.vector_db is not None
    assert retriever.default_collection == "knovera_top_k_chunks"


def test_model_alignment_verification(indexed_vector_db):
    """Tests model and dimension alignment verification."""
    vdb, col_name = indexed_vector_db
    retriever = TopKRetriever(vector_db=vdb)
    
    alignment_info = retriever.verify_model_alignment(collection_name=col_name)
    assert alignment_info["is_aligned"] is True
    assert alignment_info["status"] == "ALIGNED"
    assert alignment_info["query_dimension"] == alignment_info["collection_expected_dimension"]
    assert alignment_info["collection_record_count"] == 5


def test_single_top_k_retrieval(indexed_vector_db):
    """Task 1-3: Tests single top-k similarity search returning scores and metadata."""
    vdb, col_name = indexed_vector_db
    retriever = TopKRetriever(vector_db=vdb)

    query = "How can a learner reset their password?"
    results = retriever.retrieve(query=query, k=3, collection_name=col_name)

    assert len(results) == 3

    # Verify score ranking order (descending similarity)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # Verify top result is password reset chunk
    top_result = results[0]
    assert top_result["rank"] == 1
    assert "password" in top_result["text"].lower() or "recover" in top_result["text"].lower()
    assert top_result["metadata"]["source"] == "account-guide.md"
    assert top_result["metadata"]["chunk_index"] in (0, 1)

    # Verify payload schema match
    for res in results:
        assert "rank" in res
        assert "score" in res
        assert "distance" in res
        assert "text" in res
        assert "metadata" in res
        assert "source" in res["metadata"]
        assert "chunk_index" in res["metadata"]
        assert "id" in res


def test_demonstrate_changing_k(indexed_vector_db):
    """Task 4: Tests running the query across k=1, k=3, and k=5."""
    vdb, col_name = indexed_vector_db
    retriever = TopKRetriever(vector_db=vdb)

    query = "How can a learner reset their password?"
    comparison = retriever.compare_k(query=query, k_values=[1, 3, 5], collection_name=col_name)

    assert comparison["query"] == query
    assert len(comparison["runs"]) == 3

    run_k1 = comparison["runs"][0]
    run_k3 = comparison["runs"][1]
    run_k5 = comparison["runs"][2]

    assert run_k1["k"] == 1 and len(run_k1["results"]) == 1
    assert run_k3["k"] == 3 and len(run_k3["results"]) == 3
    assert run_k5["k"] == 5 and len(run_k5["results"]) == 5

    # Top result should be identical across all k values
    top_k1_id = run_k1["results"][0]["id"]
    top_k3_id = run_k3["results"][0]["id"]
    top_k5_id = run_k5["results"][0]["id"]
    assert top_k1_id == top_k3_id == top_k5_id

    # Context character count must expand with larger k
    assert run_k1["total_context_characters"] < run_k3["total_context_characters"] < run_k5["total_context_characters"]


def test_edge_cases_and_input_validation(indexed_vector_db):
    """Tests handling of empty strings, whitespace, and invalid k values."""
    vdb, col_name = indexed_vector_db
    retriever = TopKRetriever(vector_db=vdb)

    # Empty / whitespace query
    assert retriever.retrieve("", k=3, collection_name=col_name) == []
    assert retriever.retrieve("   ", k=3, collection_name=col_name) == []

    # Invalid k <= 0
    with pytest.raises(ValueError):
        retriever.retrieve("Valid query", k=0, collection_name=col_name)

    with pytest.raises(ValueError):
        retriever.retrieve("Valid query", k=-5, collection_name=col_name)
