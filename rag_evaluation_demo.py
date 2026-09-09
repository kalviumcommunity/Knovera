"""
rag_evaluation_demo.py

Interactive Demonstration: End-to-End RAG System Evaluation & Answer Quality Scoring (Assignment 3.43)

Demonstrates:
1. End-to-end evaluation of RAG system responses across Correctness, Grounding, and Citation Accuracy.
2. Indexing a multi-topic reference document corpus into ChromaDB.
3. Scoring test set questions with ground-truth expected points and expected sources.
4. Automated root-cause failure diagnosis for sub-optimal answers.
5. Generating an overall quality summary and exporting scored results to JSON and TXT logs.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure Knovera root is in sys.path
_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from src.vector_store import VectorDatabase
from src.corpus_indexer import CorpusIndexer
from src.embedding_generator import EmbeddingGenerator
from src.rag_evaluator import RAGEvaluator

# Configure logging and output directory
os.makedirs("outputs", exist_ok=True)
log_file_path = "outputs/rag_evaluation_output.txt"
json_export_path = "sample_rag_evaluation_results.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Evaluation Reference Corpus
EVALUATION_CORPUS = [
    {
        "id": "submission-rubric.md:0",
        "text": "Project submission evidence requirement: Final project submissions require a GitHub pull request link, a sample output JSON file, and a 3-5 minute video explanation demonstrating working tests and pipeline execution.",
        "metadata": {
            "source": "submission-rubric.md",
            "doc_title": "Project Submission Guidelines",
            "section": "Submission Evidence",
            "category": "Academics"
        }
    },
    {
        "id": "guardrails.md:0",
        "text": "System Guardrails & Missing Context Behavior: When supporting context is missing or insufficient to answer a query, the system must refuse to answer and state 'I don't have enough reliable context to answer that.' The system must never hallucinate facts.",
        "metadata": {
            "source": "guardrails.md",
            "doc_title": "RAG Guardrails & Safety",
            "section": "Fallback Behavior",
            "category": "Safety"
        }
    },
    {
        "id": "video-policy.md:0",
        "text": "Video explanation guidelines: The video explanation must be recorded as a 3-5 minute screen-share walkthrough, uploaded to Google Drive with permissions set to 'Anyone with the link can view', and tested in an incognito window.",
        "metadata": {
            "source": "video-policy.md",
            "doc_title": "Video Recording Guidelines",
            "section": "Sharing Permissions",
            "category": "Media"
        }
    },
    {
        "id": "vector-db-specs.md:0",
        "text": "Vector Database Specifications: For unit-normalized embedding vectors (such as text-embedding-3-small), Cosine Similarity and Dot Product produce identical similarity orderings. Cosine Similarity is recommended.",
        "metadata": {
            "source": "vector-db-specs.md",
            "doc_title": "Vector Store Engine Specs",
            "section": "Distance Metrics",
            "category": "Architecture"
        }
    },
    {
        "id": "chunking-guide.md:0",
        "text": "Chunking Strategy Recommendations: For dense technical source code and JSON data, use a smaller chunk size of 256 tokens and preserve logical function/class boundaries to maintain contextual coherence.",
        "metadata": {
            "source": "chunking-guide.md",
            "doc_title": "Text & Code Chunking Standards",
            "section": "Dense Code Sizing",
            "category": "Ingestion"
        }
    }
]


def load_test_set() -> List[Dict[str, Any]]:
    """Loads the test set from data/sample_rag_test_set.json or fallback list."""
    test_set_path = Path("data/sample_rag_test_set.json")
    if test_set_path.exists():
        with open(test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Convert expected_sources list to set for evaluation
            for item in data:
                item["expected_sources"] = set(item.get("expected_sources", []))
            return data
    else:
        return [
            {
                "id": "q1",
                "question": "What evidence is required for project submission?",
                "expected_points": ["PR link", "sample output", "video explanation"],
                "expected_sources": {"submission-rubric.md"}
            },
            {
                "id": "q2",
                "question": "What should the system do when context is missing?",
                "expected_points": ["refuse", "say not enough information"],
                "expected_sources": {"guardrails.md"}
            }
        ]


def main():
    print("=" * 80)
    print("  KNOVERA RAG ASSISTANT - E2E EVALUATION & ANSWER QUALITY SCORING DEMO")
    print("=" * 80)

    # 1. Initialize Generator and Vector DB Collection
    generator = EmbeddingGenerator()
    vector_db = VectorDatabase(in_memory=True, embedding_generator=generator)
    collection_name = "eval_demo_collection"
    
    # Ingest Evaluation Corpus Chunks
    print("\n[Stage 1] Ingesting Evaluation Corpus Chunks into ChromaDB...")
    indexer = CorpusIndexer(vector_db=vector_db, default_collection=collection_name)
    embedded_records = generator.embed_chunks(EVALUATION_CORPUS)
    indexer.index_corpus(embedded_records, collection_name=collection_name, reset_collection=True)

    # 2. Load Evaluation Test Set
    print("\n[Stage 2] Loading Evaluation Test Set...")
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test set questions.")

    # 3. Instantiate Evaluator
    evaluator = RAGEvaluator(
        vector_db=vector_db,
        generator=generator,
        collection_name=collection_name,
        use_api=False
    )

    # 4. Execute End-to-End Evaluation Suite
    print("\n[Stage 3] Executing End-to-End RAG Answer Quality Evaluation...")
    eval_results = evaluator.evaluate_test_set(test_set)
    rows = eval_results["rows"]
    summary = eval_results["summary"]

    # 5. Print Detailed Per-Query Evaluation Table
    print("\n" + "=" * 100)
    print(f"{'#':<3} | {'QUESTION':<45} | {'CORR':<5} | {'GRND':<5} | {'CITE':<5} | {'FAILURE CAUSE'}")
    print("=" * 100)

    for i, r in enumerate(rows, 1):
        q_text = (r["question"][:42] + "...") if len(r["question"]) > 45 else r["question"]
        corr_str = f"{r['correctness']:.2f}"
        grnd_str = f"{r['grounding']:.2f}"
        cite_str = f"{r['citation_accuracy']:.2f}"
        fail_str = r["failure_cause"] or "NONE (PASS)"

        print(f"{i:<3} | {q_text:<45} | {corr_str:<5} | {grnd_str:<5} | {cite_str:<5} | {fail_str}")

    print("=" * 100)

    # 6. Print Aggregate Summary Metrics
    print("\n" + "=" * 60)
    print("           EVALUATION SUMMARY & QUALITY METRICS")
    print("=" * 60)
    print(f" Total Questions Evaluated : {summary['questions']}")
    print(f" Average Correctness Score : {summary['avg_correctness']:.4f} ({(summary['avg_correctness']*100):.1f}%)")
    print(f" Average Grounding Score   : {summary['avg_grounding']:.4f} ({(summary['avg_grounding']*100):.1f}%)")
    print(f" Average Citation Accuracy : {summary['avg_citation_accuracy']:.4f} ({(summary['avg_citation_accuracy']*100):.1f}%)")
    print(f" Overall Quality Score     : {summary['overall_quality_score']:.4f} ({(summary['overall_quality_score']*100):.1f}%)")
    print(f" Weakest Quality Dimension : {summary['weakest_dimension'].upper()}")
    print(f" Total Quality Failures    : {summary['failures_count']}")
    print("=" * 60)

    # 7. Print Failure Deep-Dive
    if summary["failures_count"] > 0:
        print("\n[Stage 4] Inspecting Quality Failure Cases & Diagnosed Root Causes:")
        for idx, f_row in enumerate(summary["failures"], 1):
            print(f"\nFailure #{idx}:")
            print(f"  Question        : {f_row['question']}")
            print(f"  Answer          : {f_row['answer']}")
            print(f"  Scores          : Correctness={f_row['correctness']}, Grounding={f_row['grounding']}, Citations={f_row['citation_accuracy']}")
            print(f"  Cited Sources   : {f_row['citations']}")
            print(f"  Expected Sources: {f_row['expected_sources']}")
            print(f"  Diagnosed Cause : {f_row['failure_cause']}")
    else:
        print("\nAll evaluation examples passed quality standards with 100% scores across all dimensions!")

    # 8. Export Results to JSON
    export_payload = {
        "summary": summary,
        "results": rows
    }
    with open(json_export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n[Export] Full evaluation payload exported to '{json_export_path}'.")
    print(f"[Export] Log audit saved to '{log_file_path}'.")


if __name__ == "__main__":
    main()
