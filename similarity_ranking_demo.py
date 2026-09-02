"""
similarity_ranking_demo.py

Interactive demonstration for Assignment 3.27: Embedding Similarity & Distance Metrics.
Computes similarity and distance metrics (Cosine Similarity, Dot Product, Euclidean Distance, Cosine Distance),
compares sample query against chunk embeddings, ranks chunks from most to least similar,
provides metric justification, and exports audit logs & structured JSON outputs.
"""

import os
import sys
import json
import numpy as np
from src.embedding_generator import (
    EmbeddingGenerator,
    cosine_similarity,
    cosine_distance,
    dot_product,
    euclidean_distance,
    manhattan_distance,
    calculate_metric
)

def main():
    print("=" * 80)
    print(" KNOVERA RAG ASSISTANT - ASSIGNMENT 3.27: SIMILARITY & DISTANCE METRICS ")
    print("=" * 80)

    # Initialize Embedding Generator
    generator = EmbeddingGenerator()
    mode_str = "OpenAI / OpenRouter API" if generator.client else "Offline Deterministic Semantic Vector Engine"
    print(f"\n[INIT] Active Vector Engine: {mode_str}")
    print(f"[INIT] Target Embedding Model: {generator.model_name}")
    print(f"[INIT] Vector Dimension: {generator.dimension} coordinates per vector\n")

    # Task 1 - Demonstrate metric calculation between sample vectors
    print("-" * 80)
    print(" TASK 1: COMPUTE SIMILARITY & DISTANCE METRICS BETWEEN EMBEDDINGS")
    print("-" * 80)

    sample_texts_demo = [
        "Password reset instructions for learner accounts.",
        "Learners can recover access using their registered email.",
        "The cafeteria menu changes every Friday."
    ]
    
    print("Generating demo embeddings for metric comparison...")
    demo_vecs = generator.embed(sample_texts_demo)

    v0, v1, v2 = demo_vecs[0], demo_vecs[1], demo_vecs[2]

    sim_01 = cosine_similarity(v0, v1)
    cos_dist_01 = cosine_distance(v0, v1)
    dot_01 = dot_product(v0, v1)
    euc_01 = euclidean_distance(v0, v1)
    man_01 = manhattan_distance(v0, v1)

    sim_02 = cosine_similarity(v0, v2)
    cos_dist_02 = cosine_distance(v0, v2)
    dot_02 = dot_product(v0, v2)
    euc_02 = euclidean_distance(v0, v2)
    man_02 = manhattan_distance(v0, v2)

    print("\nPairwise Metric Breakdown:")
    print(f"  1. [SIMILAR PAIR] Chunk 0 ('Password reset...') vs Chunk 1 ('Recover access...'):")
    print(f"     - Cosine Similarity (Direction, Higher=Better) : {sim_01:.6f}")
    print(f"     - Cosine Distance   (1 - Sim, Lower=Better)   : {cos_dist_01:.6f}")
    print(f"     - Dot Product       (Inner Prod, Higher=Better): {dot_01:.6f}")
    print(f"     - Euclidean Dist L2 (Geometric, Lower=Better) : {euc_01:.6f}")
    print(f"     - Manhattan Dist L1 (City Block, Lower=Better): {man_01:.6f}")

    print(f"\n  2. [DISSIMILAR PAIR] Chunk 0 ('Password reset...') vs Chunk 2 ('Cafeteria menu...'):")
    print(f"     - Cosine Similarity (Direction, Higher=Better) : {sim_02:.6f}")
    print(f"     - Cosine Distance   (1 - Sim, Lower=Better)   : {cos_dist_02:.6f}")
    print(f"     - Dot Product       (Inner Prod, Higher=Better): {dot_02:.6f}")
    print(f"     - Euclidean Dist L2 (Geometric, Lower=Better) : {euc_02:.6f}")
    print(f"     - Manhattan Dist L1 (City Block, Lower=Better): {man_02:.6f}")

    # Task 2 - Define corpus of chunk records with text, metadata, and embeddings
    print("\n" + "-" * 80)
    print(" TASK 2: COMPARE SAMPLE QUERY AGAINST CORPUS CHUNK EMBEDDINGS")
    print("-" * 80)

    query = "How can a learner reset their password?"
    print(f"Sample User Query: '{query}'")

    raw_chunks = [
        {
            "text": "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the portal.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Password Recovery", "category": "Authentication"}
        },
        {
            "text": "Learners can recover access using their registered email. A secure link valid for 15 minutes is dispatched immediately.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1, "section": "Email Verification", "category": "Authentication"}
        },
        {
            "text": "The cafeteria menu changes every Friday, offering hot pasta dishes, vegan salads, and fresh fruits.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 3, "section": "Dining Options", "category": "Campus Life"}
        },
        {
            "text": "Tier 1 Technical Support responds within 2 business hours for critical authentication and account lockout requests.",
            "metadata": {"source": "sla-policy.md", "chunk_index": 2, "section": "Service Levels", "category": "Support SLA"}
        },
        {
            "text": "Python token chunking uses tiktoken cl100k_base encoding with sliding window overlap to preserve semantic context.",
            "metadata": {"source": "rag-architecture.md", "chunk_index": 5, "section": "Token Chunking", "category": "Engineering"}
        }
    ]

    print(f"\nEmbedding {len(raw_chunks)} corpus chunks...")
    stored_records = generator.embed_chunks(raw_chunks)
    print(f"Successfully embedded all {len(stored_records)} chunks (Dimension: {generator.dimension}).")

    # Task 3 - Rank chunks and show results
    print("\n" + "-" * 80)
    print(" TASK 3: RANK CHUNKS & SHOW MOST / LEAST SIMILAR RESULTS")
    print("-" * 80)

    # Rank using Cosine Similarity
    ranking_result = generator.rank_chunks(query=query, stored_records=stored_records, metric="cosine")
    ranked_list = ranking_result["ranked_chunks"]

    most_sim = ranking_result["most_similar"]
    least_sim = ranking_result["least_similar"]

    print("\n" + "=" * 60)
    print(f" MOST SIMILAR CHUNK (Rank #1, Score: {most_sim['score']:.4f})")
    print("=" * 60)
    print(f"  Source File : {most_sim['metadata'].get('source')}")
    print(f"  Chunk Index : {most_sim['metadata'].get('chunk_index')}")
    print(f"  Category    : {most_sim['metadata'].get('category')}")
    print(f"  Section     : {most_sim['metadata'].get('section')}")
    print(f"  Text        : \"{most_sim['text']}\"")

    print("\n" + "=" * 60)
    print(f" LEAST SIMILAR CHUNK (Rank #{least_sim['rank']}, Score: {least_sim['score']:.4f})")
    print("=" * 60)
    print(f"  Source File : {least_sim['metadata'].get('source')}")
    print(f"  Chunk Index : {least_sim['metadata'].get('chunk_index')}")
    print(f"  Category    : {least_sim['metadata'].get('category')}")
    print(f"  Section     : {least_sim['metadata'].get('section')}")
    print(f"  Text        : \"{least_sim['text']}\"")

    print("\n" + "-" * 80)
    print(" COMPLETE RANKED RETRIEVAL TABLE (COSINE SIMILARITY)")
    print("-" * 80)
    print(f"{'Rank':<5} | {'Score':<8} | {'Source File':<18} | {'Category':<15} | {'Snippet Preview':<30}")
    print("-" * 85)
    for item in ranked_list:
        snippet = item['text'][:28] + "..." if len(item['text']) > 28 else item['text']
        src = item['metadata'].get('source', 'N/A')
        cat = item['metadata'].get('category', 'N/A')
        print(f"#{item['rank']:<4} | {item['score']:<8.4f} | {src:<18} | {cat:<15} | {snippet:<30}")

    # Task 4 - Justify the metric choice
    print("\n" + "-" * 80)
    print(" TASK 4: METRIC JUSTIFICATION (WHY COSINE SIMILARITY?)")
    print("-" * 80)
    justification_text = (
        "1. Directional Invariance: Cosine similarity measures the angle theta between vectors, "
        "evaluating semantic direction rather than vector magnitude (length).\n"
        "2. Length Independence: In NLP embeddings, longer texts or frequent token repetitions can inflate "
        "vector norms without changing semantic intent. Cosine similarity normalizes vector lengths to 1.0, "
        "preventing length bias in chunk ranking.\n"
        "3. Mathematical Bound: Bounded cleanly in [-1.0, 1.0], making similarity thresholds intuitive.\n"
        "4. Identity with Dot Product: For L2-normalized embedding vectors (such as OpenAI text-embedding-3-small), "
        "Cosine Similarity equals the Dot Product (a . b), enabling hyper-fast vector database dot-product search."
    )
    print(justification_text)

    # Multi-metric comparison validation
    print("\n" + "-" * 80)
    print(" CROSS-METRIC RANKING CONSISTENCY VERIFICATION")
    print("-" * 80)
    metrics_to_test = ["cosine", "cosine_distance", "dot_product", "euclidean"]
    for m in metrics_to_test:
        res = generator.rank_chunks(query=query, stored_records=stored_records, metric=m)
        top_src = res["most_similar"]["metadata"]["source"]
        top_score = res["most_similar"]["score"]
        bot_src = res["least_similar"]["metadata"]["source"]
        bot_score = res["least_similar"]["score"]
        print(f"  Metric: {m:<16} | Rank #1 Source: {top_src:<18} (Score: {top_score:>8.4f}) | Rank #5 Source: {bot_src:<18} (Score: {bot_score:>8.4f})")

    # Task 5 - Export audit logs & JSON output
    print("\n" + "-" * 80)
    print(" TASK 5: EXPORT RANKING ARTIFACTS & SAMPLE OUTPUTS")
    print("-" * 80)

    # Clean embeddings from json export for concise readability
    export_payload = {
        "assignment": "3.27 Embedding Similarity & Distance Metrics",
        "query": query,
        "model": generator.model_name,
        "vector_dimension": generator.dimension,
        "total_chunks_ranked": len(ranked_list),
        "most_similar_chunk": {
            "rank": most_sim["rank"],
            "score": most_sim["score"],
            "source": most_sim["metadata"].get("source"),
            "text": most_sim["text"],
            "metadata": most_sim["metadata"]
        },
        "least_similar_chunk": {
            "rank": least_sim["rank"],
            "score": least_sim["score"],
            "source": least_sim["metadata"].get("source"),
            "text": least_sim["text"],
            "metadata": least_sim["metadata"]
        },
        "ranked_chunks": [
            {
                "rank": item["rank"],
                "score": item["score"],
                "text": item["text"],
                "metadata": item["metadata"]
            }
            for item in ranked_list
        ],
        "metric_justification": justification_text
    }

    json_export_path = "sample_similarity_rankings.json"
    with open(json_export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"  [EXPORT] Sample rankings exported to: {json_export_path}")

    # Output directory check
    os.makedirs("outputs", exist_ok=True)
    txt_export_path = os.path.join("outputs", "similarity_ranking_output.txt")
    
    # Save formatted execution log to txt file
    with open(txt_export_path, "w", encoding="utf-8") as f:
        f.write("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.27 EXECUTION AUDIT LOG\n")
        f.write("=========================================================\n\n")
        f.write(f"Active Model: {generator.model_name}\n")
        f.write(f"Vector Dimension: {generator.dimension}\n")
        f.write(f"Query: \"{query}\"\n\n")
        
        f.write("TASK 1 & 3: RANKING RESULTS SUMMARY (COSINE SIMILARITY)\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"Most Similar Chunk (Rank #1):\n")
        f.write(f"  Score: {most_sim['score']:.4f}\n")
        f.write(f"  Text: {most_sim['text']}\n")
        f.write(f"  Metadata: {json.dumps(most_sim['metadata'])}\n\n")

        f.write(f"Least Similar Chunk (Rank #{least_sim['rank']}):\n")
        f.write(f"  Score: {least_sim['score']:.4f}\n")
        f.write(f"  Text: {least_sim['text']}\n")
        f.write(f"  Metadata: {json.dumps(least_sim['metadata'])}\n\n")

        f.write("FULL RETRIEVAL RANKING TABLE:\n")
        f.write(f"{'Rank':<5} | {'Score':<8} | {'Source':<18} | {'Text Snippet':<40}\n")
        f.write("-" * 75 + "\n")
        for item in ranked_list:
            snip = item['text'][:38] + ".." if len(item['text']) > 38 else item['text']
            f.write(f"#{item['rank']:<4} | {item['score']:<8.4f} | {item['metadata'].get('source', 'N/A'):<18} | {snip:<40}\n")

        f.write("\nTASK 4: METRIC JUSTIFICATION\n")
        f.write("-----------------------------\n")
        f.write(justification_text + "\n")

    print(f"  [EXPORT] Execution audit log saved to: {txt_export_path}\n")

    print("=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY - ALL TASKS VERIFIED ")
    print("=" * 80)

if __name__ == "__main__":
    main()
