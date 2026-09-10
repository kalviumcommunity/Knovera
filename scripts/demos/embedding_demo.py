"""
embedding_demo.py

Interactive demonstration for Assignment 3.25: Embeddings Fundamentals & Vector Representation.
Generates embeddings, reports vector dimensions, calculates cosine similarity using NumPy,
compares similar vs dissimilar texts, and outputs audit logs & sample JSON exports.
"""

import os
import sys
import json
import numpy as np
from src.embedding_generator import EmbeddingGenerator, cosine_similarity

def main():
    print("=" * 80)
    print(" KNOVERA RAG ASSISTANT - ASSIGNMENT 3.25: EMBEDDINGS FUNDAMENTALS ")
    print("=" * 80)

    # Initialize Embedding Generator
    generator = EmbeddingGenerator()
    mode_str = "OpenAI / OpenRouter API" if generator.client else "Offline Deterministic Semantic Vector Engine"
    print(f"\n[INIT] Active Vector Engine: {mode_str}")
    print(f"[INIT] Target Embedding Model: {generator.model_name}")
    print(f"[INIT] Target Dimension: {generator.dimension} coordinates per vector\n")

    # Task 1 - Define sample texts (similar pair vs dissimilar text)
    sample_texts = [
        "How do I reset my account password?",
        "Steps to recover access to my login",
        "The cafeteria menu has pasta today",
        "How can I change my user password?",
        "What options are available for lunch in the canteen?"
    ]

    print("-" * 80)
    print(" TASK 1: GENERATE EMBEDDINGS FOR SAMPLE TEXTS")
    print("-" * 80)
    for idx, t in enumerate(sample_texts):
        print(f"  [{idx}] '{t}'")

    print("\nGenerating embeddings...")
    embeddings = generator.embed(sample_texts)
    print(f"Successfully generated {len(embeddings)} embedding vectors.")

    # Task 2 - Report vector dimension & verify uniform length
    print("\n" + "-" * 80)
    print(" TASK 2: REPORT VECTOR DIMENSION & VERIFY UNIFORMITY")
    print("-" * 80)
    
    dimension = generator.get_dimension(embeddings)
    is_uniform, exp_dim, lengths = generator.verify_dimensions(embeddings)

    print(f"Reported Vector Dimension: {dimension}")
    print(f"Vector Lengths per Text: {lengths}")
    print(f"Uniformity Status: {'PASSED (All vectors have identical length)' if is_uniform else 'FAILED'}")

    print("\nFirst 8 Float Coordinates preview for each sample text:")
    for idx, t in enumerate(sample_texts):
        preview = [round(val, 6) for val in embeddings[idx][:8]]
        print(f"  Text [{idx}] (Dim {len(embeddings[idx])}): {preview} ...")

    # Task 3 - Compare similar and dissimilar texts using Cosine Similarity
    print("\n" + "-" * 80)
    print(" TASK 3: COMPARE SIMILAR AND DISSIMILAR TEXTS (COSINE SIMILARITY)")
    print("-" * 80)

    print("Cosine Similarity Formula: cos(theta) = (a . b) / (||a|| * ||b||)")
    print("Using numpy.dot and numpy.linalg.norm for vector algebra.\n")

    # Calculate specific target comparisons
    sim_pair_1 = cosine_similarity(embeddings[0], embeddings[1])
    sim_pair_2 = cosine_similarity(embeddings[0], embeddings[3])
    dissim_pair_1 = cosine_similarity(embeddings[0], embeddings[2])
    dissim_pair_2 = cosine_similarity(embeddings[1], embeddings[2])
    sim_pair_dining = cosine_similarity(embeddings[2], embeddings[4])

    print("Target Pair Similarity Scores:")
    print(f"  1. [SIMILAR - Auth] 'reset password' vs 'recover access login':")
    print(f"     -> Cosine Similarity = {sim_pair_1:.4f}")
    
    print(f"  2. [SIMILAR - Auth] 'reset password' vs 'change user password':")
    print(f"     -> Cosine Similarity = {sim_pair_2:.4f}")

    print(f"  3. [SIMILAR - Dining] 'cafeteria menu' vs 'lunch in canteen':")
    print(f"     -> Cosine Similarity = {sim_pair_dining:.4f}")

    print(f"  4. [DISSIMILAR - Cross Domain] 'reset password' vs 'cafeteria menu':")
    print(f"     -> Cosine Similarity = {dissim_pair_1:.4f}")

    print(f"  5. [DISSIMILAR - Cross Domain] 'recover access login' vs 'cafeteria menu':")
    print(f"     -> Cosine Similarity = {dissim_pair_2:.4f}")

    # Assertion verification
    assert sim_pair_1 > dissim_pair_1, "Error: Similar pair 1 score should be higher than dissimilar pair 1!"
    assert sim_pair_2 > dissim_pair_1, "Error: Similar pair 2 score should be higher than dissimilar pair 1!"
    print("\n[VERIFICATION] ASSERTION PASSED: Similar text pairs score significantly higher than dissimilar pairs.")
    print(f"  Difference (Similar vs Dissimilar): {sim_pair_1 - dissim_pair_1:+.4f}")

    # Full Pairwise Similarity Matrix
    print("\nPairwise Similarity Matrix:")
    print(f"{'Idx':<5} | " + " | ".join([f"[{i}]" for i in range(len(sample_texts))]))
    print("-" * 55)
    for i in range(len(sample_texts)):
        row_str = f"[{i}]   | "
        for j in range(len(sample_texts)):
            s = cosine_similarity(embeddings[i], embeddings[j])
            row_str += f"{s:6.3f} "
        print(row_str)

    # Task 4 - Explain what vectors represent
    print("\n" + "-" * 80)
    print(" TASK 4: EXPLANATION - WHAT EMBEDDING VECTORS REPRESENT")
    print("-" * 80)
    
    explanation_note = (
        "1. NUMERIC REPRESENTATION OF SEMANTIC MEANING:\n"
        "   Embedding vectors translate textual concepts into dense, high-dimensional numeric coordinate spaces.\n"
        "   Unlike arbitrary database auto-increment IDs or sparse keyword counts (e.g., Bag-of-Words / One-Hot encodings),\n"
        "   embeddings map semantic relationships into geometric proximity.\n\n"
        "2. GEOMETRIC NEIGHBORHOODS & KEYWORD INDEPENDENCE:\n"
        "   Texts that convey identical or related intent ('reset password' vs 'recover login access') land close to\n"
        "   each other in vector space (high cosine similarity ~0.80 - 0.99), even when sharing zero exact vocabulary words.\n"
        "   Conversely, texts discussing unrelated topics ('cafeteria menu') lie nearly orthogonal in vector space (~0.00).\n\n"
        "3. WHY EMBEDDINGS ENABLE SEMANTIC SEARCH IN RAG:\n"
        "   Traditional keyword search fails when users ask questions using different vocabulary than the corpus documents.\n"
        "   In a RAG assistant, chunked documents and user questions are both converted to embedding vectors.\n"
        "   Retrieval becomes a nearest-neighbor vector search, finding document chunks that match the *meaning* of\n"
        "   the query regardless of exact phrasing."
    )
    print(explanation_note)

    # Task 5 - Export Outputs & Save Demonstration Artifacts
    print("\n" + "-" * 80)
    print(" TASK 5: EXPORT DEMONSTRATION LOGS AND JSON ARTIFACTS")
    print("-" * 80)

    # Prepare JSON structure
    export_data = {
        "assignment": "3.25 Embeddings Fundamentals & Vector Representation",
        "engine_mode": mode_str,
        "model_name": generator.model_name,
        "vector_dimension": dimension,
        "is_dimension_uniform": is_uniform,
        "sample_texts": sample_texts,
        "trimmed_vectors": {
            f"text_{idx}": {
                "text": text,
                "vector_dimension": len(embeddings[idx]),
                "first_8_values": [round(val, 6) for val in embeddings[idx][:8]]
            }
            for idx, text in enumerate(sample_texts)
        },
        "similarity_results": {
            "similar_pair_auth_1": {
                "pair": [sample_texts[0], sample_texts[1]],
                "cosine_similarity": round(sim_pair_1, 4),
                "relationship": "similar"
            },
            "similar_pair_auth_2": {
                "pair": [sample_texts[0], sample_texts[3]],
                "cosine_similarity": round(sim_pair_2, 4),
                "relationship": "similar"
            },
            "similar_pair_dining": {
                "pair": [sample_texts[2], sample_texts[4]],
                "cosine_similarity": round(sim_pair_dining, 4),
                "relationship": "similar"
            },
            "dissimilar_pair_1": {
                "pair": [sample_texts[0], sample_texts[2]],
                "cosine_similarity": round(dissim_pair_1, 4),
                "relationship": "dissimilar"
            },
            "dissimilar_pair_2": {
                "pair": [sample_texts[1], sample_texts[2]],
                "cosine_similarity": round(dissim_pair_2, 4),
                "relationship": "dissimilar"
            }
        },
        "explanation_note": explanation_note
    }

    # Save sample_embeddings.json
    json_path = "sample_embeddings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
    print(f"Exported sample embeddings & metrics to: {json_path}")

    # Save output log file
    os.makedirs("outputs", exist_ok=True)
    log_path = os.path.join("outputs", "embedding_output.txt")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"KNOVERA RAG ASSISTANT - ASSIGNMENT 3.25 EXECUTION AUDIT LOG\n")
        f.write(f"Vector Engine: {mode_str}\n")
        f.write(f"Model: {generator.model_name}\n")
        f.write(f"Vector Dimension: {dimension}\n")
        f.write(f"Dimension Uniformity: {is_uniform}\n\n")
        f.write("SAMPLE TEXTS & PREVIEWS:\n")
        for idx, t in enumerate(sample_texts):
            p = [round(val, 6) for val in embeddings[idx][:8]]
            f.write(f"  [{idx}] '{t}' -> Dim: {len(embeddings[idx])}, First 8: {p}\n")
        f.write("\nSIMILARITY SCORES:\n")
        f.write(f"  Similar Auth (0 vs 1): {sim_pair_1:.4f}\n")
        f.write(f"  Similar Auth (0 vs 3): {sim_pair_2:.4f}\n")
        f.write(f"  Similar Dining (2 vs 4): {sim_pair_dining:.4f}\n")
        f.write(f"  Dissimilar (0 vs 2): {dissim_pair_1:.4f}\n")
        f.write(f"  Dissimilar (1 vs 2): {dissim_pair_2:.4f}\n\n")
        f.write("EXPLANATION NOTE:\n")
        f.write(explanation_note + "\n")
        
    print(f"Saved execution audit log to: {log_path}")
    print("\n" + "=" * 80)
    print(" DEMONSTRATION COMPLETED SUCCESSFULLY ")
    print("=" * 80)

if __name__ == "__main__":
    main()
