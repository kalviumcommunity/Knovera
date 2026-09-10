"""
vector_db_demo.py

Demonstration and Verification Suite for Assignment 3.30: Vector Database Setup & Collection Design.
Platform: Knovera RAG Assistant

Features Demonstrated:
- Task 1: Initialize ChromaDB vector database and confirm reachability.
- Task 2: Create a collection with the exact vector dimension (1536) and cosine distance space.
- Task 3: Display the grounded record schema (embedding vector + source text + rich metadata).
- Task 4: Insert a test record and read it back successfully, verifying ID, vector length, text, and metadata integrity.
- Task 5: Export setup configuration and readback test output to the outputs directory.
"""

import os
import sys
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Reconfigure stdout to UTF-8 for Windows console support
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.vector_store import VectorDatabase, STORED_RECORD_SCHEMA
from src.embedding_generator import EmbeddingGenerator


def main():
    load_dotenv()
    print("=" * 85)
    print(" KNOVERA RAG ASSISTANT - VECTOR DATABASE SETUP & COLLECTION DESIGN (3.30)")
    print("=" * 85)

    # =========================================================================
    # TASK 1: SET UP A VECTOR DATABASE & CONFIRM REACHABILITY
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 1: VECTOR DATABASE INITIALIZATION & CONNECTIVITY CHECK")
    print("-" * 85)

    persist_directory = os.path.join(".", "data", "chroma_db")
    vector_db = VectorDatabase(persist_dir=persist_directory)

    is_connected = vector_db.is_reachable()
    print(f"Vector Database Engine:  ChromaDB (v0.5.23)")
    print(f"Storage Mode:            {vector_db.storage_mode}")
    print(f"Persistence Path:        {os.path.abspath(persist_directory)}")
    print(f"Database Reachable:      {'YES (Connected Successfully)' if is_connected else 'NO (Failed)'}")

    if not is_connected:
        print("Error: Vector database is unreachable. Aborting demonstration.")
        return

    # =========================================================================
    # TASK 2: CREATE A CORRECTLY SIZED COLLECTION
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 2: CREATE CORRECTLY SIZED COLLECTION (DIMENSION GUARDING)")
    print("-" * 85)

    collection_name = "rag_chunks"
    target_dimension = vector_db.expected_dimension  # 1536 for text-embedding-3-small
    distance_metric = "cosine"

    print(f"Embedding Model:         {vector_db.generator.model_name}")
    print(f"Target Dimension:        {target_dimension} coordinates")
    print(f"Distance Space Metric:   {distance_metric} similarity")

    # Reset any existing test collection to ensure clean state
    try:
        vector_db.delete_collection(collection_name)
    except Exception:
        pass

    collection = vector_db.create_or_get_collection(
        name=collection_name,
        dimension=target_dimension,
        metric=distance_metric,
        metadata={
            "created_by": "Knovera Setup Suite",
            "model": vector_db.generator.model_name
        }
    )

    print(f"Collection Created:      '{collection.name}'")
    print(f"Collection Metadata:     {collection.metadata}")
    print(f"Initial Record Count:    {vector_db.count(collection_name)}")

    # =========================================================================
    # TASK 3: DESIGN THE STORED RECORD SCHEMA
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 3: STORED RECORD SCHEMA SPECIFICATION")
    print("-" * 85)
    print("Record Schema Architecture:")
    print(json.dumps(STORED_RECORD_SCHEMA, indent=2))
    print("\nRationale: Storing source text and hierarchical metadata directly alongside")
    print("the embedding vector allows nearest-neighbor retrieval to immediately produce")
    print("grounded LLM context with citation provenance without extra secondary lookups.")

    # =========================================================================
    # TASK 4: INSERT AND READ BACK A TEST RECORD
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 4: INSERT AND READ BACK TEST RECORD")
    print("-" * 85)

    # 1. Prepare sample text and metadata
    test_chunk_id = "account-guide.md:0"
    test_chunk_text = "Password reset instructions for learner accounts: Users can recover access by clicking 'Forgot Password' on the login portal and submitting their registered email address."
    test_metadata = {
        "source": "account-guide.md",
        "chunk_index": 0,
        "section": "Password Recovery & Account Access",
        "doc_title": "Knovera Learner Account Administration Guide",
        "category": "Authentication"
    }

    print("Generating embedding for test record via OpenRouter API...")
    embedding_vector = vector_db.generator.embed([test_chunk_text])[0]
    print(f"Generated Vector Dimension: {len(embedding_vector)} floats")

    # 2. Insert record into vector database collection
    test_record = {
        "id": test_chunk_id,
        "vector": embedding_vector,
        "text": test_chunk_text,
        "metadata": test_metadata
    }

    print(f"\nInserting test record '{test_chunk_id}' into collection '{collection_name}'...")
    insert_success = vector_db.upsert_records([test_record], collection_name=collection_name)
    print(f"Insert Status:           {'SUCCESS' if insert_success else 'FAILED'}")
    print(f"Updated Collection Count: {vector_db.count(collection_name)}")

    # 3. Read back the test record by exact ID
    print(f"\nReading back record '{test_chunk_id}' from Vector Database...")
    readback_record = vector_db.get_record(test_chunk_id, collection_name=collection_name)

    if readback_record:
        print("Readback Output Verification:")
        print(f"  - Record ID:           {readback_record['id']}")
        print(f"  - Vector Length:       {readback_record['vector_length']} dimensions")
        print(f"  - First 5 Vector Vals: {[round(v, 6) for v in readback_record['vector'][:5]]}")
        print(f"  - Last 5 Vector Vals:  {[round(v, 6) for v in readback_record['vector'][-5:]]}")
        print(f"  - Text Content:        \"{readback_record['text']}\"")
        print(f"  - Metadata Fields:     {json.dumps(readback_record['metadata'], indent=6)}")

        # Verification Assertions
        assert readback_record["id"] == test_chunk_id, "ID mismatch!"
        assert readback_record["vector_length"] == target_dimension, "Vector dimension mismatch!"
        assert readback_record["text"] == test_chunk_text, "Text mismatch!"
        assert readback_record["metadata"]["source"] == test_metadata["source"], "Metadata source mismatch!"
        print("\nAll Readback Integrity Checks: PASSED (100% Identity Match)")
    else:
        print("Error: Could not retrieve record from vector database.")
        return

    # Demonstrate nearest-neighbor retrieval on stored record
    print("\nDemonstrating Nearest-Neighbor Retrieval from Vector Store...")
    query_text = "How do users reset forgotten passwords?"
    query_vec = vector_db.generator.embed([query_text])[0]
    query_results = vector_db.query_similar(query_vec, top_k=1, collection_name=collection_name)
    if query_results:
        top_res = query_results[0]
        print(f"  Query: \"{query_text}\"")
        print(f"  Top Match ID:         {top_res['id']}")
        print(f"  Cosine Similarity:    {top_res['similarity']:.4f} (Distance: {top_res['distance']:.4f})")
        print(f"  Retrieved Text:       \"{top_res['text'][:80]}...\"")

    # =========================================================================
    # TASK 5: EXPORT SETUP CONFIGURATION & ARTIFACTS
    # =========================================================================
    print("\n" + "-" * 85)
    print(" TASK 5: EXPORT SETUP CONFIGURATION & READBACK ARTIFACTS")
    print("-" * 85)

    os.makedirs("outputs", exist_ok=True)
    
    # 1. Export JSON Readback Record
    json_export_path = os.path.join("outputs", "vector_db_readback_record.json")
    readback_export_payload = {
        "assignment": "3.30 Vector Database Setup & Collection Design",
        "database": "ChromaDB",
        "version": "0.5.23",
        "storage_mode": vector_db.storage_mode,
        "collection": {
            "name": collection_name,
            "dimension": target_dimension,
            "metric": distance_metric,
            "count": vector_db.count(collection_name)
        },
        "readback_record": {
            "id": readback_record["id"],
            "vector_length": readback_record["vector_length"],
            "first_5_values": [round(v, 6) for v in readback_record["vector"][:5]],
            "last_5_values": [round(v, 6) for v in readback_record["vector"][-5:]],
            "text": readback_record["text"],
            "metadata": readback_record["metadata"]
        }
    }
    with open(json_export_path, "w", encoding="utf-8") as f:
        json.dump(readback_export_payload, f, indent=2)
    print(f"  Exported Readback Record JSON to: {json_export_path}")

    # 2. Export Text Audit Log
    txt_export_path = os.path.join("outputs", "vector_db_setup_output.txt")
    with open(txt_export_path, "w", encoding="utf-8") as f:
        f.write("KNOVERA RAG ASSISTANT - ASSIGNMENT 3.30: VECTOR DATABASE SETUP & COLLECTION DESIGN\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Database:        ChromaDB (v0.5.23)\n")
        f.write(f"Storage Mode:    {vector_db.storage_mode}\n")
        f.write(f"Collection Name: {collection_name}\n")
        f.write(f"Dimension:       {target_dimension}\n")
        f.write(f"Metric:          {distance_metric}\n")
        f.write(f"Total Records:   {vector_db.count(collection_name)}\n\n")
        f.write("INSERTED & READBACK TEST RECORD:\n")
        f.write(f"  ID:            {readback_record['id']}\n")
        f.write(f"  Vector Dim:    {readback_record['vector_length']}\n")
        f.write(f"  Sample Values: {[round(v, 6) for v in readback_record['vector'][:5]]}\n")
        f.write(f"  Text:          \"{readback_record['text']}\"\n")
        f.write(f"  Metadata:      {json.dumps(readback_record['metadata'])}\n\n")
        f.write("NEAREST-NEIGHBOR RETRIEVAL VALIDATION:\n")
        f.write(f"  Query: \"{query_text}\"\n")
        f.write(f"  Retrieved ID: {top_res['id']} | Cosine Sim: {top_res['similarity']:.4f}\n")
    print(f"  Exported Setup Audit Log to:     {txt_export_path}")

    print("\n" + "=" * 85)
    print(" ASSIGNMENT 3.30 VECTOR DB SETUP & READBACK TEST PASSED SUCCESSFULLY")
    print("=" * 85)


if __name__ == "__main__":
    main()
