"""
rag_api_demo.py

Demonstration and Verification Script for Knovera RAG Backend API Service (Assignment 3.44).
Executes:
1. Health & Configuration diagnostics via GET /health and GET /config.
2. Grounded Question Answering via POST /query with source citations and latency telemetry.
3. Input validation & error handling (short queries, oversized queries, missing fields).
4. Guardrail refusal response on unsupported out-of-domain queries.
5. Saves formatted sample request and response payloads to sample_api_query_response.json and outputs/.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient

# Ensure project root is in path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.rag_api import create_app, app
from src.api_config import get_config


def print_banner(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def run_api_demonstration():
    """Run comprehensive API test cases and generate sample output files."""
    print_banner("Knovera RAG Backend REST API Demonstration")
    
    cfg = get_config()
    print(f"• Active Environment:   {cfg.app_env}")
    print(f"• Target Host & Port:   {cfg.api_host}:{cfg.api_port}")
    print(f"• Embedding Model:      {cfg.embedding_model}")
    print(f"• Chat Model:           {cfg.chat_model}")
    print(f"• Vector Database:      {cfg.vector_db_url} (Collection: '{cfg.collection_name}')")

    client = TestClient(app)
    
    # ------------------------------------------------------------------------
    # STEP 1: SERVICE HEALTH & CONFIGURATION CHECK
    # ------------------------------------------------------------------------
    print_banner("Step 1: Inspecting GET /health & GET /config Endpoints")
    
    health_resp = client.get("/health")
    print(f"• GET /health Status: {health_resp.status_code}")
    print(json.dumps(health_resp.json(), indent=2))
    
    config_resp = client.get("/config")
    print(f"\n• GET /config Status: {config_resp.status_code}")
    print(json.dumps(config_resp.json(), indent=2))

    # ------------------------------------------------------------------------
    # STEP 2: TEST QUERIES - VALID GROUNDED RAG REQUESTS
    # ------------------------------------------------------------------------
    print_banner("Step 2: Executing Grounded RAG Queries via POST /query")
    
    test_queries = [
        {
            "name": "Service Level Agreement (SLA) & Uptime Question",
            "payload": {
                "question": "What is the Knovera uptime SLA guarantee and refund policy?",
                "k": 4,
                "score_threshold": 0.50
            }
        },
        {
            "name": "Enterprise Security & Multi-Factor Authentication Question",
            "payload": {
                "question": "What are the security requirements for two-factor authentication and password recovery?",
                "k": 3,
                "score_threshold": 0.40
            }
        },
        {
            "name": "SDK Installation and Setup Question",
            "payload": {
                "question": "How do I install the Knovera Python SDK and configure environment variables?",
                "k": 3,
                "score_threshold": 0.40
            }
        }
    ]

    captured_sample: Dict[str, Any] = {}

    for item in test_queries:
        print(f"\n--- Scenario: {item['name']} ---")
        print(f"Request Payload:\n{json.dumps(item['payload'], indent=2)}")
        
        t0 = time.perf_counter()
        resp = client.post("/query", json=item["payload"])
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        
        print(f"\nResponse Code: {resp.status_code} (Client Roundtrip: {elapsed} ms)")
        resp_data = resp.json()
        print(f"Status:        {resp_data.get('status')}")
        print(f"Answer:        {resp_data.get('answer')}")
        print(f"Sources Count: {len(resp_data.get('sources', []))}")
        
        for idx, src in enumerate(resp_data.get("sources", []), 1):
            score_str = f"{src.get('score'):.4f}" if src.get("score") is not None else "N/A"
            print(f"   [{idx}] {src.get('source')} (Score: {score_str}) - ID: {src.get('chunk_id')}")
            
        if not captured_sample:
            captured_sample = {
                "sample_request": item["payload"],
                "sample_response": resp_data,
                "curl_command": (
                    f"curl -X POST http://localhost:8000/query \\\n"
                    f"  -H \"Content-Type: application/json\" \\\n"
                    f"  -d '{json.dumps(item['payload'])}'"
                )
            }

    # ------------------------------------------------------------------------
    # STEP 3: OUT-OF-DOMAIN / GUARDRAIL REFUSAL TEST
    # ------------------------------------------------------------------------
    print_banner("Step 3: Testing Hallucination Guardrail Refusal on Out-Of-Domain Query")
    
    ood_payload = {
        "question": "What is the secret recipe for Martian space cheese fondue?",
        "k": 4
    }
    print(f"Request Payload:\n{json.dumps(ood_payload, indent=2)}")
    ood_resp = client.post("/query", json=ood_payload)
    print(f"\nResponse Code: {ood_resp.status_code}")
    print(json.dumps(ood_resp.json(), indent=2))

    # ------------------------------------------------------------------------
    # STEP 4: INPUT VALIDATION & ERROR HANDLING TESTS
    # ------------------------------------------------------------------------
    print_banner("Step 4: Testing Input Validation & HTTP Error Codes")
    
    validation_cases = [
        ("Too short question (< 3 characters)", {"question": "Hi"}, 422),
        ("Whitespace only question", {"question": "      "}, 422),
        ("Missing question field in body", {"k": 4}, 422),
        ("Oversized question (> 1000 characters)", {"question": "Explain policy. " * 70}, 422),
        ("Invalid k parameter (k=0)", {"question": "What is Knovera?", "k": 0}, 422)
    ]

    for label, payload, expected_code in validation_cases:
        resp = client.post("/query", json=payload)
        status_match = "PASS" if resp.status_code == expected_code else "FAIL"
        print(f"• Case: {label:42} -> HTTP {resp.status_code} [{status_match}]")
        print(f"  Response Body: {resp.json().get('detail')}")

    # ------------------------------------------------------------------------
    # STEP 5: SAVE SAMPLE ARTIFACTS
    # ------------------------------------------------------------------------
    print_banner("Step 5: Exporting Sample Request & Response Artifacts")

    os.makedirs(_root / "outputs", exist_ok=True)
    
    sample_file_root = _root / "sample_api_query_response.json"
    with open(sample_file_root, "w", encoding="utf-8") as f:
        json.dump(captured_sample, f, indent=2)
    print(f"✔ Saved root sample file: {sample_file_root}")

    sample_file_outputs = _root / "outputs" / "api_query_sample.json"
    with open(sample_file_outputs, "w", encoding="utf-8") as f:
        json.dump(captured_sample, f, indent=2)
    print(f"✔ Saved outputs sample file: {sample_file_outputs}")

    # Display Curl Snippet for developer ease
    print("\n" + "-" * 80)
    print("Example cURL command for testing locally:")
    print("-" * 80)
    print(captured_sample["curl_command"])
    print("-" * 80)
    print("\n✔ Knovera Backend RAG API demonstration completed successfully.")


if __name__ == "__main__":
    run_api_demonstration()
