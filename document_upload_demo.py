"""
document_upload_demo.py

Interactive Demonstration and Verification Script for Document Upload & Indexing Endpoint (Assignment 3.45).
Demonstrates:
1. Pre-upload state: Querying an unindexed topic returns safe refusal.
2. Runtime Document Upload: POST /documents ingests, cleans, chunks, embeds, and indexes a new document.
3. Post-upload runtime searchability: Querying the new topic immediately returns a grounded answer with citations without server restart.
4. Error Handling: Rejection of unsupported file extensions (HTTP 415) and empty files (HTTP 400).
5. Exports structured JSON artifact to sample_document_upload_run.json and outputs/.
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient

# Ensure Knovera root is in sys.path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.rag_api import app
from src.api_config import get_config


def print_banner(title: str):
    """Print formatted terminal banner."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def run_document_upload_demonstration():
    """Run end-to-end document upload, indexing, and query verification."""
    print_banner("Knovera Document Upload & Runtime Indexing Demonstration")
    
    cfg = get_config()
    print(f"• Active Environment:   {cfg.app_env}")
    print(f"• Target Collection:    {cfg.collection_name}")
    print(f"• Upload Directory:     {cfg.upload_dir}")
    print(f"• Max Upload Size:      {cfg.max_upload_size_mb} MB")

    client = TestClient(app)

    # ------------------------------------------------------------------------
    # STEP 1: PRE-UPLOAD QUERY VERIFICATION (Absence of Knowledge)
    # ------------------------------------------------------------------------
    print_banner("Step 1: Querying Topic BEFORE Upload (Verifying Initial State)")
    
    target_question = "What is the Knovera travel reimbursement policy for domestic flight bookings and daily meal allowance?"
    query_payload = {
        "question": target_question,
        "k": 3,
        "score_threshold": 0.40
    }
    
    print(f"Question: \"{target_question}\"")
    t0 = time.perf_counter()
    resp_before = client.post("/query", json=query_payload)
    elapsed_before = round((time.perf_counter() - t0) * 1000, 2)
    
    data_before = resp_before.json()
    print(f"Response Status Code: {resp_before.status_code} ({elapsed_before} ms)")
    print(f"Query Status:         {data_before.get('status')}")
    print(f"Answer:               {data_before.get('answer')}")
    print(f"Sources Count:        {len(data_before.get('sources', []))}")

    # ------------------------------------------------------------------------
    # STEP 2: RUNTIME DOCUMENT UPLOAD & INDEXING
    # ------------------------------------------------------------------------
    print_banner("Step 2: Uploading & Indexing New Document via POST /documents")
    
    sample_policy_content = (
        "# Knovera Travel & Expense Reimbursement Policy\n\n"
        "## Domestic Flight Bookings\n"
        "Employees traveling on approved company business must book economy class domestic flights "
        "at least 14 days in advance through the corporate travel portal. Any flight upgrades require "
        "prior written authorization from department directors.\n\n"
        "## Daily Meal & Incidental Allowance\n"
        "A daily per-diem meal allowance of $75 is provided for domestic travel ($20 breakfast, $25 lunch, $30 dinner). "
        "Original itemized receipts must be submitted via the expense portal within 15 business days of travel completion.\n\n"
        "## Hotel & Lodging Standards\n"
        "Standard single-occupancy hotel rooms up to $200 per night (excluding taxes) are reimbursable in major metropolitan zones."
    )
    
    filename = "travel-reimbursement-policy.md"
    files = {
        "file": (filename, io.BytesIO(sample_policy_content.encode("utf-8")), "text/markdown")
    }
    
    print(f"Uploading file: '{filename}' ({len(sample_policy_content)} characters)...")
    t0 = time.perf_counter()
    upload_resp = client.post("/documents", files=files)
    elapsed_upload = round((time.perf_counter() - t0) * 1000, 2)
    
    print(f"\nUpload Status Code: {upload_resp.status_code} (Processing Time: {elapsed_upload} ms)")
    upload_data = upload_resp.json()
    print(json.dumps(upload_data, indent=2))

    # ------------------------------------------------------------------------
    # STEP 3: POST-UPLOAD RUNTIME SEARCHABILITY CONFIRMATION
    # ------------------------------------------------------------------------
    print_banner("Step 3: Querying Topic AFTER Upload (Runtime Searchability Proof)")
    
    print(f"Executing identical question query without restarting the server...")
    t0 = time.perf_counter()
    resp_after = client.post("/query", json=query_payload)
    elapsed_after = round((time.perf_counter() - t0) * 1000, 2)
    
    data_after = resp_after.json()
    print(f"\nResponse Status Code: {resp_after.status_code} ({elapsed_after} ms)")
    print(f"Query Status:         {data_after.get('status')}")
    print(f"Answer:               {data_after.get('answer')}")
    print(f"Sources Count:        {len(data_after.get('sources', []))}")
    
    for idx, src in enumerate(data_after.get("sources", []), 1):
        print(f"   [{idx}] Source: {src.get('source')} | Section: {src.get('section')} | Score: {src.get('score')}")

    # ------------------------------------------------------------------------
    # STEP 4: ERROR HANDLING VERIFICATION
    # ------------------------------------------------------------------------
    print_banner("Step 4: Testing Upload Error Handling & Status Codes")
    
    error_cases = [
        ("Unsupported File Extension (.exe)", "malware.exe", b"binary content", 415),
        ("Unsupported File Extension (.docx)", "report.docx", b"docx binary content", 415),
        ("Empty File (0 bytes)", "empty-policy.md", b"", 400),
    ]

    for label, test_fname, test_bytes, expected_code in error_cases:
        err_files = {"file": (test_fname, io.BytesIO(test_bytes), "application/octet-stream")}
        err_resp = client.post("/documents", files=err_files)
        match_str = "PASS" if err_resp.status_code == expected_code else "FAIL"
        print(f"• Case: {label:42} -> HTTP {err_resp.status_code} [{match_str}]")
        print(f"  Detail: {err_resp.json().get('detail')}")

    # ------------------------------------------------------------------------
    # STEP 5: EXPORT SAMPLE RUN ARTIFACTS
    # ------------------------------------------------------------------------
    print_banner("Step 5: Exporting Sample Run Artifacts")

    sample_run_payload = {
        "upload_request": {
            "endpoint": "POST /documents",
            "filename": filename,
            "content_type": "text/markdown",
            "file_size_bytes": len(sample_policy_content.encode("utf-8")),
            "curl_command": f"curl -X POST http://localhost:8000/documents -F \"file=@{filename}\""
        },
        "upload_response": upload_data,
        "runtime_searchability_test": {
            "query_before_upload": {
                "question": target_question,
                "status": data_before.get("status"),
                "answer": data_before.get("answer"),
                "sources_count": len(data_before.get("sources", []))
            },
            "query_after_upload": {
                "question": target_question,
                "status": data_after.get("status"),
                "answer": data_after.get("answer"),
                "sources": data_after.get("sources", []),
                "latency_ms": data_after.get("latency_ms")
            }
        }
    }

    root_sample_path = _root / "sample_document_upload_run.json"
    with open(root_sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_run_payload, f, indent=2)
    print(f"✔ Saved root sample file: {root_sample_path}")

    outputs_sample_path = _root / "outputs" / "sample_upload_query_result.json"
    os.makedirs(_root / "outputs", exist_ok=True)
    with open(outputs_sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_run_payload, f, indent=2)
    print(f"✔ Saved outputs sample file: {outputs_sample_path}")

    print("\n✔ Document upload & runtime indexing demonstration completed successfully.")


if __name__ == "__main__":
    run_document_upload_demonstration()
