import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# Step 1: Initial question
payload1 = {
    "session_id": "test_conv_followup_123",
    "message": "Tell me about refund policy",
    "history": []
}

print("--- Question 1: 'Tell me about refund policy' ---")
r1 = requests.post(f"{BASE_URL}/api/chat", json=payload1)
data1 = r1.json()
print("Status:", data1.get("status"))
print("Answer snippet:", data1.get("message")[:120])
print("Sources count:", len(data1.get("sources", [])))

# Step 2: Follow-up question referencing 'this'
payload2 = {
    "session_id": "test_conv_followup_123",
    "message": "can you tell more about this",
    "history": [
        {"role": "user", "content": "Tell me about refund policy"},
        {"role": "assistant", "content": data1.get("message")}
    ]
}

print("\n--- Question 2: 'can you tell more about this' ---")
r2 = requests.post(f"{BASE_URL}/api/chat", json=payload2)
data2 = r2.json()
print("Status:", data2.get("status"))
print("Answer full:\n", data2.get("message"))
print("Sources count:", len(data2.get("sources", [])))
if data2.get("sources"):
    print("Top Source Doc:", data2["sources"][0].get("doc_title") or data2["sources"][0].get("source"))
    print("Top Source Score:", data2["sources"][0].get("score"))
