import urllib.request
import json
import sys

urls = [
    ("Frontend Root", "http://localhost:3001/"),
    ("Frontend User Login", "http://localhost:3001/login"),
    ("Frontend Admin Login", "http://localhost:3001/admin/login"),
    ("Frontend Chatbot", "http://localhost:3001/chat"),
    ("Frontend Admin Console", "http://localhost:3001/admin"),
    ("Backend Health", "http://127.0.0.1:8000/api/health"),
    ("Backend Dashboard Stats", "http://127.0.0.1:8000/api/dashboard/stats"),
    ("Backend Guardrails", "http://127.0.0.1:8000/api/guardrails"),
    ("Backend Conversations", "http://127.0.0.1:8000/api/conversations"),
    ("Backend Logs", "http://127.0.0.1:8000/api/logs"),
    ("Backend Chunks", "http://127.0.0.1:8000/api/chunks"),
    ("Backend Documents", "http://127.0.0.1:8000/api/documents"),
]

all_passed = True
print("=== VERIFYING KNOVERA ENDPOINTS & DATABASE LAYER ===")
for name, url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KnoveraVerifier/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            code = resp.getcode()
            if code in [200, 201]:
                print(f"[PASS] {code} - {name} ({url})")
            else:
                print(f"[WARN] {code} - {name} ({url})")
                all_passed = False
    except Exception as e:
        print(f"[FAIL] {name} ({url}) - Error: {e}")
        all_passed = False

if all_passed:
    print("\nALL 12 ENDPOINTS PASSED SUCCESSFULLY!")
else:
    print("\nSOME ENDPOINTS HAD ISSUES!")
    sys.exit(1)
