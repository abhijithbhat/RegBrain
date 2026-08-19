import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from retrieval.retrieve import retrieve

q1 = "What are the KYC requirements for NBFCs?"
chunks = retrieve(q1)

context_for_prompt = [
    {
        "citation_id": f"C{index:02d}",
        "document_id": c.get("doc_id", ""),
        "clause_label": c.get("clause_id", ""),
        "clause_text": c["clause_text"][:1200] + ("..." if len(c["clause_text"]) > 1200 else "")
    }
    for index, c in enumerate(chunks[:5], start=1)
]

user_prompt = (
    f"Context chunks:\n{json.dumps(context_for_prompt, indent=2)}\n\n"
    f"Question: {q1}"
)

from generation.generate import SYSTEM_PROMPT, MODEL, GROQ_URL, GROQ_API_KEY

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    "temperature": 0,
    "response_format": {"type": "json_object"},
}

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

print(f"Payload size in characters: {len(json.dumps(payload))}", flush=True)
resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
print(f"Status Code: {resp.status_code}", flush=True)
print(f"Headers: {dict(resp.headers)}", flush=True)
print(f"Response: {resp.text[:300]}", flush=True)
