"""
Step 26 – Grounded generation toy example.

Sends a hardcoded query + a single fake regulatory chunk to Groq
(openai/gpt-oss-120b) with strict instructions to answer ONLY from
the provided context.  The response is parsed as JSON and pretty-printed.
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Hardcoded inputs ────────────────────────────────────────────
QUERY = "What is the capital adequacy requirement for NBFCs?"

CHUNK = {
    "clause_id": "3.2",
    "clause_text": (
        "NBFCs shall maintain a minimum Capital to Risk-Weighted "
        "Assets Ratio (CRAR) of 15 percent."
    ),
}

# ── System prompt: grounded generation with citation ────────────
SYSTEM_PROMPT = """\
You are a regulatory compliance assistant.
Answer the user's question using ONLY the provided context chunks.
Do NOT use any prior knowledge.  If the context does not contain
enough information, say so explicitly.

Respond with valid JSON in exactly this shape (no markdown fences):
{
  "answer": "<your answer>",
  "claims": [
    {"text": "<atomic claim>", "cited_clause_id": "<clause_id from context>"}
  ]
}
"""

USER_PROMPT = f"""\
Context chunks:
{json.dumps([CHUNK], indent=2)}

Question: {QUERY}
"""

# ── Call Groq ───────────────────────────────────────────────────
payload = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    "temperature": 0,
}

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

print("Sending grounded-generation request to Groq …")

response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

if response.status_code != 200:
    print(f"Request failed [{response.status_code}]")
    print(response.text)
    sys.exit(1)

raw_text = response.json()["choices"][0]["message"]["content"]

# ── Parse and pretty-print ──────────────────────────────────────
try:
    parsed = json.loads(raw_text)
except json.JSONDecodeError:
    print("⚠️  Model did not return valid JSON. Raw output:")
    print(raw_text)
    sys.exit(1)

print("\n── Parsed JSON response ──────────────────────────────")
print(json.dumps(parsed, indent=2))
print("─────────────────────────────────────────────────────")
print("\n✅ Grounded generation succeeded.")
