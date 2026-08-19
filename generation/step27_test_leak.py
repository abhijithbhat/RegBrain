"""
Step 27 – Leak test: deliberately unrelated chunk.

Same grounded-generation setup as step26, but the provided chunk is
completely irrelevant to the question.  A well-grounded model should
refuse to answer from its own knowledge and flag the gap.
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

# ── Hardcoded inputs (deliberately wrong chunk) ─────────────────
QUERY = "What is the capital adequacy requirement for NBFCs?"

CHUNK = {
    "clause_id": "9.9",
    "clause_text": (
        "Banks must display their working hours at all branch entrances."
    ),
}

# ── System prompt: identical grounding instructions as step26 ───
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

print("Sending LEAK-TEST request to Groq …")
print(f"  Query : {QUERY}")
print(f"  Chunk : {CHUNK['clause_id']} – {CHUNK['clause_text']}")

response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

if response.status_code != 200:
    print(f"Request failed [{response.status_code}]")
    print(response.text)
    sys.exit(1)

raw_text = response.json()["choices"][0]["message"]["content"]

# ── Print raw response ──────────────────────────────────────────
print("\n── Raw response ──────────────────────────────────────")
print(raw_text)
print("─────────────────────────────────────────────────────")

# ── Quick verdict ───────────────────────────────────────────────
try:
    parsed = json.loads(raw_text)
    answer = parsed.get("answer", "").lower()
    leaked = not any(
        phrase in answer
        for phrase in ["not contain", "no information", "does not", "insufficient", "not available", "cannot"]
    )
    if leaked:
        print("\n⚠️  POSSIBLE LEAK – model may have used its own knowledge.")
    else:
        print("\n✅ Model correctly refused – grounding held.")
except json.JSONDecodeError:
    print("\n⚠️  Response was not valid JSON.")
