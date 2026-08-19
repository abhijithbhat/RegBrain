"""
Step 39 – Query Decomposition Toy Example
============================================
Takes a hardcoded multi-hop query and asks Groq (openai/gpt-oss-120b)
to decompose it into 2-3 independent sub-queries suitable for separate
retrieve() calls.

Query: "What are the KYC requirements and capital adequacy norms for NBFCs?"

The LLM is forced to return JSON:
  {"sub_queries": ["...", "..."]}
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────────
load_dotenv()  # reads .env from project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "openai/gpt-oss-120b"

# ── Hardcoded multi-hop query ───────────────────────────────────────
QUERY = "What are the KYC requirements and capital adequacy norms for NBFCs?"

# ── Decomposition prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a query planner for a regulatory Q&A system backed by a
vector store of RBI circulars and master directions.

The user's query requires more than one independent lookup to answer
fully. Decompose it into 2-3 independent sub-queries, where each
sub-query can be answered with a single retrieve() call against the
vector store.

Rules:
  • Each sub-query must be self-contained (no pronouns like "it" or
    "the same").
  • Each sub-query should target a distinct fact or time-period.
  • Return EXACTLY a JSON object — no markdown, no extra text:
    {"sub_queries": ["...", "..."]}
"""


def decompose_query(query: str) -> dict:
    """Ask Groq to split *query* into independent sub-queries; return parsed JSON."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

    if response.status_code != 200:
        return {
            "sub_queries": [],
            "error": f"Groq API error [{response.status_code}]: {response.text}",
        }

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "sub_queries": [],
            "error": f"Unparseable LLM output: {raw}",
        }


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("Step 39 – Query Decomposition Toy Example")
    print(f"  Model : {MODEL}")
    print("=" * 64)

    print(f"\n  Original query: \"{QUERY}\"")
    print()

    result = decompose_query(QUERY)

    if "error" in result:
        print(f"  ⚠ Error: {result['error']}")
        sys.exit(1)

    sub_queries = result.get("sub_queries", [])

    print(f"  Decomposed into {len(sub_queries)} sub-queries:\n")
    for i, sq in enumerate(sub_queries, 1):
        print(f"    {i}. {sq}")

    print("\n" + "=" * 64)
    print("✅ Decomposition complete.")
