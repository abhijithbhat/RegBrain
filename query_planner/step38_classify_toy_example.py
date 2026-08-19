"""
Step 38 – Query Classification Toy Example
=============================================
Classifies two hardcoded queries as single-hop or multi-hop by asking
Groq (openai/gpt-oss-120b) whether answering them fully requires more
than one independent lookup.

Queries:
  A: "What are the KYC requirements for NBFCs?"          → single-hop
  B: "Has the NBFC lending limit changed since 2023?"    → multi-hop

The LLM is forced to return JSON:
  {"needs_decomposition": true/false, "reasoning": "..."}
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

# ── Hardcoded queries ───────────────────────────────────────────────
QUERIES = {
    "A": "What are the KYC requirements for NBFCs?",
    "B": "What are the KYC requirements and capital adequacy norms for NBFCs?",
}

# ── Classification prompt ───────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a query planner for a regulatory Q&A system backed by a
vector store of RBI circulars and master directions.

Given a user query, decide whether answering it fully requires:
  • a SINGLE retrieve() call (one topic, one time-frame, one lookup), OR
  • MULTIPLE independent lookups (e.g. comparing across time, combining
    facts from different regulations, or verifying a change over time).

Respond with ONLY a JSON object — no markdown, no explanation outside
the JSON:
{"needs_decomposition": true/false, "reasoning": "<one-sentence explanation>"}
"""


def classify_query(query: str) -> dict:
    """Ask Groq whether *query* needs decomposition; return parsed JSON."""
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
            "needs_decomposition": None,
            "reasoning": f"Groq API error [{response.status_code}]: {response.text}",
        }

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "needs_decomposition": None,
            "reasoning": f"Unparseable LLM output: {raw}",
        }


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("Step 38 – Query Classification Toy Example")
    print(f"  Model : {MODEL}")
    print("=" * 64)

    for label, query in QUERIES.items():
        print(f"\n── Query {label} ──────────────────────────────────────")
        print(f"  \"{query}\"")
        print()

        result = classify_query(query)

        needs = result.get("needs_decomposition")
        reasoning = result.get("reasoning", "")

        tag = (
            "MULTI-HOP  → decompose"
            if needs
            else "SINGLE-HOP → retrieve() once"
            if needs is False
            else "ERROR"
        )

        print(f"  needs_decomposition : {needs}")
        print(f"  reasoning           : {reasoning}")
        print(f"  classification      : {tag}")

    print("\n" + "=" * 64)
    print("✅ Classification complete.")
