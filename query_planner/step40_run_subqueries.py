"""
Step 40 – Run Sub-queries Through the Full Pipeline
======================================================
Takes the sub-queries produced by step 39's decomposition of
"What are the KYC requirements and capital adequacy norms for NBFCs?"
and runs each one through the existing answer_query() pipeline
(retrieve → generate → verify → finalize) completely unchanged.

Hardcoded sub-queries (from step 39 output):
  1. KYC requirements for NBFCs as specified in RBI circulars
  2. Capital adequacy norms for NBFCs as specified in RBI master directions
"""

import json
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv                        # noqa: E402
from generation.answer_query import answer_query      # noqa: E402

load_dotenv()  # reads .env from project root

# ── Hardcoded sub-queries from step 39 ──────────────────────────────
ORIGINAL_QUERY = "What are the KYC requirements and capital adequacy norms for NBFCs?"

SUB_QUERIES = [
    "KYC requirements for NBFCs as specified in RBI circulars",
    "Capital adequacy norms for NBFCs as specified in RBI master directions",
]


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("Step 40 – Run Sub-queries Through answer_query()")
    print("=" * 64)
    print(f"\n  Original query: \"{ORIGINAL_QUERY}\"")
    print(f"  Sub-queries   : {len(SUB_QUERIES)}")

    for i, sq in enumerate(SUB_QUERIES, 1):
        print(f"\n{'─' * 64}")
        print(f"  Sub-query {i}/{len(SUB_QUERIES)}: \"{sq}\"")
        print(f"{'─' * 64}\n")

        result = answer_query(sq)

        status     = result.get("status", "unknown")
        answer     = result.get("answer", result.get("reason", "—"))
        citations  = result.get("citations", [])
        confidence = result.get("confidence")

        print(f"  Status     : {status}")
        print(f"  Confidence : {confidence}")
        print(f"  Citations  : {len(citations)} claim(s)")
        print()

        # Print each citation compactly
        for j, cite in enumerate(citations, 1):
            claim    = cite.get("claim", cite.get("text", ""))
            clause   = cite.get("clause_id", "?")
            verdict  = cite.get("verdict", "?")
            print(f"    [{j}] clause {clause} ({verdict}): {claim}")

        print(f"\n  Answer:\n")
        # Indent the answer for readability
        # pyrefly: ignore [missing-attribute]
        for line in answer.splitlines():
            print(f"    {line}")

        # Respect Groq rate limits between calls
        if i < len(SUB_QUERIES):
            wait = 12
            print(f"\n  ⏳ Sleeping {wait}s to respect Groq rate limits …")
            time.sleep(wait)

    print(f"\n{'=' * 64}")
    print("✅ All sub-queries processed.")
