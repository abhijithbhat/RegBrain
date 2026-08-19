"""
eval/sanity_check_verifier.py

Sanity-check the final 2-gate citation verifier rule against the 71 claims
in raw_outputs_regbrain.json (no Groq API calls).
"""

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation.verify import verify_claims
from retrieval.retrieve import retrieve


def run_sanity_check():
    raw_path = os.path.join(os.path.dirname(__file__), "raw_outputs_regbrain.json")
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        records: List[Dict] = json.load(f)

    print("=" * 110)
    print("  SANITY CHECK: FINAL CITATION VERIFIER RULE ON SAVED CLAIMS (raw_outputs_regbrain.json)")
    print("=" * 110)

    answered = [r for r in records if r.get("status") == "answered"]
    print(f"Total Records: {len(records)} | Answered Records: {len(answered)}\n")

    total_claims = 0
    total_supported = 0
    total_lexical_pass = 0
    total_nli_pass = 0

    table_rows = []

    for rec in answered:
        qid = rec["question_id"]
        q_text = rec["question"]
        citations = rec.get("citations", [])

        # Re-retrieve chunks from Qdrant (vector DB only, 0 LLM calls)
        chunks = retrieve(q_text)
        chunk_lookup: Dict[str, str] = {}
        for c in chunks:
            cid = c["clause_id"]
            if cid in chunk_lookup:
                chunk_lookup[cid] += "\n" + c["clause_text"]
            else:
                chunk_lookup[cid] = c["clause_text"]

        # Run final verifier logic
        verified = verify_claims(citations, chunk_lookup)

        q_claims = len(verified)
        q_lexical = sum(1 for c in verified if c.get("lexical_pass", False))
        q_nli = sum(1 for c in verified if c.get("nli_pass", False))
        q_supp = sum(1 for c in verified if c.get("supported", False))

        total_claims += q_claims
        total_lexical_pass += q_lexical
        total_nli_pass += q_nli
        total_supported += q_supp

        table_rows.append({
            "qid": qid,
            "question": q_text[:50] + "..." if len(q_text) > 50 else q_text,
            "total_claims": q_claims,
            "lexical_pass": q_lexical,
            "nli_pass": q_nli,
            "supported": q_supp,
            "unsupported": q_claims - q_supp,
        })

    # Print Table
    print(f"{'QID':>4}  {'Question':<53}  {'Claims':>6}  {'LexPass':>7}  {'NLIPass':>7}  {'Supp':>5}  {'Unsupp':>6}")
    print(f"{'─'*4}  {'─'*53}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*6}")

    for r in table_rows:
        print(f"Q{r['qid']:>2d}  {r['question']:<53}  {r['total_claims']:>6d}  "
              f"{r['lexical_pass']:>7d}  {r['nli_pass']:>7d}  {r['supported']:>5d}  {r['unsupported']:>6d}")

    print(f"{'─'*4}  {'─'*53}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*6}")
    print(f"TOTAL  {'All Answered Questions':<53}  {total_claims:>6d}  "
          f"{total_lexical_pass:>7d}  {total_nli_pass:>7d}  {total_supported:>5d}  {total_claims - total_supported:>6d}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    run_sanity_check()
