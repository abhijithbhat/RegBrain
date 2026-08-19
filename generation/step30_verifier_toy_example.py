"""
Step 30 – Verifier Toy Example
================================
Demonstrates a simple fuzzy text-overlap verifier that checks whether
a claim is genuinely supported by its cited chunk.

Uses three checks (a claim must pass ALL to be verified):
  1. difflib.SequenceMatcher  – overall textual similarity  (≥ 0.70)
  2. Keyword overlap ratio    – general token coverage       (≥ 0.70)
  3. Number match check       – every number in the claim
                                must appear in the chunk text
"""

import re
from difflib import SequenceMatcher

# ── Hardcoded chunk ─────────────────────────────────────────────────
CHUNK = {
    "clause_id": "3.2",
    "clause_text": "NBFCs shall maintain a minimum CRAR of 15 percent.",
}

# ── Two fake claims to verify ──────────────────────────────────────
CLAIMS = [
    {
        "label": "A",
        "text": "NBFCs must maintain a minimum CRAR of 15 percent.",
        "cited_clause_id": "3.2",
        "expected": "PASS",
    },
    {
        "label": "B",
        "text": "NBFCs must maintain a minimum CRAR of 20 percent.",
        "cited_clause_id": "3.2",
        "expected": "FAIL",
    },
]

SEQ_THRESHOLD = 0.70
KW_THRESHOLD  = 0.70


# ── Scoring helpers ─────────────────────────────────────────────────
def fuzzy_score(claim_text: str, chunk_text: str) -> float:
    """SequenceMatcher similarity ratio in [0, 1]."""
    return SequenceMatcher(
        None,
        claim_text.lower(),
        chunk_text.lower(),
    ).ratio()


def keyword_overlap_score(claim_text: str, chunk_text: str) -> float:
    """
    Tokenise both texts, return |intersection| / |claim_tokens|.
    """
    tokenise = lambda t: set(re.findall(r"\b[\w.%]+\b", t.lower()))
    claim_tokens = tokenise(claim_text)
    chunk_tokens = tokenise(chunk_text)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


def numbers_match(claim_text: str, chunk_text: str) -> bool:
    """
    Extract every number (int / float / percentage) from the claim
    and verify that every one of them also appears in the chunk text.
    This is the critical guard against hallucinated figures.
    """
    claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim_text))
    chunk_nums = set(re.findall(r"\d+(?:\.\d+)?", chunk_text))
    if not claim_nums:
        return True          # no numbers to verify
    return claim_nums.issubset(chunk_nums)


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("Verifier Toy Example")
    print(f"  SequenceMatcher threshold : {SEQ_THRESHOLD:.2f}")
    print(f"  Keyword-overlap threshold : {KW_THRESHOLD:.2f}")
    print(f"  Number-match              : all claim numbers ⊆ chunk")
    print("=" * 64)
    print()
    print(f"Chunk  [{CHUNK['clause_id']}]: {CHUNK['clause_text']}")
    print("-" * 64)

    for claim in CLAIMS:
        if claim["cited_clause_id"] != CHUNK["clause_id"]:
            print(f"\n  Claim {claim['label']}: cited clause "
                  f"'{claim['cited_clause_id']}' not found → FAIL")
            continue

        seq     = fuzzy_score(claim["text"], CHUNK["clause_text"])
        kw      = keyword_overlap_score(claim["text"], CHUNK["clause_text"])
        nums_ok = numbers_match(claim["text"], CHUNK["clause_text"])

        verdict = (
            "PASS"
            if (seq >= SEQ_THRESHOLD and kw >= KW_THRESHOLD and nums_ok)
            else "FAIL"
        )
        match_icon = "✅" if verdict == claim["expected"] else "❌"

        print(f"\n  Claim {claim['label']}: \"{claim['text']}\"")
        print(f"    SequenceMatcher  : {seq:.4f}  {'✓' if seq >= SEQ_THRESHOLD else '✗'}")
        print(f"    Keyword overlap  : {kw:.4f}  {'✓' if kw >= KW_THRESHOLD else '✗'}")
        print(f"    Numbers match    : {nums_ok}    {'✓' if nums_ok else '✗'}")
        print(f"    Verdict          : {verdict}  (expected {claim['expected']})  {match_icon}")
