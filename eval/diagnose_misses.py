"""Diagnose the 3 missed questions: is the chunk missing, or just not retrieved?"""

import pickle
import re

BM25_PATH = "retrieval/bm25_index.pkl"

# The 3 missed questions and what we're looking for
MISSED = [
    {
        "question": "Q5: What is the minimum CRR that banks must maintain?",
        "expected_clause_id": "A.",
        "keywords": ["CRR", "cash reserve ratio", "minimum"],
    },
    {
        "question": "Q8: What are the exposure limits for a single borrower?",
        "expected_clause_id": "D.",
        "keywords": ["exposure limit", "single borrower", "concentration"],
    },
    {
        "question": "Q9: What is the SLR requirement for banks?",
        "expected_clause_id": "A.",
        "keywords": ["SLR", "statutory liquidity ratio"],
    },
]

with open(BM25_PATH, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
print(f"Total chunks in index: {len(chunks)}\n")

for case in MISSED:
    print("=" * 100)
    print(f"  {case['question']}")
    print(f"  Expected clause_id: {case['expected_clause_id']}")
    print("=" * 100)

    # Search all chunks for keyword matches
    matches = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("clause_text", "").lower()
        clause_id = chunk.get("clause_id", "")
        doc_id = chunk.get("doc_id", "")

        for kw in case["keywords"]:
            if kw.lower() in text:
                matches.append((i, doc_id, clause_id, kw, chunk.get("clause_text", "")[:200]))
                break  # one match per chunk is enough

    if matches:
        print(f"\n  Found {len(matches)} chunk(s) containing keywords {case['keywords']}:\n")
        for idx, doc_id, clause_id, kw, preview in matches[:10]:
            preview_clean = preview.replace("\n", " ")
            print(f"    chunk[{idx}]  doc={doc_id}  clause={clause_id}  matched='{kw}'")
            print(f"      → {preview_clean}\n")
    else:
        print(f"\n  ⚠ NO chunks found containing any of {case['keywords']}")
        print(f"    → This is a CHUNKING problem: the content was never indexed.\n")

    print()
