"""Check what clause_ids the 3 missed queries actually need."""

import pickle

BM25_PATH = "retrieval/bm25_index.pkl"

with open(BM25_PATH, "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]

CASES = [
    ("Q5: CRR minimum", ["crr", "cash reserve ratio"], ["150MD"]),
    ("Q8: Exposure limits single borrower", ["exposure limit", "single borrower"], []),
    ("Q9: SLR requirement", ["slr", "statutory liquidity ratio"], ["150MD"]),
]

for label, keywords, preferred_docs in CASES:
    print("=" * 90)
    print(f"  {label}   keywords={keywords}   preferred_docs={preferred_docs}")
    print("=" * 90)

    for i, chunk in enumerate(chunks):
        text = chunk.get("clause_text", "").lower()
        doc_id = chunk.get("doc_id", "")
        clause_id = chunk.get("clause_id", "")

        # If preferred docs specified, only look in those
        if preferred_docs and doc_id not in preferred_docs:
            continue

        for kw in keywords:
            if kw in text:
                preview = chunk.get("clause_text", "")[:250].replace("\n", " ")
                print(f"  chunk[{i}]  doc={doc_id}  clause_id={clause_id}")
                print(f"    → {preview}\n")
                break
    print()
