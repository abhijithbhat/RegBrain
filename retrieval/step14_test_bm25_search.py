"""Step 14: Test BM25 keyword search against the saved index."""

import pickle
import sys
import numpy as np

INDEX_PATH = "retrieval/bm25_index.pkl"
TOP_K = 5


def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieval/step14_test_bm25_search.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')

    # --- Load index ---
    print(f"Loading BM25 index from {INDEX_PATH} ...")
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    # --- Tokenize query (same as indexing) ---
    query_tokens = query.lower().split()

    # --- Score all documents ---
    scores = bm25.get_scores(query_tokens)

    # --- Get top K ---
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    # --- Print results ---
    print(f"\nTop {TOP_K} results:\n")
    print(f"{'#':>2}  {'Score':>8}  {'Doc ID':<15}  {'Clause':<15}  {'Category':<18}  Text (first 150 chars)")
    print("─" * 130)

    for i, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]
        text_preview = chunk["clause_text"][:150].replace("\n", " ")
        print(
            f"{i:>2}  {scores[idx]:>8.4f}  "
            f"{chunk.get('doc_id', '?'):<15}  "
            f"{chunk.get('clause_id', '?'):<15}  "
            f"{chunk.get('category', '?'):<18}  "
            f"{text_preview}"
        )

    print()


if __name__ == "__main__":
    main()
