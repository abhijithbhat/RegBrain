"""Unified search test: dense (Qdrant) + sparse (BM25) side by side."""

import os
import pickle
import sys
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "regbrain"
BM25_PATH = "retrieval/bm25_index.pkl"
TOP_K = 5

QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def print_results(title, results):
    """Print a formatted results table."""
    print(f"\n{'─' * 130}")
    print(f"  {title}")
    print(f"{'─' * 130}")
    print(f" {'#':>2}  {'Score':>8}  {'Doc ID':<15}  {'Clause':<15}  {'Category':<18}  Text (first 150 chars)")
    print(f" {'─'*2}  {'─'*8}  {'─'*15}  {'─'*15}  {'─'*18}  {'─'*60}")

    for i, (score, chunk) in enumerate(results, start=1):
        text_preview = chunk.get("clause_text", "")[:150].replace("\n", " ")
        print(
            f" {i:>2}  {score:>8.4f}  "
            f"{chunk.get('doc_id', '?'):<15}  "
            f"{chunk.get('clause_id', '?'):<15}  "
            f"{chunk.get('category', '?'):<18}  "
            f"{text_preview}"
        )


def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieval/test_search.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"')

    # ── Load model ─────────────────────────────────────────────────────
    model = SentenceTransformer(MODEL_NAME)
    query_vector = model.encode(query).tolist()

    # ── Dense search (Qdrant) ──────────────────────────────────────────
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    qdrant_results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
    )

    dense_results = [
        (hit.score, hit.payload if hit.payload is not None else {})
        for hit in qdrant_results.points
    ]

    # ── BM25 search ────────────────────────────────────────────────────
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    bm25_results = [
        (scores[idx], chunks[idx])
        for idx in top_indices
    ]

    # ── Print both ─────────────────────────────────────────────────────
    print_results("DENSE RESULTS (semantic similarity)", dense_results)
    print_results("BM25 RESULTS (keyword match)", bm25_results)
    print()


if __name__ == "__main__":
    main()
