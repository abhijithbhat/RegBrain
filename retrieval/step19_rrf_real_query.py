"""
step19_rrf_real_query.py
Reciprocal Rank Fusion over real dense (Qdrant) + BM25 results.

Usage:
    python retrieval/step19_rrf_real_query.py "your query here"
"""

import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "regbrain"
BM25_PATH = "retrieval/bm25_index.pkl"
TOP_K = 20
RRF_K = 60

QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# ── Reciprocal Rank Fusion ────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Return {doc_key: rrf_score} across all supplied ranked lists."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_0, doc_key in enumerate(ranked_list):
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank_0 + 1)
    return scores


# ── Search helpers ────────────────────────────────────────────────────
def dense_search(query: str, model, client, top_k: int):
    """Return [(chunk_key, payload), ...] from Qdrant."""
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
    )
    out = []
    for hit in results.points:
        payload = hit.payload if hit.payload is not None else {}
        key = f"{payload.get('doc_id', '?')}::{payload.get('clause_id', '?')}"
        out.append((key, payload))
    return out


def bm25_search(query: str, bm25, chunks, top_k: int):
    """Return [(chunk_key, payload), ...] from BM25 index."""
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in top_indices:
        chunk = chunks[idx]
        key = f"{chunk.get('doc_id', '?')}::{chunk.get('clause_id', '?')}"
        out.append((key, chunk))
    return out


# ── Main ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieval/step19_rrf_real_query.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"')
    print(f"Top-K per list: {TOP_K}   RRF k: {RRF_K}\n")

    # ── Load resources ────────────────────────────────────────────
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]

    # ── Retrieve both ranked lists ────────────────────────────────
    dense_hits = dense_search(query, model, client, TOP_K)
    bm25_hits = bm25_search(query, bm25, chunks, TOP_K)

    dense_keys = [key for key, _ in dense_hits]
    bm25_keys = [key for key, _ in bm25_hits]

    # ── Merge payload metadata by key ─────────────────────────────
    payload_map: dict[str, dict] = {}
    for key, payload in dense_hits + bm25_hits:
        if key not in payload_map:
            payload_map[key] = payload

    # ── Track provenance ──────────────────────────────────────────
    dense_set = set(dense_keys)
    bm25_set = set(bm25_keys)

    # ── Fuse ──────────────────────────────────────────────────────
    rrf_scores = reciprocal_rank_fusion([dense_keys, bm25_keys], k=RRF_K)
    fused_ranking = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[
        :TOP_K
    ]

    # ── Print ─────────────────────────────────────────────────────
    print(f"{'#':>3}  {'RRF Score':>10}  {'Doc ID':<15}  {'Clause ID':<20}  Source(s)")
    print(f"{'─'*3}  {'─'*10}  {'─'*15}  {'─'*20}  {'─'*18}")

    for pos, key in enumerate(fused_ranking, start=1):
        payload = payload_map.get(key, {})
        doc_id = payload.get("doc_id", "?")
        clause_id = payload.get("clause_id", "?")

        sources = []
        if key in dense_set:
            sources.append("dense")
        if key in bm25_set:
            sources.append("bm25")

        print(
            f"{pos:>3}  {rrf_scores[key]:>10.6f}  "
            f"{doc_id:<15}  {clause_id:<20}  "
            f"{' + '.join(sources)}"
        )

    # ── Summary stats ─────────────────────────────────────────────
    both = dense_set & bm25_set
    dense_only = dense_set - bm25_set
    bm25_only = bm25_set - dense_set
    print(f"\nOverlap: {len(both)} in both | {len(dense_only)} dense-only | {len(bm25_only)} bm25-only")


if __name__ == "__main__":
    main()
