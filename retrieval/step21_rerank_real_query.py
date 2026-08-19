"""
step21_rerank_real_query.py
RRF fusion → cross-encoder reranking pipeline.

Usage:
    python retrieval/step21_rerank_real_query.py "your query here"

Pipeline:
  1. Dense (Qdrant) top-20  +  BM25 top-20
  2. Reciprocal Rank Fusion (k=60) → fused top-20
  3. Cross-encoder reranking (bge-reranker-base) → final top-5
"""

import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
COLLECTION = "regbrain"
BM25_PATH = "retrieval/bm25_index.pkl"
RETRIEVE_K = 20          # per retriever
RRF_K = 60
RERANK_TOP = 5           # final results to show

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
        print('Usage: python retrieval/step21_rerank_real_query.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"')
    print(f"Retrieve: {RETRIEVE_K}/list  |  RRF k={RRF_K}  |  Rerank top: {RERANK_TOP}\n")

    # ── 1. Load models & index ────────────────────────────────────
    print("Loading embedding model …")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Loading cross-encoder reranker …")
    reranker = CrossEncoder(RERANKER_MODEL)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]

    # ── 2. Retrieve ───────────────────────────────────────────────
    print("Running dense + BM25 retrieval …")
    dense_hits = dense_search(query, embed_model, client, RETRIEVE_K)
    bm25_hits = bm25_search(query, bm25, chunks, RETRIEVE_K)

    dense_keys = [key for key, _ in dense_hits]
    bm25_keys = [key for key, _ in bm25_hits]

    # Merge payload metadata by key
    payload_map: dict[str, dict] = {}
    for key, payload in dense_hits + bm25_hits:
        if key not in payload_map:
            payload_map[key] = payload

    # ── 3. RRF fusion ─────────────────────────────────────────────
    rrf_scores = reciprocal_rank_fusion([dense_keys, bm25_keys], k=RRF_K)
    fused_ranking = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[
        :RETRIEVE_K
    ]

    # Map each key to its 1-indexed position in the fused list
    fused_rank_of = {key: pos for pos, key in enumerate(fused_ranking, start=1)}

    # ── 4. Rerank with cross-encoder ──────────────────────────────
    print("Reranking with cross-encoder …\n")
    pairs = []
    valid_keys = []
    for key in fused_ranking:
        text = payload_map.get(key, {}).get("clause_text", "")
        if text:
            pairs.append([query, text])
            valid_keys.append(key)

    reranker_scores = reranker.predict(pairs)  # type: ignore[arg-type]

    scored = list(zip(valid_keys, reranker_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # ── 5. Print top results ──────────────────────────────────────
    top_results = scored[:RERANK_TOP]

    dense_set = set(dense_keys)
    bm25_set = set(bm25_keys)

    header = (
        f"{'#':>2}  {'Reranker':>9}  {'Fused ↕':>7}  "
        f"{'Doc ID':<15}  {'Clause ID':<20}  {'Source':<14}  Text"
    )
    print(header)
    print("─" * len(header) + "─" * 60)

    for new_pos, (key, score) in enumerate(top_results, start=1):
        payload = payload_map.get(key, {})
        doc_id = payload.get("doc_id", "?")
        clause_id = payload.get("clause_id", "?")
        text = payload.get("clause_text", "")[:150].replace("\n", " ")

        old_pos = fused_rank_of.get(key, "?")
        if isinstance(old_pos, int):
            delta = old_pos - new_pos
            if delta > 0:
                movement = f"#{old_pos} ▲{delta}"
            elif delta < 0:
                movement = f"#{old_pos} ▼{abs(delta)}"
            else:
                movement = f"#{old_pos}  ─"
        else:
            movement = "?"

        sources = []
        if key in dense_set:
            sources.append("dense")
        if key in bm25_set:
            sources.append("bm25")

        print(
            f"{new_pos:>2}  {score:>9.4f}  {movement:>7}  "
            f"{doc_id:<15}  {clause_id:<20}  {'+'.join(sources):<14}  "
            f"{text}"
        )

    print()


if __name__ == "__main__":
    main()
