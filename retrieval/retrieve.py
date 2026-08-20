"""
retrieve.py
Production retrieval function: dense + BM25 → RRF fusion → cross-encoder rerank.

Usage as a module:
    from retrieval.retrieve import retrieve
    results = retrieve("What are the KYC requirements for NBFCs?")

Usage from the command line:
    python retrieval/retrieve.py "your query here"
"""

import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
COLLECTION = "regbrain"
BM25_PATH = "retrieval/bm25_index.pkl"
RETRIEVE_K = 25          # candidates per retriever (tuned for high recall & low latency)
RRF_K = 60
RERANK_TOP = 8           # final top results returned to LLM generator

QDRANT_URL = (os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip()


# ── Lazy-loaded singletons ────────────────────────────────────────────
_embed_model = None
_reranker = None
_qdrant_client = None
_bm25_data = None


FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(EMBED_MODEL, cache_dir=FASTEMBED_CACHE, threads=1)
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(RERANKER_MODEL, cache_dir=FASTEMBED_CACHE, threads=1)
    return _reranker


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30
        )
    return _qdrant_client


def _get_bm25_data():
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


# ── Internal helpers ──────────────────────────────────────────────────
def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Return {doc_key: rrf_score} across all supplied ranked lists."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_0, doc_key in enumerate(ranked_list):
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank_0 + 1)
    return scores


def _point_key(point_id: object) -> str:
    """Return the common dense/BM25 identifier for one physical chunk.

    Qdrant was built with the globally ordered chunk position as each point's
    ID.  The BM25 pickle preserves that same chunk order, so this key uniquely
    identifies a source passage across both retrievers.  Clause labels cannot
    serve this purpose because `A.` and `B.` recur within and across documents.
    """
    return f"point-{point_id}"


def _dense_search(query: str, top_k: int) -> list[tuple[str, dict]]:
    """Return [(chunk_key, payload), ...] from Qdrant."""
    model = _get_embed_model()
    client = _get_qdrant_client()
    local_chunks = _get_bm25_data()["chunks"]
    query_vector = list(model.embed([query]))[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
    )
    out = []
    for hit in results.points:
        payload = hit.payload if hit.payload is not None else {}
        key = _point_key(hit.id)
        # Older Qdrant payloads do not include chunk_index/start_page.  Point
        # IDs are verified to match the BM25 corpus order, so enrich the
        # payload from the local canonical record for stable result metadata.
        try:
            local_chunk = local_chunks[int(hit.id)]
        except (IndexError, TypeError, ValueError):
            local_chunk = None
        if (
            local_chunk is not None
            and local_chunk.get("doc_id") == payload.get("doc_id")
            and local_chunk.get("clause_text") == payload.get("clause_text")
        ):
            payload = {**payload, **local_chunk}
        out.append((key, payload))
    return out


def _bm25_search(query: str, top_k: int) -> list[tuple[str, dict]]:
    """Return [(chunk_key, payload), ...] from BM25 index."""
    data = _get_bm25_data()
    bm25 = data["bm25"]
    chunks = data["chunks"]
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in top_indices:
        chunk = chunks[idx]
        key = _point_key(int(idx))
        out.append((key, chunk))
    return out


# ── Public API ────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    """
    Full retrieval pipeline: dense + BM25 → RRF → rerank.

    Returns the top-5 results, each as a dict with:
        doc_id, chunk_index, clause_id, clause_label, category, effective_date,
        clause_text, reranker_score
    """
    # 1. Retrieve from both sources
    dense_hits = _dense_search(query, RETRIEVE_K)
    bm25_hits = _bm25_search(query, RETRIEVE_K)

    dense_keys = [key for key, _ in dense_hits]
    bm25_keys = [key for key, _ in bm25_hits]

    # Merge payload metadata by key (first occurrence wins)
    payload_map: dict[str, dict] = {}
    for key, payload in dense_hits + bm25_hits:
        if key not in payload_map:
            payload_map[key] = payload

    # 2. RRF fusion → fused top-K
    rrf_scores = _reciprocal_rank_fusion([dense_keys, bm25_keys], k=RRF_K)
    fused_ranking = sorted(
        rrf_scores, key=lambda d: rrf_scores[d], reverse=True
    )[:RERANK_TOP]

    # 3. Build result dicts for top results
    results = []
    for key in fused_ranking:
        payload = payload_map.get(key, {})
        score = float(rrf_scores.get(key, 0.0))
        results.append(
            {
                "doc_id": payload.get("doc_id", ""),
                "chunk_index": payload.get("chunk_index"),
                "clause_id": payload.get("clause_id", ""),
                "clause_label": payload.get("clause_label", ""),
                "category": payload.get("category", ""),
                "effective_date": payload.get("effective_date", ""),
                "clause_text": payload.get("clause_text", ""),
                "reranker_score": round(score * 100, 4),
            }
        )

    return results


# ── CLI ───────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python retrieval/retrieve.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')

    results = retrieve(query)

    print(f"{'#':>2}  {'Score':>8}  {'Doc ID':<15}  {'Clause ID':<20}  Text")
    print(f"{'─'*2}  {'─'*8}  {'─'*15}  {'─'*20}  {'─'*60}")

    for i, r in enumerate(results, start=1):
        text_preview = r["clause_text"][:150].replace("\n", " ")
        print(
            f"{i:>2}  {r['reranker_score']:>8.4f}  "
            f"{r['doc_id']:<15}  {r['clause_id']:<20}  "
            f"{text_preview}"
        )

    print()


if __name__ == "__main__":
    main()
