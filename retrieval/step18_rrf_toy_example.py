"""
step18_rrf_toy_example.py
Reciprocal Rank Fusion (RRF) – toy demonstration.

Two fake ranked lists are fused using: RRF_score(d) = Σ 1/(k + rank_i(d))
where k=60 and rank is 1-indexed.  A document absent from a list contributes 0.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Return {doc_id: rrf_score} across all supplied ranked lists."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank_0, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank_0 + 1)
    return scores


def main() -> None:
    # ── Two fake ranked lists ────────────────────────────────────
    dense_ranking = ["docA", "docB", "docC", "docD", "docE"]
    bm25_ranking  = ["docC", "docA", "docE", "docF", "docB"]

    k = 60

    # ── Compute RRF scores ───────────────────────────────────────
    rrf_scores = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=k)

    # ── Build a lookup: doc → 1-indexed rank in each list (None if absent)
    dense_rank = {doc: i + 1 for i, doc in enumerate(dense_ranking)}
    bm25_rank  = {doc: i + 1 for i, doc in enumerate(bm25_ranking)}

    # ── Print per-document breakdown ─────────────────────────────
    print(f"{'Doc':<8} {'Dense rank':>11} {'BM25 rank':>10} {'RRF score':>12}")
    print("-" * 45)

    all_docs = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
    for doc in all_docs:
        dr = dense_rank.get(doc)
        br = bm25_rank.get(doc)
        print(
            f"{doc:<8} "
            f"{(str(dr) if dr else '—'):>11} "
            f"{(str(br) if br else '—'):>10} "
            f"{rrf_scores[doc]:>12.6f}"
        )

    # ── Final fused ranking ──────────────────────────────────────
    print("\nFused ranking (highest RRF score first):")
    for position, doc in enumerate(all_docs, start=1):
        print(f"  {position}. {doc}  (score {rrf_scores[doc]:.6f})")


if __name__ == "__main__":
    main()
