"""
step20_reranker_toy_example.py
Cross-encoder reranking demo with BAAI/bge-reranker-base.

Scores three hardcoded candidate passages against a single query and
prints them sorted by relevance score (highest first).
"""

from sentence_transformers import CrossEncoder


def main() -> None:
    # ── Query ─────────────────────────────────────────────────────
    query = "What is the capital adequacy requirement for NBFCs?"

    # ── Candidate passages ────────────────────────────────────────
    candidates = [
        (
            "NBFC capital adequacy",
            "Every non-banking financial company shall maintain a minimum "
            "capital ratio consisting of Tier I and Tier II capital which "
            "shall not be less than 15 percent of its aggregate risk-weighted "
            "assets on balance sheet and risk-adjusted value of off-balance "
            "sheet items."
        ),
        (
            "NBFC KYC norms",
            "NBFCs are required to follow Know Your Customer guidelines "
            "issued by the Reserve Bank, including customer identification "
            "procedures, maintenance of records, and reporting of suspicious "
            "transactions to the Financial Intelligence Unit."
        ),
        (
            "Bank holiday closures",
            "All scheduled commercial banks shall remain closed on the "
            "second and fourth Saturday of every month, as well as on "
            "national holidays declared by the central government."
        ),
    ]

    # ── Load cross-encoder ────────────────────────────────────────
    print("Loading cross-encoder model …")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # ── Score each (query, passage) pair ──────────────────────────
    pairs = [(query, text) for _, text in candidates]
    scores = reranker.predict(pairs)

    # ── Combine & sort ────────────────────────────────────────────
    scored = [
        (score, label, text)
        for (label, text), score in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Print ─────────────────────────────────────────────────────
    print(f'\nQuery: "{query}"\n')
    print(f"{'#':>2}  {'Score':>8}  {'Label':<25}  Text")
    print(f"{'─'*2}  {'─'*8}  {'─'*25}  {'─'*60}")

    for i, (score, label, text) in enumerate(scored, start=1):
        preview = text[:100].replace("\n", " ")
        print(f"{i:>2}  {score:>8.4f}  {label:<25}  {preview} …")


if __name__ == "__main__":
    main()
