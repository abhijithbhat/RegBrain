"""
test_retrieval.py
Evaluate retrieval quality: run each test question through retrieve()
and check whether the expected clause appears in the first five results.

Usage:
    python eval/test_retrieval.py
"""

import json
import sys
import os

# Ensure the project root is on sys.path so `retrieval.retrieve` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retrieve import retrieve  # noqa: E402

TEST_FILE = os.path.join(os.path.dirname(__file__), "test_questions.json")
EVALUATION_TOP_K = 5


def main() -> None:
    with open(TEST_FILE, "r") as f:
        test_cases = json.load(f)

    total = len(test_cases)
    hits = 0

    print(f"Running {total} test questions …\n")
    print("=" * 120)

    for idx, tc in enumerate(test_cases, start=1):
        question = tc["question"]
        expected = tc["expected_clause_id"]

        results = retrieve(question)

        # retrieve() intentionally returns seven candidates for downstream
        # generation.  This regression suite evaluates a strict top-five
        # window, so do not let ranks 6–7 inflate the reported metric.
        evaluated_results = results[:EVALUATION_TOP_K]
        top5_pairs = [(r["doc_id"], r["clause_id"]) for r in evaluated_results]
        top5_clause_ids = [r["clause_id"] for r in evaluated_results]

        found = expected in top5_clause_ids
        if found:
            hits += 1

        status = "✅ HIT" if found else "❌ MISS"

        print(f"\n  Q{idx}: {question}")
        print(f"  Expected clause_id: {expected}   →  {status}")
        print(f"  First {EVALUATION_TOP_K} results (of {len(results)} retrieved):")
        for rank, (doc_id, clause_id) in enumerate(top5_pairs, start=1):
            marker = " ◀" if clause_id == expected else ""
            print(f"    {rank}. {doc_id}  /  {clause_id}{marker}")

        print(f"{'─' * 120}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  SUMMARY:  {hits} / {total} questions found the expected clause in the top-5")
    pct = (hits / total * 100) if total else 0
    print(f"  Accuracy: {pct:.1f}%")
    print(f"{'=' * 120}\n")


if __name__ == "__main__":
    main()
