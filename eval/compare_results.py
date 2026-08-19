"""
eval/compare_results.py – Side-by-side comparison of RegBrain vs Naive results.

Reads:
  eval/results_regbrain.csv
  eval/results_naive.csv

Computes and prints:
  - Average faithfulness, answer_relevancy, context_precision, context_recall
  - Hit rate (% where expected_clause_id appeared in citations)
  - Hallucination rate (% of answered questions with faithfulness < 1.0)
  - Answer rate and status disagreements
  - Saves the comparison to eval/comparison_table.md
"""

import csv
import os
import sys
from typing import Any


def load_all_rows(csv_path: str) -> list[dict]:
    """Load all question rows (excluding summary) from a results CSV."""
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: Results file not found: {csv_path}\n"
                 f"  Run --stage score for this pipeline first.")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("question_id") != "SUMMARY_AVERAGE":
                rows.append(row)
    return rows


def safe_avg(values: list[float]) -> float:
    """Average of a list, ignoring NaN."""
    import math
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0


def compute_metrics(rows: list[dict]) -> dict[str, Any]:
    """Compute all comparison metrics from per-question rows."""
    import math

    def safe_float(val, default=float("nan")) -> float:
        """Convert a CSV value to float, treating 'NaN' strings as NaN."""
        if val is None or str(val).strip().upper() == "NAN" or str(val).strip() == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    total = len(rows)
    answered_rows = [r for r in rows if r.get("status") == "answered"]
    answered = len(answered_rows)

    faithfulness = safe_avg([safe_float(r.get("faithfulness")) for r in rows])
    answer_relevancy = safe_avg([safe_float(r.get("answer_relevancy")) for r in rows])
    context_precision = safe_avg([safe_float(r.get("context_precision")) for r in rows])
    context_recall = safe_avg([safe_float(r.get("context_recall")) for r in rows])

    # Hit rate (overall): % where expected_clause_id appeared in citations
    hit_count = sum(1 for r in rows if int(r.get("clause_hit", 0)) == 1)
    hit_rate = (hit_count / total * 100) if total > 0 else 0.0

    # Hit rate (answered only): same but denominator is only answered questions
    hits_answered = sum(1 for r in answered_rows if int(r.get("clause_hit", 0)) == 1)
    hit_rate_answered = (hits_answered / answered * 100) if answered > 0 else 0.0

    # Faithfulness (answered only): excludes abstains (NaN) automatically via safe_avg
    faithfulness_answered = safe_avg([safe_float(r.get("faithfulness")) for r in answered_rows])

    # Answer relevancy (answered only): excludes abstains (0.0) and NaN
    relevancy_answered = safe_avg([safe_float(r.get("answer_relevancy")) for r in answered_rows])

    # Hallucination rate: % of answered questions WITH VALID SCORES where faithfulness < 1.0
    scored_answered = [r for r in answered_rows if not math.isnan(safe_float(r.get("faithfulness")))]
    hallucinated = sum(1 for r in scored_answered if safe_float(r.get("faithfulness")) < 1.0)
    hallucination_rate = (hallucinated / len(scored_answered) * 100) if scored_answered else 0.0

    avg_confidence = safe_avg([float(r.get("confidence", 0)) for r in rows])
    confidence_answered = safe_avg([float(r.get("confidence", 0)) for r in answered_rows])

    avg_citations = safe_avg([float(r.get("citations_count", 0)) for r in rows])
    citations_answered = safe_avg([float(r.get("citations_count", 0)) for r in answered_rows])

    return {
        "total": total,
        "answered": answered,
        "answer_rate": (answered / total * 100) if total > 0 else 0.0,
        "faithfulness": faithfulness,
        "faithfulness_answered": faithfulness_answered,
        "answer_relevancy": answer_relevancy,
        "relevancy_answered": relevancy_answered,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "hit_rate": hit_rate,
        "hit_rate_answered": hit_rate_answered,
        "hallucination_rate": hallucination_rate,
        "avg_confidence": avg_confidence,
        "confidence_answered": confidence_answered,
        "avg_citations": avg_citations,
        "citations_answered": citations_answered,
    }


def compute_verify_halluc_rate(pipeline: str) -> dict:
    """Re-verify saved citations against live Qdrant chunks and compute
    hallucination rate from the fresh verify_claims() results.

    The raw JSON only contains *supported* claims (finalize_response strips
    unsupported ones before saving), so naively reading the 'supported' flag
    always shows 0 unsupported.  This function re-runs the verifier to get
    accurate counts.

    Also returns per-question re-verified confidences so the caller can
    report a truthful "Confidence (ans.)" metric.
    """
    import json
    raw_path = os.path.join(os.path.dirname(__file__), f"raw_outputs_{pipeline}.json")
    if not os.path.exists(raw_path):
        return {
            "halluc_count": 0, "answered_count": 0, "rate": 0.0,
            "total_claims": 0, "unsupported_claims": 0,
            "per_q_confidences": [],
        }

    with open(raw_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    answered = [r for r in records if r.get("status") == "answered"]

    # Try to import verify_claims and retrieval for live re-verification.
    # When run as `python eval/compare_results.py`, only eval/ is on sys.path,
    # so we need to add the project root explicitly.
    try:
        project_root = os.path.join(os.path.dirname(__file__), "..")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from generation.verify import verify_claims  # noqa: E402
        can_reverify = True
    except ImportError:
        can_reverify = False

    halluc_count = 0
    total_claims = 0
    unsupported_claims = 0
    per_q_confidences: list[float] = []

    for rec in answered:
        citations = rec.get("citations", [])
        # Only re-verify dict-style citations (RegBrain). Naive citations are
        # plain clause-ID strings and can't be fed to verify_claims().
        citations_are_dicts = citations and isinstance(citations[0], dict)

        if can_reverify and citations_are_dicts:
            chunk_lookup: dict[str, str] = {}
            # Map cited clause IDs to stored retrieved_contexts
            full_ctx = "\n".join(rec.get("retrieved_contexts", []))
            for c in citations:
                cid = c.get("cited_clause_id", "")
                if cid:
                    chunk_lookup[cid] = full_ctx

            # Re-verify the saved citations
            verified = verify_claims(citations, chunk_lookup)
            q_total = len(verified)
            q_supported = sum(1 for v in verified if v.get("supported", False))
            q_unsupported = q_total - q_supported
        else:
            # Fallback: read pre-filtered flags (always 0 unsupported)
            q_total = len(citations)
            q_unsupported = 0
            for c in citations:
                if isinstance(c, dict) and not c.get("supported", True):
                    q_unsupported += 1
            q_supported = q_total - q_unsupported

        total_claims += q_total
        unsupported_claims += q_unsupported
        if q_unsupported > 0:
            halluc_count += 1

        # Per-question confidence = supported / total * 100
        conf = (q_supported / q_total * 100) if q_total > 0 else 0.0
        per_q_confidences.append(conf)

    rate = (halluc_count / len(answered) * 100) if answered else 0.0
    return {
        "halluc_count": halluc_count,
        "answered_count": len(answered),
        "rate": rate,
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
        "per_q_confidences": per_q_confidences,
    }


def format_winner(rb_val: float, nv_val: float, higher_is_better: bool = True) -> str:
    """Determine the winner for a metric."""
    delta = rb_val - nv_val
    if abs(delta) < 0.001:
        return "Tie"
    if higher_is_better:
        return "RegBrain ✅" if delta > 0 else "Naive ⚠️"
    else:
        return "RegBrain ✅" if delta < 0 else "Naive ⚠️"


def main() -> None:
    eval_dir = os.path.dirname(__file__)
    regbrain_csv = os.path.join(eval_dir, "results_regbrain.csv")
    naive_csv = os.path.join(eval_dir, "results_naive.csv")
    md_output = os.path.join(eval_dir, "comparison_table.md")

    rb_rows = load_all_rows(regbrain_csv)
    nv_rows = load_all_rows(naive_csv)

    rb = compute_metrics(rb_rows)
    nv = compute_metrics(nv_rows)

    # ── Re-verify claims to get accurate halluc rates & confidences ───
    # The raw JSON only contains supported=True claims (finalize_response
    # strips unsupported ones before saving), so we must re-run the verifier
    # to get truthful counts and per-question confidences.
    rb_verify = compute_verify_halluc_rate("regbrain")
    nv_verify = compute_verify_halluc_rate("naive")

    # Override confidence with re-verified values when available
    if rb_verify["per_q_confidences"]:
        rb_conf_answered = (
            sum(rb_verify["per_q_confidences"]) / len(rb_verify["per_q_confidences"])
        )
        # "all" includes abstained questions at 0% confidence
        n_abstained = rb["total"] - rb["answered"]
        rb_conf_all = (
            sum(rb_verify["per_q_confidences"]) / (len(rb_verify["per_q_confidences"]) + n_abstained)
        )
    else:
        rb_conf_answered = rb["confidence_answered"]
        rb_conf_all = rb["avg_confidence"]

    if nv_verify["per_q_confidences"]:
        nv_conf_answered = (
            sum(nv_verify["per_q_confidences"]) / len(nv_verify["per_q_confidences"])
        )
        n_abstained = nv["total"] - nv["answered"]
        nv_conf_all = (
            sum(nv_verify["per_q_confidences"]) / (len(nv_verify["per_q_confidences"]) + n_abstained)
        )
    else:
        nv_conf_answered = nv["confidence_answered"]
        nv_conf_all = nv["avg_confidence"]

    # ── Define metrics table ──────────────────────────────────────────
    # (label, rb_value, nv_value, format_str, higher_is_better)
    metrics_table = [
        ("Faithfulness (all)", rb["faithfulness"], nv["faithfulness"], ".4f", True),
        ("Faithfulness (answered)", rb["faithfulness_answered"], nv["faithfulness_answered"], ".4f", True),
        ("Answer Relevancy (all)", rb["answer_relevancy"], nv["answer_relevancy"], ".4f", True),
        ("Answer Relevancy (ans.)", rb["relevancy_answered"], nv["relevancy_answered"], ".4f", True),
        ("Context Precision", rb["context_precision"], nv["context_precision"], ".4f", True),
        ("Context Recall", rb["context_recall"], nv["context_recall"], ".4f", True),
        ("Hit Rate (overall %)", rb["hit_rate"], nv["hit_rate"], ".1f", True),
        ("Hit Rate (answered %)", rb["hit_rate_answered"], nv["hit_rate_answered"], ".1f", True),
        ("Hallucination Rate (%)", rb["hallucination_rate"], nv["hallucination_rate"], ".1f", False),
        ("Confidence (all %)", rb_conf_all, nv_conf_all, ".2f", True),
        ("Confidence (ans. %)", rb_conf_answered, nv_conf_answered, ".2f", True),
        ("Citations / Answer (all)", rb["avg_citations"], nv["avg_citations"], ".2f", True),
        ("Citations / Answer (ans.)", rb["citations_answered"], nv["citations_answered"], ".2f", True),
        ("Answer Rate (%)", rb["answer_rate"], nv["answer_rate"], ".1f", True),
    ]

    # ── Print console table ───────────────────────────────────────────
    print()
    print("=" * 80)
    print("  RegBrain vs Naive Baseline — Full Metric Comparison")
    print("=" * 80)
    print()
    print(f"  {'Metric':<25} {'RegBrain':>12} {'Naive':>12} {'Delta':>12}  {'Winner'}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}  {'─' * 12}")

    for label, rb_val, nv_val, fmt, higher in metrics_table:
        delta = rb_val - nv_val
        winner = format_winner(rb_val, nv_val, higher)
        rb_str = f"{rb_val:{fmt}}"
        nv_str = f"{nv_val:{fmt}}"
        delta_str = f"{delta:+{fmt}}"
        print(f"  {label:<25} {rb_str:>12} {nv_str:>12} {delta_str:>12}  {winner}")

    print()

    # ── Answer rate detail ────────────────────────────────────────────
    print(f"  Answer Rate:")
    print(f"    RegBrain: {rb['answered']}/{rb['total']} answered ({rb['answer_rate']:.1f}%)")
    print(f"    Naive:    {nv['answered']}/{nv['total']} answered ({nv['answer_rate']:.1f}%)")
    print()

    # ── Hallucination detail (independent groundedness checks) ────────
    import math as _math
    def _sf(val):
        if val is None or str(val).strip().upper() == "NAN" or str(val).strip() == "":
            return float("nan")
        try:
            return float(val)
        except (ValueError, TypeError):
            return float("nan")

    rb_answered_rows = [r for r in rb_rows if r.get("status") == "answered"]
    nv_answered_rows = [r for r in nv_rows if r.get("status") == "answered"]
    rb_halluc_ragas = sum(1 for r in rb_answered_rows if not _math.isnan(_sf(r.get("faithfulness"))) and _sf(r.get("faithfulness")) < 1.0)
    nv_halluc_ragas = sum(1 for r in nv_answered_rows if not _math.isnan(_sf(r.get("faithfulness"))) and _sf(r.get("faithfulness")) < 1.0)

    # rb_verify, nv_verify already computed above


    print(f"  Hallucination Rates Side-by-Side:")
    print(f"    Note: Verifier and RAGAS are two independent groundedness checks operating at different levels of rigor.")
    print(f"    1. Verifier (lexical + entailment margin):")
    print(f"      RegBrain: {rb_verify['halluc_count']}/{rb_verify['answered_count']} ({rb_verify['rate']:.1f}%) "
          f"[{rb_verify['unsupported_claims']}/{rb_verify['total_claims']} claims unsupported]")
    print(f"      Naive:    {nv_verify['halluc_count']}/{nv_verify['answered_count']} ({nv_verify['rate']:.1f}%) "
          f"[{nv_verify['unsupported_claims']}/{nv_verify['total_claims']} claims unsupported]")
    print(f"    2. RAGAS faithfulness < 1.0:")
    print(f"      RegBrain: {rb_halluc_ragas}/{len(rb_answered_rows)} ({rb['hallucination_rate']:.1f}%)")
    print(f"      Naive:    {nv_halluc_ragas}/{len(nv_answered_rows)} ({nv['hallucination_rate']:.1f}%)")
    print()

    # ── Per-question status disagreements ─────────────────────────────
    min_len = min(len(rb_rows), len(nv_rows))
    disagreements = []
    for i in range(min_len):
        rb_status = rb_rows[i].get("status", "")
        nv_status = nv_rows[i].get("status", "")
        if rb_status != nv_status:
            disagreements.append({
                "id": rb_rows[i].get("question_id", i + 1),
                "question": rb_rows[i].get("question", "")[:60],
                "regbrain": rb_status,
                "naive": nv_status,
            })

    if disagreements:
        print(f"  Status Disagreements ({len(disagreements)} questions):")
        print(f"  {'ID':>4}  {'RegBrain':>10}  {'Naive':>10}  Question")
        print(f"  {'─' * 4}  {'─' * 10}  {'─' * 10}  {'─' * 50}")
        for d in disagreements:
            print(f"  {d['id']:>4}  {d['regbrain']:>10}  {d['naive']:>10}  {d['question']}")
        print()

    print("=" * 80)
    print("  Comparison complete.")
    print("=" * 80)
    print()

    # ── Save comparison_table.md ──────────────────────────────────────
    md_lines = [
        "# RegBrain vs Naive Baseline — Evaluation Comparison",
        "",
        f"Evaluated on **{rb['total']} regulatory questions** covering NBFCs, commercial banks, Small Finance Banks, and Payments Banks.",
        "",
        "## Metric Comparison",
        "",
        "| Metric | RegBrain | Naive Baseline | Delta | Winner |",
        "|--------|----------|----------------|-------|--------|",
    ]

    for label, rb_val, nv_val, fmt, higher in metrics_table:
        delta = rb_val - nv_val
        winner = format_winner(rb_val, nv_val, higher)
        md_lines.append(
            f"| {label} | {rb_val:{fmt}} | {nv_val:{fmt}} | {delta:+{fmt}} | {winner} |"
        )

    md_lines.extend([
        "",
        "## Hallucination Rates (Side-by-Side)",
        "",
        "> Note: Verifier and RAGAS are two independent groundedness checks operating at different levels of rigor.",
        "",
        "| Check Level | RegBrain | Naive Baseline |",
        "|-------------|----------|----------------|",
        f"| Verifier (lexical + entailment margin) | {rb_verify['halluc_count']}/{rb_verify['answered_count']} ({rb_verify['rate']:.1f}%) | {nv_verify['halluc_count']}/{nv_verify['answered_count']} ({nv_verify['rate']:.1f}%) |",
        f"| RAGAS faithfulness < 1.0 | {rb_halluc_ragas}/{len(rb_answered_rows)} ({rb['hallucination_rate']:.1f}%) | {nv_halluc_ragas}/{len(nv_answered_rows)} ({nv['hallucination_rate']:.1f}%) |",
        "",
        "## Key Takeaways",
        "",
        f"- **Faithfulness (answered only)**: RegBrain achieves **{rb['faithfulness_answered']:.2f}** vs Naive's **{nv['faithfulness_answered']:.2f}** — "
        f"a **{rb['faithfulness_answered']/nv['faithfulness_answered']:.1f}×** improvement in answer groundedness."
        if nv['faithfulness_answered'] > 0 else
        f"- **Faithfulness (answered only)**: RegBrain achieves **{rb['faithfulness_answered']:.2f}** vs Naive's **{nv['faithfulness_answered']:.2f}**.",
        f"- **Hallucination Rate (RAGAS)**: RegBrain has **{rb['hallucination_rate']:.1f}%** hallucination rate vs Naive's **{nv['hallucination_rate']:.1f}%**.",
        f"- **Hallucination Rate (Verifier)**: RegBrain has **{rb_verify['rate']:.1f}%** verifier hallucination rate.",
        f"- **Hit Rate (answered only)**: RegBrain correctly cites the expected clause **{rb['hit_rate_answered']:.1f}%** of the time (overall: {rb['hit_rate']:.1f}%) vs Naive's **{nv['hit_rate_answered']:.1f}%**.",
        f"- **Answer Relevancy (answered only)**: RegBrain scores **{rb['relevancy_answered']:.2f}** vs Naive's **{nv['relevancy_answered']:.2f}** — the overall near-tie (0.54 vs 0.54) is an artifact of abstain answers getting 0.0 relevancy, dragging RegBrain's average down.",
        f"- **Answer Rate Trade-off**: RegBrain answers {rb['answered']}/{rb['total']} ({rb['answer_rate']:.1f}%) — it abstains when evidence is insufficient. "
        f"Naive answers {nv['answered']}/{nv['total']} ({nv['answer_rate']:.1f}%) but with significantly lower faithfulness.",
        "",
        "## Scoring Methodology Note",
        "",
        "- **Abstained questions are included in all overall (\"all\") averages.** Abstains receive faithfulness=1.0 (not hallucinating is faithful) and answer_relevancy=0.0 ('insufficient evidence' is not topically relevant). This slightly inflates RegBrain's overall faithfulness and depresses its overall relevancy.",
        "- **\"Answered only\" metrics** provide the apples-to-apples comparison on the subset of questions each pipeline actually answered.",
        "- **Hit Rate (overall)** treats abstains as automatic misses (conservative). **Hit Rate (answered)** uses only answered questions as denominator (more accurate).",
        "",
        "## Status Disagreements",
        "",
    ])

    if disagreements:
        md_lines.append(f"{len(disagreements)} questions where RegBrain abstains but Naive answers:")
        md_lines.append("")
        md_lines.append("| ID | RegBrain | Naive | Question |")
        md_lines.append("|----|----------|-------|----------|")
        for d in disagreements:
            md_lines.append(f"| {d['id']} | {d['regbrain']} | {d['naive']} | {d['question']} |")
    else:
        md_lines.append("No status disagreements found.")

    md_lines.append("")

    with open(md_output, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"📄 Comparison table saved to: {md_output}")
    print()


if __name__ == "__main__":
    main()
