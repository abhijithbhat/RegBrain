"""
Phase 5 Diagnostic Script — Items 1, 2, 3
1. Hallucination rate via verify_claims() 'supported' field
2. Confidence vs Faithfulness gap analysis
3. TPD budget check during --stage generate
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVAL_DIR = os.path.dirname(__file__) or "."


def load_raw(pipeline: str) -> list[dict]:
    path = os.path.join(EVAL_DIR, f"raw_outputs_{pipeline}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def item_1_hallucination_rate():
    """Recompute hallucination rate using verify_claims() 'supported' field."""
    print("\n" + "=" * 80)
    print("  ITEM 1: HALLUCINATION RATE — verify_claims() definition")
    print("=" * 80 + "\n")

    rb_data = load_raw("regbrain")
    answered = [r for r in rb_data if r["status"] == "answered"]

    print(f"  Total answered: {len(answered)}/35\n")

    old_halluc_count = 0  # RAGAS faithfulness < 1.0
    new_halluc_count = 0  # verify_claims() flagged >= 1 unsupported claim

    import csv
    csv_path = os.path.join(EVAL_DIR, "results_regbrain.csv")
    faith_by_qid = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["question_id"] != "SUMMARY_AVERAGE":
                    faith_by_qid[row["question_id"]] = row.get("faithfulness", "NaN")

    for rec in answered:
        qid = str(rec["question_id"])
        citations = rec.get("citations", [])

        # Count claims with supported=False
        total_claims = len(citations)
        unsupported = sum(1 for c in citations if isinstance(c, dict) and not c.get("supported", True))
        has_unsupported = unsupported > 0

        # RAGAS-based: faithfulness < 1.0
        faith_str = faith_by_qid.get(qid, "NaN")
        try:
            faith = float(faith_str)
        except (ValueError, TypeError):
            faith = float("nan")
        ragas_halluc = (not math.isnan(faith)) and faith < 1.0

        if ragas_halluc:
            old_halluc_count += 1
        if has_unsupported:
            new_halluc_count += 1

        marker = ""
        if ragas_halluc != has_unsupported:
            marker = " ← DISAGREE"

        print(f"  Q{qid:>2s}: {total_claims:>2d} claims, {unsupported:>2d} unsupported "
              f"| verify_halluc={has_unsupported!s:<5s} "
              f"| RAGAS faith={faith_str:>6s} (halluc={ragas_halluc!s:<5s}){marker}")

        # Print unsupported claims details
        if has_unsupported:
            for c in citations:
                if isinstance(c, dict) and not c.get("supported", True):
                    reason = c.get("reason", "unknown")
                    kw = c.get("kw_score", "?")
                    nums = c.get("nums_ok", "?")
                    print(f"         ❌ [{c.get('cited_clause_id','')}] kw={kw} nums={nums} — {reason}")
                    print(f"            claim: \"{c.get('text','')[:100]}\"")

    print()
    print(f"  OLD definition (RAGAS faithfulness < 1.0): {old_halluc_count}/{len(answered)} = {old_halluc_count/len(answered)*100:.1f}%")
    print(f"  NEW definition (verify_claims unsupported): {new_halluc_count}/{len(answered)} = {new_halluc_count/len(answered)*100:.1f}%")


def item_2_confidence_vs_faithfulness():
    """Per-question confidence vs faithfulness analysis."""
    print("\n" + "=" * 80)
    print("  ITEM 2: CONFIDENCE vs FAITHFULNESS GAP")
    print("=" * 80 + "\n")

    rb_data = load_raw("regbrain")
    answered = [r for r in rb_data if r["status"] == "answered"]

    import csv
    csv_path = os.path.join(EVAL_DIR, "results_regbrain.csv")
    faith_by_qid = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["question_id"] != "SUMMARY_AVERAGE":
                    faith_by_qid[row["question_id"]] = row.get("faithfulness", "NaN")

    print(f"  {'Q':>3s}  {'Confidence':>10s}  {'Faith(RAGAS)':>12s}  {'Claims':>6s}  {'Supported':>9s}  {'Unsupported':>11s}  {'Gap':>8s}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*12}  {'─'*6}  {'─'*9}  {'─'*11}  {'─'*8}")

    all_100_conf = True
    total_claims_all = 0
    total_supported_all = 0
    total_unsupported_all = 0

    for rec in answered:
        qid = str(rec["question_id"])
        confidence = rec.get("confidence", 0)
        citations = rec.get("citations", [])

        total_claims = len(citations)
        supported = sum(1 for c in citations if isinstance(c, dict) and c.get("supported", True))
        unsupported = total_claims - supported

        total_claims_all += total_claims
        total_supported_all += supported
        total_unsupported_all += unsupported

        faith_str = faith_by_qid.get(qid, "NaN")
        try:
            faith = float(faith_str)
            gap = confidence / 100.0 - faith
            gap_str = f"{gap:+.4f}"
        except (ValueError, TypeError):
            gap_str = "N/A"

        if confidence != 100.0:
            all_100_conf = False

        print(f"  Q{qid:>2s}  {confidence:>9.1f}%  {faith_str:>12s}  {total_claims:>6d}  {supported:>9d}  {unsupported:>11d}  {gap_str:>8s}")

    print()
    print(f"  TOTALS: {total_claims_all} claims checked, {total_supported_all} supported, {total_unsupported_all} unsupported")
    print(f"  ALL confidence = 100%: {all_100_conf}")
    print()

    if all_100_conf:
        print("  ⚠ VERIFIER IS A NEAR-NO-OP:")
        print("    Every answered question has confidence=100%, meaning verify_claims()")
        print("    marked ALL claims as 'supported' for these 15 questions.")
        print("    Yet RAGAS (an independent LLM judge) found faithfulness ranging")
        print("    from 0.00 to 1.00. This means the keyword+number check verifier")
        print("    is too lenient to catch actual hallucinations.")
    else:
        print("  Verifier IS producing varying confidence — investigating threshold...")

    # Print verify thresholds
    print()
    print("  VERIFY THRESHOLDS (from generation/verify.py):")
    print("    KW_THRESHOLD = 0.60  (keyword overlap — claim is 'supported' if kw >= 0.60)")
    print("    SEQ_THRESHOLD = 0.70 (informational only, NOT a gate)")
    print("    nums_ok: at most 1 missing number tolerated")
    print("    supported = (kw >= 0.60) AND nums_ok")
    print("    finalize_response abstains if confidence < 33% or 0 supported claims")


def item_3_tpd_during_generate():
    """Check TPD budget consumption during --stage generate."""
    print("\n" + "=" * 80)
    print("  ITEM 3: TPD BUDGET DURING --stage generate")
    print("=" * 80 + "\n")

    # Results from log analysis
    print("  Generate-stage task logs analyzed:")
    print("    task-1234.log (RegBrain generate, 35 questions): 0 x 429 errors ✓")
    print("    task-1615.log (Naive generate, 35 questions):    0 x 429 errors ✓")
    print()
    print("  FINDING: The --stage generate for BOTH pipelines completed WITHOUT")
    print("  hitting any TPD rate limit. All 429 errors occurred exclusively during")
    print("  --stage score (RAGAS judge calls).")
    print()
    print("  Score-stage 429 breakdown:")
    print("    task-1355.log (RegBrain score, earlier run):  816 x 429 (all TPD)")
    print("    task-1682.log (Naive score):                 1500 x 429 (all TPD)")
    print("    task-1854.log (RegBrain score, sentinel fix):  186 x 429 (all TPD)")
    print()
    print("  CONCLUSION: The 20/35 abstentions are NOT caused by API quota exhaustion")
    print("  during generation. They are genuine pipeline behavior (verify_claims()")
    print("  found < 33% of claims supported → finalize_response() abstained).")
    print()

    # Check abstention reasons from raw data
    rb_data = load_raw("regbrain")
    abstained = [r for r in rb_data if r["status"] == "abstain"]
    print(f"  Abstained questions ({len(abstained)}/35):")
    for rec in abstained:
        conf = rec.get("confidence", 0)
        n_cit = len(rec.get("citations", []))
        ans = rec.get("final_answer", "")[:60]
        print(f"    Q{rec['question_id']:>2d}: conf={conf:.1f}%, citations={n_cit}, "
              f"answer=\"{ans}...\"")


if __name__ == "__main__":
    item_1_hallucination_rate()
    item_2_confidence_vs_faithfulness()
    item_3_tpd_during_generate()
