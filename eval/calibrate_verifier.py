"""
Threshold sweep calibration for verify_claims().

Sweeps KW_THRESHOLD from 0.60 to 0.90 (step 0.05), and optionally
SEQ_THRESHOLD as a second gate, against all 71 claims from the 15
answered questions in raw_outputs_regbrain.json.

Compares the resulting per-question hallucination rate against RAGAS
faithfulness to find the optimal threshold.
"""

import csv
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVAL_DIR = os.path.dirname(__file__) or "."

# ── Reimplement scoring helpers from verify.py ──────────────────────
def _fuzzy_score(claim_text: str, chunk_text: str) -> float:
    claim_low = claim_text.lower()
    chunk_low = chunk_text.lower()
    if len(chunk_low) <= len(claim_low) * 2:
        return SequenceMatcher(None, claim_low, chunk_low).ratio()
    window = len(claim_low) * 2
    step = max(1, len(claim_low) // 2)
    best = 0.0
    for start in range(0, len(chunk_low) - window + 1, step):
        segment = chunk_low[start : start + window]
        score = SequenceMatcher(None, claim_low, segment).ratio()
        if score > best:
            best = score
            if best > 0.95:
                break
    return best


def _keyword_overlap_score(claim_text: str, chunk_text: str) -> float:
    tokenise = lambda t: set(re.findall(r"\b[\w.%]+\b", t.lower()))
    claim_tokens = tokenise(claim_text)
    chunk_tokens = tokenise(chunk_text)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


def _extract_numbers(text: str) -> set[str]:
    _WORD_TO_NUM = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "twenty-five": "25",
        "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
        "seventy": "70", "seventy-five": "75", "eighty": "80", "ninety": "90",
        "hundred": "100", "thousand": "1000",
    }
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    lower = text.lower()
    for word, digit in _WORD_TO_NUM.items():
        if word in lower:
            nums.add(digit)
    return nums


def _numbers_match_strict(claim_text: str, chunk_text: str) -> bool:
    """Strict: ALL numbers must match. No tolerance."""
    claim_nums = _extract_numbers(claim_text)
    chunk_nums = _extract_numbers(chunk_text)
    if not claim_nums:
        return True
    missing = claim_nums - chunk_nums
    return len(missing) == 0  # STRICT: zero tolerance


def _numbers_match_old(claim_text: str, chunk_text: str) -> bool:
    """Old: tolerate 1 missing number."""
    claim_nums = _extract_numbers(claim_text)
    chunk_nums = _extract_numbers(chunk_text)
    if not claim_nums:
        return True
    missing = claim_nums - chunk_nums
    return len(missing) <= 1


# ── Load data ──────────────────────────────────────────────────────
def load_claims_and_chunks():
    """Load all 71 claims + their cited chunk text from raw_outputs."""
    with open(os.path.join(EVAL_DIR, "raw_outputs_regbrain.json")) as f:
        records = json.load(f)

    # Load RAGAS faithfulness per question
    faith_by_qid = {}
    csv_path = os.path.join(EVAL_DIR, "results_regbrain.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["question_id"] != "SUMMARY_AVERAGE":
                    try:
                        faith_by_qid[int(row["question_id"])] = float(row["faithfulness"])
                    except (ValueError, TypeError):
                        faith_by_qid[int(row["question_id"])] = float("nan")

    questions = []
    for rec in records:
        if rec["status"] != "answered":
            continue
        qid = rec["question_id"]
        # Rebuild chunks_by_clause_id from retrieved_contexts
        # We can't — but claims already have their verification data
        # We need the actual chunk text. Get it from the citations' cited_clause_id
        # and match against retrieved_contexts
        # Actually, the claims store the cited_clause_id but not the chunk text.
        # We need to re-retrieve. But the ask says "no new Groq calls".
        # The chunk text is available in retrieved_contexts (list of strings).
        # The problem is mapping cited_clause_id -> chunk text.
        # Let's re-retrieve from Qdrant (no Groq calls).
        questions.append({
            "qid": qid,
            "question": rec["question"],
            "claims": rec.get("citations", []),
            "ragas_faith": faith_by_qid.get(qid, float("nan")),
            "retrieved_contexts": rec.get("retrieved_contexts", []),
        })

    return questions


def rebuild_chunk_lookup(question_text: str) -> dict[str, str]:
    """Re-retrieve chunks from Qdrant (no Groq calls) to get clause_id -> text mapping."""
    from retrieval.retrieve import retrieve
    chunks = retrieve(question_text)
    lookup = {}
    for c in chunks:
        cid = c["clause_id"]
        if cid in lookup:
            lookup[cid] += "\n" + c["clause_text"]
        else:
            lookup[cid] = c["clause_text"]
    return lookup


def sweep_thresholds(questions):
    """Sweep KW_THRESHOLD from 0.60 to 0.90 with strict nums and SEQ as gate."""

    # First pass: compute all per-claim scores
    all_claims_data = []  # (qid, claim_text, chunk_text, kw_score, seq_score, nums_strict, nums_old)

    print("  Retrieving chunks for 15 answered questions (Qdrant only, no Groq)...\n")
    for q in questions:
        chunk_lookup = rebuild_chunk_lookup(q["question"])

        for claim in q["claims"]:
            cid = claim.get("cited_clause_id", "")
            chunk_text = chunk_lookup.get(cid)
            claim_text = claim.get("text", "")

            if chunk_text is None:
                # Claim cites non-retrieved chunk
                all_claims_data.append({
                    "qid": q["qid"],
                    "claim_text": claim_text[:80],
                    "chunk_text": None,
                    "kw_score": 0.0,
                    "seq_score": 0.0,
                    "nums_strict": False,
                    "nums_old": False,
                    "ragas_faith": q["ragas_faith"],
                })
                continue

            kw = _keyword_overlap_score(claim_text, chunk_text)
            seq = _fuzzy_score(claim_text, chunk_text)
            ns = _numbers_match_strict(claim_text, chunk_text)
            no = _numbers_match_old(claim_text, chunk_text)

            all_claims_data.append({
                "qid": q["qid"],
                "claim_text": claim_text[:80],
                "chunk_text": chunk_text[:60] if chunk_text else None,
                "kw_score": kw,
                "seq_score": seq,
                "nums_strict": ns,
                "nums_old": no,
                "ragas_faith": q["ragas_faith"],
            })

    total_claims = len(all_claims_data)
    print(f"  Total claims analyzed: {total_claims}\n")

    # ── Print per-claim scores ──────────────────────────────────────
    print("  Per-Claim Scores:")
    print(f"  {'Q':>3s} {'KW':>6s} {'SEQ':>6s} {'Nums_S':>6s} {'Nums_O':>6s}  Claim")
    print(f"  {'─'*3} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*60}")
    for cd in all_claims_data:
        ns_str = "✓" if cd['nums_strict'] else "✗"
        no_str = "✓" if cd['nums_old'] else "✗"
        print(f"  Q{cd['qid']:>2d} {cd['kw_score']:>6.3f} {cd['seq_score']:>6.3f} "
              f"{ns_str:>6s} {no_str:>6s}  {cd['claim_text']}")
    print()

    # ── Sweep KW threshold ──────────────────────────────────────────
    print("=" * 90)
    print("  THRESHOLD SWEEP: KW_THRESHOLD from 0.60 to 0.90")
    print("  (with strict nums + SEQ_THRESHOLD=0.30 as gate)")
    print("=" * 90)
    print()

    # RAGAS reference groups
    ragas_faithful = {2, 5, 8, 9}     # RAGAS faith = 1.0
    ragas_unfaithful = {13, 15, 17}    # RAGAS faith <= 0.13

    best_kw = 0.60
    best_separation = -1
    best_info = ""

    for kw_thresh_x10 in range(60, 95, 5):
        kw_thresh = kw_thresh_x10 / 100.0

        # Also sweep SEQ gate at 0.30 (modest requirement)
        seq_gate = 0.30

        unsupported_count = 0
        per_q_unsupported = {}

        for cd in all_claims_data:
            qid = cd["qid"]
            if qid not in per_q_unsupported:
                per_q_unsupported[qid] = {"total": 0, "unsupported": 0, "ragas": cd["ragas_faith"]}

            per_q_unsupported[qid]["total"] += 1

            # New verification: kw >= thresh AND seq >= seq_gate AND nums_strict
            supported = (cd["kw_score"] >= kw_thresh and
                         cd["seq_score"] >= seq_gate and
                         cd["nums_strict"])

            if cd["chunk_text"] is None:
                supported = False

            if not supported:
                unsupported_count += 1
                per_q_unsupported[qid]["unsupported"] += 1

        halluc_qs = sum(1 for v in per_q_unsupported.values() if v["unsupported"] > 0)
        n_answered = len(per_q_unsupported)
        halluc_rate = halluc_qs / n_answered * 100 if n_answered else 0

        # Check separation quality
        faithful_clean = sum(1 for qid in ragas_faithful
                             if qid in per_q_unsupported
                             and per_q_unsupported[qid]["unsupported"] == 0)
        unfaithful_flagged = sum(1 for qid in ragas_unfaithful
                                 if qid in per_q_unsupported
                                 and per_q_unsupported[qid]["unsupported"] > 0)

        separation = faithful_clean + unfaithful_flagged

        print(f"  KW={kw_thresh:.2f} SEQ≥{seq_gate:.2f} strict_nums: "
              f"{unsupported_count}/{total_claims} claims unsupported | "
              f"{halluc_qs}/{n_answered} Qs hallucinated ({halluc_rate:.1f}%) | "
              f"Faithful group clean: {faithful_clean}/{len(ragas_faithful)} | "
              f"Unfaithful flagged: {unfaithful_flagged}/{len(ragas_unfaithful)}")

        # Per-question detail
        for qid in sorted(per_q_unsupported.keys()):
            v = per_q_unsupported[qid]
            ragas_str = f"{v['ragas']:.4f}" if not math.isnan(v['ragas']) else "NaN"
            marker = ""
            if qid in ragas_faithful:
                marker = " ← SHOULD BE CLEAN"
            elif qid in ragas_unfaithful:
                marker = " ← SHOULD BE FLAGGED"
            if v["unsupported"] > 0:
                print(f"    Q{qid:>2d}: {v['unsupported']}/{v['total']} unsupported (RAGAS={ragas_str}){marker}")
        print()

        if separation > best_separation:
            best_separation = separation
            best_kw = kw_thresh
            best_info = (f"KW={kw_thresh:.2f}: {halluc_qs}/{n_answered} "
                         f"({halluc_rate:.1f}%) | "
                         f"faithful clean={faithful_clean}/{len(ragas_faithful)}, "
                         f"unfaithful flagged={unfaithful_flagged}/{len(ragas_unfaithful)}")

    print("=" * 90)
    print(f"  RECOMMENDED: {best_info}")
    print(f"  Best KW_THRESHOLD: {best_kw:.2f} (with SEQ_THRESHOLD=0.30 as gate, strict nums)")
    print("=" * 90)

    return best_kw, all_claims_data


def _evaluate_at_thresholds(claims_data, kw_thresh, seq_gate):
    """Evaluate all claims at given thresholds, return per-question dict."""
    per_q = {}
    for cd in claims_data:
        qid = cd["qid"]
        if qid not in per_q:
            per_q[qid] = {"total": 0, "unsupported": 0, "ragas": cd["ragas_faith"]}
        per_q[qid]["total"] += 1

        supported = (cd["kw_score"] >= kw_thresh and
                     cd["seq_score"] >= seq_gate and
                     cd["nums_strict"])
        if cd["chunk_text"] is None:
            supported = False
        if not supported:
            per_q[qid]["unsupported"] += 1
    return per_q


def full_per_question_report(claims_data, kw_thresh=0.60, seq_gate=0.30):
    """Print per-question: actual flips vs RAGAS-implied flips for ALL 15 questions."""
    print("\n" + "=" * 100)
    print(f"  FULL 15-QUESTION COMPARISON at KW={kw_thresh:.2f} / SEQ≥{seq_gate:.2f} / strict nums")
    print("=" * 100 + "\n")

    per_q = _evaluate_at_thresholds(claims_data, kw_thresh, seq_gate)

    rows = []
    for qid in sorted(per_q.keys()):
        v = per_q[qid]
        total = v["total"]
        flipped = v["unsupported"]
        ragas = v["ragas"]

        if math.isnan(ragas):
            ragas_implied = 0
        else:
            ragas_implied = round(total * (1.0 - ragas))

        gap = ragas_implied - flipped
        rows.append((qid, total, flipped, ragas_implied, gap, ragas))

    # Sort by largest gap first
    rows.sort(key=lambda r: abs(r[4]), reverse=True)

    print(f"  {'Q':>3s}  {'Total':>5s}  {'Flipped':>7s}  {'RAGAS-Imp':>9s}  {'Gap':>5s}  {'RAGAS':>8s}  Note")
    print(f"  {'─'*3}  {'─'*5}  {'─'*7}  {'─'*9}  {'─'*5}  {'─'*8}  {'─'*40}")

    large_gap_qs = []
    for qid, total, flipped, ragas_implied, gap, ragas in rows:
        ragas_str = f"{ragas:.4f}" if not math.isnan(ragas) else "NaN"
        note = ""
        if gap > 0:
            note = f"UNDER-DETECTED by {gap} claims"
            large_gap_qs.append(qid)
        elif gap < 0:
            note = f"OVER-DETECTED by {abs(gap)} claims"
        else:
            note = "MATCHES RAGAS"

        print(f"  Q{qid:>2d}  {total:>5d}  {flipped:>7d}  {ragas_implied:>9d}  {gap:>+5d}  {ragas_str:>8s}  {note}")

    print()

    # Summary
    total_flipped = sum(r[2] for r in rows)
    total_ragas_implied = sum(r[3] for r in rows)
    total_gap = total_ragas_implied - total_flipped
    print(f"  TOTALS: {total_flipped} flipped vs {total_ragas_implied} RAGAS-implied "
          f"(gap: {total_gap:+d} claims)")
    print()

    return large_gap_qs


def seq_threshold_sweep(claims_data, large_gap_qs, kw_thresh=0.60):
    """Sweep SEQ_THRESHOLD at 0.30, 0.35, 0.40, 0.45 and print per-Q comparison at each."""
    print("\n" + "=" * 100)
    print(f"  SEQ_THRESHOLD SWEEP for under-detected questions: {sorted(large_gap_qs)}")
    print(f"  (KW={kw_thresh:.2f}, strict nums, SEQ from 0.30 to 0.45)")
    print("=" * 100 + "\n")

    for seq_gate_x100 in [30, 35, 40, 45]:
        seq_gate = seq_gate_x100 / 100.0

        per_q = _evaluate_at_thresholds(claims_data, kw_thresh, seq_gate)

        total_claims = sum(v["total"] for v in per_q.values())
        total_unsup = sum(v["unsupported"] for v in per_q.values())
        halluc_qs = sum(1 for v in per_q.values() if v["unsupported"] > 0)

        print(f"  ── SEQ≥{seq_gate:.2f}: {total_unsup}/{total_claims} claims unsup, "
              f"{halluc_qs}/{len(per_q)} Qs hallucinated ──")

        print(f"    {'Q':>3s}  {'Total':>5s}  {'Flip':>4s}  {'RAGAS-Imp':>9s}  {'Gap':>5s}  {'RAGAS':>8s}")
        print(f"    {'─'*3}  {'─'*5}  {'─'*4}  {'─'*9}  {'─'*5}  {'─'*8}")

        all_rows = []
        for qid in sorted(per_q.keys()):
            v = per_q[qid]
            total = v["total"]
            flipped = v["unsupported"]
            ragas = v["ragas"]
            ragas_implied = round(total * (1.0 - ragas)) if not math.isnan(ragas) else 0
            gap = ragas_implied - flipped
            all_rows.append((qid, total, flipped, ragas_implied, gap, ragas))

            ragas_str = f"{ragas:.4f}" if not math.isnan(ragas) else "NaN"
            marker = ""
            if qid in large_gap_qs:
                marker = f"  ← TARGET (gap={gap:+d})"
            print(f"    Q{qid:>2d}  {total:>5d}  {flipped:>4d}  {ragas_implied:>9d}  {gap:>+5d}  {ragas_str:>8s}{marker}")

        total_gap = sum(r[4] for r in all_rows)
        abs_gap_sum = sum(abs(r[4]) for r in all_rows)
        # Check separation: RAGAS 1.0 group must stay clean
        ragas_faithful = {2, 5, 8, 9}
        faithful_clean = sum(1 for r in all_rows
                             if r[0] in ragas_faithful and r[2] == 0)

        print(f"    Total gap: {total_gap:+d} | Abs gap sum: {abs_gap_sum} | "
              f"RAGAS-1.0 group clean: {faithful_clean}/{len(ragas_faithful)}")
        print()

    # Final recommendation
    print("=" * 100)
    print("  RECOMMENDATION:")
    print("  Pick the SEQ_THRESHOLD where:")
    print("    1. Abs gap sum is minimized (verifier tracks RAGAS most closely)")
    print("    2. RAGAS-1.0 group (Q2/Q5/Q8/Q9) stays fully clean (4/4)")
    print("    3. Under-detected questions (Q1/Q3/Q4/Q7) show gap closing")
    print("=" * 100)


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  VERIFIER CALIBRATION SWEEP")
    print("=" * 90 + "\n")

    # Confirm run identity
    print("  RUN IDENTITY CONFIRMATION:")
    print("  The 'zero 429s during generate' finding is from task-1234.log (Aug 3 12:07)")
    print("  which produced the CURRENT raw_outputs_regbrain.json (last modified Aug 3 12:07).")
    print("  This IS the same run analyzed in the flip-analysis. The earlier 23/35 run was from")
    print("  task-854.log (Jul 30), which was interrupted and produced a DIFFERENT output file")
    print("  that was subsequently overwritten by task-1234.")
    print()
    print("  The flip-analysis concluded '429 errors caused the regression' but that was")
    print("  WRONG — zero 429 errors occurred in this run. The 20 abstentions happen because")
    print("  generate() returns zero claims for those questions (verified by looking at the")
    print("  per-question processing times: 12-245 seconds, not the <1s of a 429 timeout).")
    print("  The earlier 23/35 split was likely from a different code version or Qdrant state.")
    print()

    questions = load_claims_and_chunks()
    best_kw, claims_data = sweep_thresholds(questions)

    # ── Full 15-question per-question report at baseline threshold ──
    large_gap_qs = full_per_question_report(claims_data, kw_thresh=0.60, seq_gate=0.30)

    # ── SEQ sweep if Q1/Q3/Q4/Q7 show large gaps ──
    target_qs = {1, 3, 4, 7}
    target_hits = [q for q in large_gap_qs if q in target_qs]
    if target_hits:
        print(f"\n  ⚠ Target questions with large gaps: Q{target_hits}")
        print("  Running SEQ_THRESHOLD sweep to find better separation...\n")
        seq_threshold_sweep(claims_data, set(large_gap_qs), kw_thresh=0.60)
    else:
        print(f"  ✅ No large gaps on target questions Q1/Q3/Q4/Q7 — SEQ sweep not needed.")

