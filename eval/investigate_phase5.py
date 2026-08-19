"""
Phase 5 Pre-Finalization Investigation Script.

Addresses 4 items:
1. Faithfulness Sentinel — identify questions where NaN was defaulted to 1.0
2. Naive Hit Rate — print citations for naive "hits" and confirm dense-only chunks
3. Rate-Limit Error Type — parse 429 error bodies from logs
4. Context Spot Check — compare RegBrain vs Naive retrieved_contexts side-by-side
"""

import json
import os
import random
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVAL_DIR = os.path.dirname(__file__) or "."


def load_raw(pipeline: str) -> list[dict]:
    path = os.path.join(EVAL_DIR, f"raw_outputs_{pipeline}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def investigate_item_1_faithfulness_sentinel():
    """Item 1: Identify which questions got the NaN->1.0 fallback for faithfulness."""
    print("\n" + "=" * 80)
    print("  ITEM 1: FAITHFULNESS SENTINEL — Identifying NaN->1.0 fallback questions")
    print("=" * 80 + "\n")

    csv_path = os.path.join(EVAL_DIR, "results_regbrain.csv")
    if not os.path.exists(csv_path):
        print("  ERROR: results_regbrain.csv not found.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("question_id") != "SUMMARY_AVERAGE"]

    sentinel_questions = []
    answered_with_real_score = []
    abstained = []

    for row in rows:
        qid = row["question_id"]
        status = row["status"]
        faith = float(row.get("faithfulness", 0))
        relev = float(row.get("answer_relevancy", 0))

        if status == "abstain":
            abstained.append(qid)
        elif status == "answered":
            if faith == 1.0 and relev == 0.85:
                sentinel_questions.append((qid, row["question"][:70], faith, relev))
            elif faith == 1.0:
                sentinel_questions.append((qid, row["question"][:70], faith, relev))
            else:
                answered_with_real_score.append((qid, row["question"][:70], faith, relev))

    print(f"  Total questions: {len(rows)}")
    print(f"  Abstained: {len(abstained)}")
    print(f"  Answered with potentially sentinel scores: {len(sentinel_questions)}")
    print(f"  Answered with non-sentinel scores: {len(answered_with_real_score)}")
    print()

    if sentinel_questions:
        print("  Questions with POTENTIAL sentinel faithfulness=1.0:")
        for qid, q, f, r in sentinel_questions:
            marker = " <-- likely fallback" if r == 0.85 else ""
            print(f"    Q{qid}: faith={f:.4f} relev={r:.4f} \"{q}\"{marker}")
    else:
        print("  No sentinel faithfulness=1.0 detected among answered questions.")

    print()
    if answered_with_real_score:
        print("  Questions with real (non-sentinel) RAGAS scores:")
        for qid, q, f, r in answered_with_real_score:
            print(f"    Q{qid}: faith={f:.4f} relev={r:.4f} \"{q}\"")

    print()
    print("  CURRENT FALLBACK LINES IN run_eval.py:")
    print("    Line 448: rec['faithfulness'] = f_val if not math.isnan(f_val) else 1.0")
    print("    Line 449: rec['answer_relevancy'] = r_val if not math.isnan(r_val) else 0.85")
    print("    Line 453: rec['context_precision'] = p_val if not math.isnan(p_val) else 0.50")
    print("    Line 454: rec['context_recall'] = rec_val if not math.isnan(rec_val) else 0.50")
    print()
    print("  FIX NEEDED: Replace sentinel defaults with NaN and exclude from averages.")


def investigate_item_2_naive_hit_rate():
    """Item 2: Print citations for every naive-pipeline question that counted as a 'hit'."""
    print("\n" + "=" * 80)
    print("  ITEM 2: NAIVE HIT RATE — Citations for 'hit' questions")
    print("=" * 80 + "\n")

    naive_data = load_raw("naive")

    eq_path = os.path.join(EVAL_DIR, "eval_questions.json")
    with open(eq_path, "r", encoding="utf-8") as f:
        eval_qs = json.load(f)

    expected_map = {}
    for i, q in enumerate(eval_qs):
        expected_map[i + 1] = q.get("expected_clause_id", "")

    hits = 0
    total = 0
    for rec in naive_data:
        qid = rec["question_id"]
        exp_cid = expected_map.get(qid, rec.get("expected_clause_id", ""))
        citations = rec.get("citations", [])

        if not exp_cid:
            continue
        total += 1

        exp_norm = exp_cid.strip().lower()
        is_hit = False
        for cite in citations:
            if isinstance(cite, dict):
                cid = str(cite.get("cited_clause_id", cite.get("clause_id", ""))).strip().lower()
            else:
                cid = str(cite).strip().lower()
            if cid and (exp_norm == cid or exp_norm in cid or cid in exp_norm):
                is_hit = True
                break

        if is_hit:
            hits += 1
            print(f"  HIT Q{qid}: \"{rec['question'][:70]}\"")
            print(f"    Expected clause: {exp_cid}")
            print(f"    Citations ({len(citations)}):")
            for c in citations:
                if isinstance(c, dict):
                    print(f"      - {c}")
                else:
                    print(f"      - \"{c}\"")

            contexts = rec.get("retrieved_contexts", [])
            print(f"    Retrieved contexts ({len(contexts)} chunks):")
            for i, ctx in enumerate(contexts, 1):
                print(f"      Chunk {i}: \"{ctx[:120]}...\"")
            print()

    print(f"  Summary: {hits}/{total} questions with expected_clause_id were hits ({hits/total*100:.1f}%)")


def investigate_item_4_context_spot_check():
    """Item 4: Print full retrieved_contexts for 3 random RegBrain + 3 random Naive questions side by side."""
    print("\n" + "=" * 80)
    print("  ITEM 4: CONTEXT SPOT CHECK — RegBrain vs Naive retrieved_contexts")
    print("=" * 80 + "\n")

    rb_data = load_raw("regbrain")
    nv_data = load_raw("naive")

    rb_by_id = {r["question_id"]: r for r in rb_data}
    nv_by_id = {r["question_id"]: r for r in nv_data}

    common_ids = sorted(set(rb_by_id.keys()) & set(nv_by_id.keys()))
    random.seed(42)
    sample_ids = random.sample(common_ids, min(3, len(common_ids)))

    for qid in sample_ids:
        rb_rec = rb_by_id[qid]
        nv_rec = nv_by_id[qid]

        print(f"  -- Q{qid}: \"{rb_rec['question'][:70]}\" --")
        print()

        rb_ctx = rb_rec.get("retrieved_contexts", [])
        nv_ctx = nv_rec.get("retrieved_contexts", [])

        print(f"    RegBrain retrieved ({len(rb_ctx)} chunks):")
        for i, ctx in enumerate(rb_ctx, 1):
            print(f"      [{i}] \"{ctx[:150]}...\"")

        print()
        print(f"    Naive retrieved ({len(nv_ctx)} chunks):")
        for i, ctx in enumerate(nv_ctx, 1):
            print(f"      [{i}] \"{ctx[:150]}...\"")

        rb_set = set(c[:100] for c in rb_ctx)
        nv_set = set(c[:100] for c in nv_ctx)
        overlap = rb_set & nv_set
        print(f"\n    Overlap: {len(overlap)}/{max(len(rb_ctx), len(nv_ctx))} chunks share the same first 100 chars")
        if overlap:
            print(f"    WARNING: Some chunks are identical!")
        else:
            print(f"    CONFIRMED: Genuinely different chunk sets from different retrieval calls.")
        print()


if __name__ == "__main__":
    investigate_item_1_faithfulness_sentinel()
    investigate_item_2_naive_hit_rate()
    investigate_item_4_context_spot_check()
