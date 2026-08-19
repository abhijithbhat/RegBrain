"""
eval/test_sentence_nli_verifier.py

Sentence-level NLI verification to fix premise-length mismatch.
Splits cited chunks into sentences, runs NLI per sentence, and takes max entailment score.
Validates offline on all 71 claims in raw_outputs_regbrain.json.
"""

import csv
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.dirname(__file__) or "."

# ── 1. Lexical Helpers ──────────────────────────────────────────────
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

def _extract_numbers(text: str) -> set[str]:
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    lower = text.lower()
    for word, digit in _WORD_TO_NUM.items():
        if word in lower:
            nums.add(digit)
    return nums


def _numbers_match_strict(claim_text: str, chunk_text: str) -> bool:
    claim_nums = _extract_numbers(claim_text)
    chunk_nums = _extract_numbers(chunk_text)
    if not claim_nums:
        return True
    missing = claim_nums - chunk_nums
    return len(missing) == 0


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly."""
    if not text:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return sents if sents else [text]


# ── Load Data ──────────────────────────────────────────────────────
def load_eval_data():
    with open(os.path.join(EVAL_DIR, "raw_outputs_regbrain.json")) as f:
        records = json.load(f)

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

    from retrieval.retrieve import retrieve

    questions = []
    for rec in records:
        if rec["status"] != "answered":
            continue
        qid = rec["question_id"]
        q_text = rec["question"]

        chunks = retrieve(q_text)
        chunk_lookup = {}
        for c in chunks:
            cid = c["clause_id"]
            if cid in chunk_lookup:
                chunk_lookup[cid] += "\n" + c["clause_text"]
            else:
                chunk_lookup[cid] = c["clause_text"]

        questions.append({
            "qid": qid,
            "question": q_text,
            "claims": rec.get("citations", []),
            "chunk_lookup": chunk_lookup,
            "ragas_faith": faith_by_qid.get(qid, float("nan")),
        })

    return questions


def run_sentence_nli_validation():
    print("=" * 90)
    print("  LOADING NLI CROSS-ENCODER (cross-encoder/nli-deberta-v3-base)...")
    print("=" * 90)
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    id2label = nli_model.config.id2label

    questions = load_eval_data()
    print(f"Loaded {len(questions)} answered questions.\n")

    # Collect sentence-level NLI pairs
    sentence_pairs = []  # (claim_detail_idx, sentence_text, claim_text)
    claim_details = []

    for q in questions:
        for claim_idx, claim in enumerate(q["claims"]):
            claim_text = claim.get("text", "")
            cid = claim.get("cited_clause_id", "")
            chunk_text = q["chunk_lookup"].get(cid)

            if chunk_text is None:
                kw = 0.0
                seq = 0.0
                nums_strict = False
                lexical_pass = False
                sentences = []
            else:
                kw = _keyword_overlap_score(claim_text, chunk_text)
                seq = _fuzzy_score(claim_text, chunk_text)
                nums_strict = _numbers_match_strict(claim_text, chunk_text)
                lexical_pass = (kw >= 0.60 and seq >= 0.35 and nums_strict)
                sentences = _split_into_sentences(chunk_text)

            detail_idx = len(claim_details)
            info = {
                "qid": q["qid"],
                "claim_idx": claim_idx,
                "claim_text": claim_text,
                "cid": cid,
                "chunk_text": chunk_text,
                "sentences": sentences,
                "kw": kw,
                "seq": seq,
                "nums_strict": nums_strict,
                "lexical_pass": lexical_pass,
                "ragas_faith": q["ragas_faith"],
            }
            claim_details.append(info)

            for s in sentences:
                sentence_pairs.append((detail_idx, s, claim_text))

    print(f"Running NLI inference on {len(sentence_pairs)} sentence-claim pairs...")
    if sentence_pairs:
        pairs_input = [(p[1], p[2]) for p in sentence_pairs]
        scores = nli_model.predict(pairs_input)

        # Aggregate sentence NLI scores by claim
        # id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        scores_by_claim = {}
        for (detail_idx, s_text, _), score in zip(sentence_pairs, scores):
            if detail_idx not in scores_by_claim:
                scores_by_claim[detail_idx] = []
            scores_by_claim[detail_idx].append({
                "sentence": s_text,
                "scores": [float(x) for x in score],
                "contradiction": float(score[0]),
                "entailment": float(score[1]),
                "neutral": float(score[2]),
                "argmax_label": id2label[score.argmax()],
            })

        for detail_idx, cd in enumerate(claim_details):
            if detail_idx not in scores_by_claim:
                cd["nli_pass"] = False
                cd["best_sentence"] = None
                cd["max_entailment"] = -999.0
                cd["best_contradiction"] = 999.0
                cd["two_stage_pass"] = False
                continue

            sent_results = scores_by_claim[detail_idx]
            # Max entailment sentence
            best_ent = max(sent_results, key=lambda x: x["entailment"])
            cd["best_sentence"] = best_ent["sentence"]
            cd["max_entailment"] = best_ent["entailment"]
            cd["best_contradiction"] = best_ent["contradiction"]
            cd["best_label"] = best_ent["argmax_label"]

            # Criteria: supported = lexical_pass AND max_entailment > contradiction_for_best_sentence
            # Also check if best_label is entailment or neutral with entailment > contradiction
            nli_pass = (best_ent["entailment"] > best_ent["contradiction"]) and (best_ent["argmax_label"] == "entailment")
            cd["nli_pass"] = nli_pass
            cd["two_stage_pass"] = cd["lexical_pass"] and nli_pass

    # ── Report ──
    print("\n" + "=" * 105)
    print("  SENTENCE-LEVEL NLI + LEXICAL VERIFIER — PER-QUESTION RESULTS")
    print("=" * 105 + "\n")

    per_q = {}
    for cd in claim_details:
        qid = cd["qid"]
        if qid not in per_q:
            per_q[qid] = {
                "total": 0,
                "lexical_unsup": 0,
                "nli_unsup": 0,
                "two_stage_unsup": 0,
                "ragas": cd["ragas_faith"],
            }
        per_q[qid]["total"] += 1
        if not cd["lexical_pass"]:
            per_q[qid]["lexical_unsup"] += 1
        if not cd.get("nli_pass", False):
            per_q[qid]["nli_unsup"] += 1
        if not cd.get("two_stage_pass", False):
            per_q[qid]["two_stage_unsup"] += 1

    rows = []
    for qid in sorted(per_q.keys()):
        v = per_q[qid]
        total = v["total"]
        flipped_two_stage = v["two_stage_unsup"]
        flipped_lexical = v["lexical_unsup"]
        flipped_nli = v["nli_unsup"]
        ragas = v["ragas"]
        ragas_implied = round(total * (1.0 - ragas)) if not math.isnan(ragas) else 0
        gap_two_stage = ragas_implied - flipped_two_stage

        rows.append({
            "qid": qid,
            "total": total,
            "lexical_flipped": flipped_lexical,
            "nli_flipped": flipped_nli,
            "two_stage_flipped": flipped_two_stage,
            "ragas_implied": ragas_implied,
            "gap": gap_two_stage,
            "ragas": ragas,
        })

    # Sort by gap (largest gap first)
    rows_sorted = sorted(rows, key=lambda r: abs(r["gap"]), reverse=True)

    print(f"  {'Q':>3s}  {'Total':>5s}  {'LexFlip':>7s}  {'NLIFlip':>7s}  {'2StgFlip':>8s}  {'RAGAS-Imp':>9s}  {'Gap':>5s}  {'RAGAS':>8s}  Note")
    print(f"  {'─'*3}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*8}  {'─'*40}")

    for r in rows_sorted:
        ragas_str = f"{r['ragas']:.4f}" if not math.isnan(r['ragas']) else "NaN"
        note = ""
        if r["gap"] > 0:
            note = f"UNDER-DETECTED by {r['gap']}"
        elif r["gap"] < 0:
            note = f"OVER-DETECTED by {abs(r['gap'])}"
        else:
            note = "EXACT MATCH WITH RAGAS"

        if r["qid"] in (1, 4, 7):
            note += "  ← TARGET Q"
        elif r["qid"] in (2, 5, 8, 9):
            note += "  ← CLEAN GROUP"

        print(f"  Q{r['qid']:>2d}  {r['total']:>5d}  {r['lexical_flipped']:>7d}  {r['nli_flipped']:>7d}  "
              f"{r['two_stage_flipped']:>8d}  {r['ragas_implied']:>9d}  {r['gap']:>+5d}  {ragas_str:>8s}  {note}")

    print()

    # ── Focus Question Detailed Report ──
    print("=" * 105)
    print("  FOCUS QUESTION DETAILED CHECK")
    print("=" * 105)

    print("\n  1. RAGAS 1.0 Clean Group (Q2, Q5, Q8, Q9):")
    clean_qs = [2, 5, 8, 9]
    for qid in clean_qs:
        r = next(r for r in rows if r["qid"] == qid)
        is_clean = (r["two_stage_flipped"] == 0)
        status_icon = "✅ CLEAN" if is_clean else f"❌ {r['two_stage_flipped']}/{r['total']} flipped"
        print(f"     Q{qid}: Total {r['total']} | Flipped: {r['two_stage_flipped']} | Status: {status_icon}")

    print("\n  2. Target Questions (Q1, Q4, Q7 - Lexical untouchables):")
    target_qs = [1, 4, 7]
    for qid in target_qs:
        r = next(r for r in rows if r["qid"] == qid)
        print(f"     Q{qid}: Total {r['total']} | Flipped: {r['two_stage_flipped']} | RAGAS Implied: {r['ragas_implied']} | Gap: {r['gap']:+d}")

    # Totals
    total_two_stage_flipped = sum(r["two_stage_flipped"] for r in rows)
    total_lexical_flipped = sum(r["lexical_flipped"] for r in rows)
    total_ragas_implied = sum(r["ragas_implied"] for r in rows)
    gap_total = total_ragas_implied - total_two_stage_flipped

    print("\n" + "=" * 105)
    print("  SUMMARY TOTALS")
    print("=" * 105)
    print(f"  Lexical-only Flipped : {total_lexical_flipped} claims")
    print(f"  Sentence NLI Flipped : {total_two_stage_flipped} claims")
    print(f"  RAGAS Implied Target : {total_ragas_implied} claims")
    print(f"  Final Sentence-NLI Gap: {gap_total:+d} (Abs gap sum: {sum(abs(r['gap']) for r in rows)})")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_sentence_nli_validation()
