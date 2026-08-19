"""
eval/run_eval.py – Two-Stage RegBrain RAG Evaluation Suite.

Stage 1 (--stage generate):
  Run questions through the full pipeline (RegBrain or naive), save
  raw outputs to JSON.  No RAGAS calls.

Stage 2 (--stage score):
  Load raw outputs from JSON, run RAGAS metrics, compute clause hit/miss,
  and write the final CSV.

Command-line flags:
  --stage generate|score  : Which stage to run (required)
  --pipeline regbrain|naive : Pipeline to evaluate (default: regbrain)
  --limit N, -l N         : Limit to first N questions
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.tpd_tracker import tpd_tracker  # noqa: E402


# ── Clause Hit / Miss Helper ───────────────────────────────────────────
def check_clause_hit(expected_cid: str, citations: List[Dict[str, Any]]) -> int:
    """
    Return 1 (HIT) if expected_clause_id matches any citation's regulatory clause ID, else 0 (MISS).
    """
    if not expected_cid or not citations:
        return 0

    exp_norm = expected_cid.strip().lower()

    for cite in citations:
        if isinstance(cite, dict):
            candidates = [
                str(cite.get("source_clause_id", "")).strip().lower(),
                str(cite.get("clause_label", "")).strip().lower(),
                str(cite.get("clause_id", "")).strip().lower(),
                str(cite.get("doc_id", "")).strip().lower(),
            ]
        else:
            candidates = [str(cite).strip().lower()]

        for cid in candidates:
            if cid and (exp_norm == cid or exp_norm in cid or cid in exp_norm):
                return 1
    return 0


# ── File path helpers ──────────────────────────────────────────────────
def _raw_json_path(pipeline: str) -> str:
    """Return the path for the raw outputs JSON file."""
    suffix = "naive" if pipeline == "naive" else "regbrain"
    return os.path.join(os.path.dirname(__file__), f"raw_outputs_{suffix}.json")


def _csv_path(pipeline: str) -> str:
    """Return the path for the results CSV file."""
    suffix = "naive" if pipeline == "naive" else "regbrain"
    return os.path.join(os.path.dirname(__file__), f"results_{suffix}.csv")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 1: GENERATE — Run pipeline, save raw outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stage_generate(eval_questions: List[Dict], pipeline: str) -> None:
    """Run all questions through the pipeline and save raw outputs to JSON."""
    from retrieval.retrieve import retrieve  # noqa: E402

    output_path = _raw_json_path(pipeline)

    print(f"{'=' * 75}")
    print(f"  STAGE 1: GENERATE  ({pipeline} pipeline, {len(eval_questions)} questions)")
    print(f"{'=' * 75}\n")

    records: List[Dict[str, Any]] = []

    for idx, item in enumerate(eval_questions, start=1):
        q_text = item["question"]
        expected_cid = item.get("expected_clause_id", "")
        ref_answer = item.get("reference_answer", "")

        print(f"[{idx}/{len(eval_questions)}] Processing: \"{q_text[:70]}…\"", flush=True)

        t0 = time.time()

        if pipeline == "regbrain":
            from query_planner.handle_query import handle_query  # noqa: E402
            session_state: Dict[str, Any] = {}
            pipeline_out = handle_query(q_text, session_state)
            chunks = retrieve(q_text)
        else:
            # Naive pipeline: dense-only retrieval + raw generation, no verification
            from eval.naive_baseline import naive_retrieve, naive_generate  # noqa: E402
            naive_chunks = naive_retrieve(q_text)
            naive_result = naive_generate(q_text, naive_chunks)
            pipeline_out = {
                "answer": naive_result.get("answer", ""),
                "citations": naive_result.get("citations", []),
                "confidence": 0.0,  # naive has no verifier
                "status": naive_result.get("status", "answered"),
                "was_decomposed": False,
            }
            chunks = naive_chunks

        elapsed = time.time() - t0

        final_answer = pipeline_out.get("answer", "") or "No answer generated."
        citations = pipeline_out.get("citations", [])
        confidence = float(pipeline_out.get("confidence", 0.0))
        status = pipeline_out.get("status", "abstain")
        was_decomposed = bool(pipeline_out.get("was_decomposed", False))

        # Build retrieved context texts for RAGAS in stage 2
        retrieved_texts: List[str] = []
        seen = set()
        for c in chunks:
            text = c.get("clause_text", "").strip()
            if len(text) > 2500:
                text = text[:2500] + "…"
            if text and text not in seen:
                seen.add(text)
                retrieved_texts.append(text)

        if not retrieved_texts:
            retrieved_texts = ["No relevant regulatory context was retrieved."]

        rec = {
            "question_id": idx,
            "question": q_text,
            "expected_clause_id": expected_cid,
            "status": status,
            "confidence": confidence,
            "was_decomposed": was_decomposed,
            "citations": citations,
            "citations_count": len(citations),
            "final_answer": final_answer,
            "reference_answer": ref_answer,
            "retrieved_contexts": retrieved_texts,
            "pipeline_time_s": round(elapsed, 2),
        }
        records.append(rec)

        print(f"    Status: {status} | Conf: {confidence:.1f}% | "
              f"Decomposed: {was_decomposed} | ({elapsed:.1f}s)", flush=True)

        # TPD budget status every 5 questions
        if idx % 5 == 0 or idx == len(eval_questions):
            tpd_tracker.log_status(idx, len(eval_questions))
        print(flush=True)

        # Pacing delay between questions to avoid hitting Groq per-minute rate limits
        time.sleep(1.0)

    # Save raw outputs to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 75}", flush=True)
    print(f"  ✅ STAGE 1 COMPLETE — {len(records)} raw outputs saved to:")
    print(f"     {output_path}")
    print(f"{'=' * 75}")
    tpd_tracker.summary(stage=f"generate-{pipeline}")
    print("Next step: run with --stage score to compute RAGAS metrics.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 2b: RETRY-FAILED — Re-score only failed questions in existing CSV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stage_rescore_failed(pipeline: str) -> None:
    """Load existing CSV + raw JSON, find questions with fallback scores, re-score only those."""
    from eval.ragas_groq_utils import (  # noqa: E402
        make_groq_judge,
        make_embeddings,
        resilient_evaluate,
    )
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
    from ragas.metrics import (  # noqa: E402
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    raw_path = _raw_json_path(pipeline)
    output_csv_path = _csv_path(pipeline)

    if not os.path.exists(output_csv_path):
        sys.exit(
            f"ERROR: Existing results CSV not found at {output_csv_path}\n"
            f"  Run with --stage score first (without --retry-failed)."
        )
    if not os.path.exists(raw_path):
        sys.exit(
            f"ERROR: Raw outputs not found at {raw_path}\n"
            f"  Run with --stage generate first."
        )

    # Load existing CSV rows
    with open(output_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = [row for row in reader if row.get("question_id") != "SUMMARY_AVERAGE"]

    # Load raw JSON for context data
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_records: List[Dict[str, Any]] = json.load(f)

    # Build lookup: question_id -> raw record
    raw_by_id = {r["question_id"]: r for r in raw_records}

    # Identify failed rows: precision=0.50 AND recall=0.50 (fallback sentinel values)
    FALLBACK_PRECISION = 0.50
    FALLBACK_RECALL = 0.50
    failed_indices: List[int] = []
    for i, row in enumerate(csv_rows):
        prec = float(row.get("context_precision", 0))
        recall = float(row.get("context_recall", 0))
        if abs(prec - FALLBACK_PRECISION) < 0.001 and abs(recall - FALLBACK_RECALL) < 0.001:
            failed_indices.append(i)

    if not failed_indices:
        print("✅ No failed questions found — all scores look valid!")
        return

    print(f"{'=' * 75}")
    print(f"  RETRY-FAILED: Re-scoring {len(failed_indices)} questions ({pipeline} pipeline)")
    print(f"{'=' * 75}")
    for idx in failed_indices:
        print(f"  → Q{csv_rows[idx]['question_id']}: {csv_rows[idx]['question'][:60]}")
    print()

    judge_llm = make_groq_judge()
    ragas_embeddings = make_embeddings()
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # Build RAGAS samples for failed questions only
    batch_samples = []
    batch_csv_indices = []
    for idx in failed_indices:
        row = csv_rows[idx]
        qid = int(row["question_id"])
        raw_rec = raw_by_id.get(qid, {})

        sample = SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=raw_rec.get("retrieved_contexts", []),
            response=row.get("final_answer", ""),
            reference=row.get("reference_answer", ""),
        )
        batch_samples.append(sample)
        batch_csv_indices.append(idx)

    eval_dataset = EvaluationDataset(samples=batch_samples)
    try:
        ragas_res = resilient_evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=ragas_embeddings,
            show_progress=False,
        )
        for i, csv_idx in enumerate(batch_csv_indices):
            row = csv_rows[csv_idx]
            score_dict = ragas_res.scores[i]

            if row["status"] == "answered":
                f_val = float(score_dict.get("faithfulness", float("nan")))
                r_val = float(score_dict.get("answer_relevancy", float("nan")))
                if not math.isnan(f_val):
                    row["faithfulness"] = f"{f_val:.4f}"
                if not math.isnan(r_val):
                    row["answer_relevancy"] = f"{r_val:.4f}"

            p_val = float(score_dict.get("llm_context_precision_with_reference", float("nan")))
            rec_val = float(score_dict.get("context_recall", float("nan")))
            if not math.isnan(p_val):
                row["context_precision"] = f"{p_val:.4f}"
            if not math.isnan(rec_val):
                row["context_recall"] = f"{rec_val:.4f}"

            print(f"  ✅ Q{row['question_id']} re-scored: faith={row['faithfulness']} "
                  f"rel={row['answer_relevancy']} prec={row['context_precision']} "
                  f"recall={row['context_recall']}")

    except Exception as e:
        print(f"  ⚠ Re-scoring failed: {e}")
        print("  Keeping existing fallback values.")
        return

    # Recompute averages
    def safe_avg(vals: List[float]) -> float:
        valid = [v for v in vals if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else 0.0

    all_rows = csv_rows
    avg_confidence = safe_avg([float(r["confidence"]) for r in all_rows])
    avg_clause_hit = safe_avg([float(r["clause_hit"]) for r in all_rows])
    avg_faithfulness = safe_avg([float(r["faithfulness"]) for r in all_rows])
    avg_relevancy = safe_avg([float(r["answer_relevancy"]) for r in all_rows])
    avg_precision = safe_avg([float(r["context_precision"]) for r in all_rows])
    avg_recall = safe_avg([float(r["context_recall"]) for r in all_rows])
    avg_citations = safe_avg([float(r["citations_count"]) for r in all_rows])

    # Rewrite CSV with updated scores
    fieldnames = [
        "question_id", "question", "expected_clause_id", "status",
        "confidence", "was_decomposed", "clause_hit",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "citations_count", "final_answer", "reference_answer",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        writer.writerow({
            "question_id": "SUMMARY_AVERAGE",
            "question": f"Average across {len(all_rows)} questions",
            "expected_clause_id": "-",
            "status": "-",
            "confidence": f"{avg_confidence:.2f}",
            "was_decomposed": "-",
            "clause_hit": f"{avg_clause_hit:.4f}",
            "faithfulness": f"{avg_faithfulness:.4f}",
            "answer_relevancy": f"{avg_relevancy:.4f}",
            "context_precision": f"{avg_precision:.4f}",
            "context_recall": f"{avg_recall:.4f}",
            "citations_count": f"{avg_citations:.2f}",
            "final_answer": "-",
            "reference_answer": "-",
        })

    print(f"\n{'=' * 75}")
    print(f"  ✅ RETRY-FAILED COMPLETE — {len(failed_indices)} questions re-scored")
    print(f"  Updated CSV: {output_csv_path}")
    print(f"  Faithfulness: {avg_faithfulness:.4f} | Relevancy: {avg_relevancy:.4f}")
    print(f"  Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f}")
    print(f"{'=' * 75}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STAGE 2: SCORE — Load raw outputs, run RAGAS, write CSV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stage_score(pipeline: str, limit: int | None) -> None:
    """Load raw outputs from JSON, run RAGAS metrics, write CSV."""
    from eval.ragas_groq_utils import (  # noqa: E402
        make_groq_judge,
        make_embeddings,
        resilient_evaluate,
    )
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
    from ragas.metrics import (  # noqa: E402
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    raw_path = _raw_json_path(pipeline)
    output_csv_path = _csv_path(pipeline)

    if not os.path.exists(raw_path):
        sys.exit(
            f"ERROR: Raw outputs not found at {raw_path}\n"
            f"  Run with --stage generate first."
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if limit is not None and limit > 0:
        records = records[:limit]

    print(f"{'=' * 75}")
    print(f"  STAGE 2: SCORE  ({pipeline} pipeline, {len(records)} questions)")
    print(f"{'=' * 75}\n")

    # ── Compute clause hit/miss ────────────────────────────────────────
    for rec in records:
        rec["clause_hit"] = check_clause_hit(
            rec.get("expected_clause_id", ""),
            rec.get("citations", []),
        )

    # ── Run RAGAS metrics (batched) ────────────────────────────────────
    print(f"{'─' * 75}")
    print("Running RAGAS Metric Evaluation (Faithfulness, Relevancy, Precision, Recall)…")
    print(f"{'─' * 75}\n")

    judge_llm = make_groq_judge()
    ragas_embeddings = make_embeddings()

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # Initialise metric fields
    for rec in records:
        rec["faithfulness"] = float("nan")
        rec["answer_relevancy"] = float("nan")
        rec["context_precision"] = float("nan")
        rec["context_recall"] = float("nan")

    BATCH_SIZE = 5
    for b_start in range(0, len(records), BATCH_SIZE):
        b_records = records[b_start : b_start + BATCH_SIZE]
        print(f"  Batch [{b_start + 1}..{b_start + len(b_records)} of {len(records)}]:")

        batch_samples = []
        sample_to_rec_map = []

        for rec in b_records:
            if rec["status"] == "abstain":
                # Abstained questions: faithfulness is N/A (no claims to verify),
                # answer_relevancy is 0 (no useful answer produced).
                rec["faithfulness"] = float("nan")
                rec["answer_relevancy"] = 0.0

            sample = SingleTurnSample(
                user_input=rec["question"],
                retrieved_contexts=rec.get("retrieved_contexts", []),
                response=rec.get("final_answer", ""),
                reference=rec.get("reference_answer", ""),
            )
            batch_samples.append(sample)
            sample_to_rec_map.append(rec)

        if batch_samples:
            eval_dataset = EvaluationDataset(samples=batch_samples)
            try:
                ragas_res = resilient_evaluate(
                    dataset=eval_dataset,
                    metrics=metrics,
                    llm=judge_llm,
                    embeddings=ragas_embeddings,
                    show_progress=False,
                )
                for i, rec in enumerate(sample_to_rec_map):
                    score_dict = ragas_res.scores[i]
                    if rec["status"] == "answered":
                        f_val = float(score_dict.get("faithfulness", float("nan")))
                        r_val = float(score_dict.get("answer_relevancy", float("nan")))
                        # Keep NaN if RAGAS failed — do NOT default to sentinel
                        rec["faithfulness"] = f_val
                        rec["answer_relevancy"] = r_val
                        if math.isnan(f_val) or math.isnan(r_val):
                            f_str = 'NaN' if math.isnan(f_val) else f'{f_val:.4f}'
                            r_str = 'NaN' if math.isnan(r_val) else f'{r_val:.4f}'
                            print(f"    ⚠ Q{rec['question_id']}: NOT SCORED (quota/API failure) "
                                  f"faith={f_str} relev={r_str}")

                    p_val = float(score_dict.get("llm_context_precision_with_reference", float("nan")))
                    rec_val = float(score_dict.get("context_recall", float("nan")))
                    # Keep NaN — safe_avg will exclude them
                    rec["context_precision"] = p_val
                    rec["context_recall"] = rec_val
                    if math.isnan(p_val) or math.isnan(rec_val):
                        p_str = 'NaN' if math.isnan(p_val) else f'{p_val:.4f}'
                        r_str = 'NaN' if math.isnan(rec_val) else f'{rec_val:.4f}'
                        print(f"    ⚠ Q{rec['question_id']}: context metrics NOT SCORED "
                              f"prec={p_str} recall={r_str}")
            except Exception as e:
                print(f"    ⚠ Batch evaluation warning: {e}")
                # Leave all scores as NaN — they will be excluded from averages
                for rec in sample_to_rec_map:
                    print(f"    ⚠ Q{rec['question_id']}: NOT SCORED due to batch failure")

        print(f"    ✓ Batch {b_start // BATCH_SIZE + 1} completed.")
        # TPD budget status after each batch
        tpd_tracker.log_status(b_start + len(b_records), len(records))
        time.sleep(2)  # rate limit delay between batches

    # ── Compute averages & write CSV ───────────────────────────────────
    def safe_avg(vals: List[float]) -> float:
        valid = [v for v in vals if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else 0.0

    # Count and report questions not scored due to API/quota failures
    not_scored_faith = [r for r in records if r["status"] == "answered" and math.isnan(r["faithfulness"])]
    not_scored_ctx = [r for r in records if math.isnan(r["context_precision"]) and r["status"] != "abstain"]
    if not_scored_faith:
        print(f"\n  ⚠ {len(not_scored_faith)} answered question(s) NOT SCORED for faithfulness (excluded from avg):")
        for r in not_scored_faith:
            print(f"    Q{r['question_id']}: \"{r['question'][:60]}\"")
    if not_scored_ctx:
        print(f"  ⚠ {len(not_scored_ctx)} question(s) NOT SCORED for context precision/recall (excluded from avg):")
        for r in not_scored_ctx:
            print(f"    Q{r['question_id']}: \"{r['question'][:60]}\"")

    avg_confidence = safe_avg([r["confidence"] for r in records])
    avg_clause_hit = safe_avg([float(r["clause_hit"]) for r in records])
    avg_faithfulness = safe_avg([r["faithfulness"] for r in records])
    avg_relevancy = safe_avg([r["answer_relevancy"] for r in records])
    avg_precision = safe_avg([r["context_precision"] for r in records])
    avg_recall = safe_avg([r["context_recall"] for r in records])
    avg_citations = safe_avg([float(r["citations_count"]) for r in records])

    fieldnames = [
        "question_id", "question", "expected_clause_id", "status",
        "confidence", "was_decomposed", "clause_hit",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "citations_count", "final_answer", "reference_answer",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            def _fmt(val, fmt_str=".4f"):
                return "NaN" if math.isnan(val) else f"{val:{fmt_str}}"

            writer.writerow({
                "question_id": rec["question_id"],
                "question": rec["question"],
                "expected_clause_id": rec["expected_clause_id"],
                "status": rec["status"],
                "confidence": f"{rec['confidence']:.2f}",
                "was_decomposed": rec.get("was_decomposed", False),
                "clause_hit": rec["clause_hit"],
                "faithfulness": _fmt(rec["faithfulness"]),
                "answer_relevancy": _fmt(rec["answer_relevancy"]),
                "context_precision": _fmt(rec["context_precision"]),
                "context_recall": _fmt(rec["context_recall"]),
                "citations_count": rec["citations_count"],
                "final_answer": rec.get("final_answer", ""),
                "reference_answer": rec.get("reference_answer", ""),
            })

        writer.writerow({
            "question_id": "SUMMARY_AVERAGE",
            "question": f"Average across {len(records)} questions",
            "expected_clause_id": "-",
            "status": "-",
            "confidence": f"{avg_confidence:.2f}",
            "was_decomposed": "-",
            "clause_hit": f"{avg_clause_hit:.4f}",
            "faithfulness": f"{avg_faithfulness:.4f}",
            "answer_relevancy": f"{avg_relevancy:.4f}",
            "context_precision": f"{avg_precision:.4f}",
            "context_recall": f"{avg_recall:.4f}",
            "citations_count": f"{avg_citations:.2f}",
            "final_answer": "-",
            "reference_answer": "-",
        })

    # ── Print console summary ──────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  EVALUATION SUMMARY RESULTS")
    print("=" * 75)
    print(f"  Pipeline                  : {pipeline}")
    print(f"  Total Questions Evaluated : {len(records)}")
    print(f"  Average Confidence        : {avg_confidence:.2f}%")
    print(f"  Clause Hit Rate (Precision): {avg_clause_hit * 100:.1f}%")
    print(f"  Faithfulness              : {avg_faithfulness:.4f}")
    print(f"  Answer Relevancy          : {avg_relevancy:.4f}")
    print(f"  Context Precision         : {avg_precision:.4f}")
    print(f"  Context Recall            : {avg_recall:.4f}")
    print(f"  Average Citations / Ans   : {avg_citations:.2f}")
    print("=" * 75)
    tpd_tracker.summary(stage=f"score-{pipeline}")
    print(f"\n✅ Results saved to CSV: {output_csv_path}\n")


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="RegBrain RAG Benchmark Evaluator")
    parser.add_argument(
        "--stage",
        choices=["generate", "score"],
        required=True,
        help="Stage to run: 'generate' (pipeline only) or 'score' (RAGAS metrics)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["regbrain", "naive"],
        default="regbrain",
        help="Pipeline to evaluate (default: regbrain)",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Limit evaluation to the first N questions",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=False,
        help="Re-score only the questions that got fallback RAGAS values",
    )
    args = parser.parse_args()

    if args.stage == "generate":
        eval_json_path = os.path.join(os.path.dirname(__file__), "eval_questions.json")
        if not os.path.exists(eval_json_path):
            sys.exit(f"ERROR: Benchmark file not found at {eval_json_path}")

        with open(eval_json_path, "r", encoding="utf-8") as f:
            eval_questions: List[Dict[str, Any]] = json.load(f)

        if args.limit is not None and args.limit > 0:
            print(f"ℹ  Limiting to first {args.limit} of {len(eval_questions)} questions.")
            eval_questions = eval_questions[: args.limit]
        else:
            print(f"ℹ  Running full generation on all {len(eval_questions)} questions.")

        stage_generate(eval_questions, args.pipeline)

    elif args.stage == "score":
        if args.retry_failed:
            stage_rescore_failed(args.pipeline)
        else:
            stage_score(args.pipeline, args.limit)


if __name__ == "__main__":
    main()
