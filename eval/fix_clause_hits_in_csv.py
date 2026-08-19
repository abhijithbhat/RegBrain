import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import json
import math
from eval.run_eval import check_clause_hit

# Load raw records
with open("eval/raw_outputs_regbrain.json", "r", encoding="utf-8") as f:
    raw_records = json.load(f)

raw_by_id = {r["question_id"]: r for r in raw_records}

# Load existing results_regbrain.csv
with open("eval/results_regbrain.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r.get("question_id") != "SUMMARY_AVERAGE"]

# Update clause_hit for each row
for row in rows:
    qid = int(row["question_id"])
    raw_rec = raw_by_id.get(qid, {})
    exp_cid = raw_rec.get("expected_clause_id", "")
    citations = raw_rec.get("citations", [])
    hit = check_clause_hit(exp_cid, citations)
    row["clause_hit"] = hit

# Recompute summary averages
def safe_avg(vals):
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0

def safe_float(val, default=float("nan")):
    if val is None or str(val).strip().upper() == "NAN" or str(val).strip() == "":
        return default
    try:
        return float(val)
    except:
        return default

avg_confidence = safe_avg([float(r.get("confidence", 0)) for r in rows])
avg_clause_hit = safe_avg([float(r["clause_hit"]) for r in rows])
avg_faithfulness = safe_avg([safe_float(r.get("faithfulness")) for r in rows])
avg_relevancy = safe_avg([safe_float(r.get("answer_relevancy")) for r in rows])
avg_precision = safe_avg([safe_float(r.get("context_precision")) for r in rows])
avg_recall = safe_avg([safe_float(r.get("context_recall")) for r in rows])
avg_citations = safe_avg([float(r.get("citations_count", 0)) for r in rows])

fieldnames = [
    "question_id", "question", "expected_clause_id", "status",
    "confidence", "was_decomposed", "clause_hit",
    "faithfulness", "answer_relevancy",
    "context_precision", "context_recall",
    "citations_count", "final_answer", "reference_answer",
]

with open("eval/results_regbrain.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    writer.writerow({
        "question_id": "SUMMARY_AVERAGE",
        "question": f"Average across {len(rows)} questions",
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

print(f"Updated results_regbrain.csv with correct Clause Hit Rate: {avg_clause_hit * 100:.1f}%")
