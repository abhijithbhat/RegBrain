"""
Step 51 – RAGAS Contrast: GOOD vs BAD example side-by-side (resilient).

Reuses the clearly-correct Example A from Step 50, then adds a
deliberately BAD Example B (same question, but an answer with a
fabricated percentage NOT in the context, plus a partially-relevant
context chunk).  Computes faithfulness, answer_relevancy,
context_precision, and context_recall for both, printing the scores
side by side so the metric spread is obvious.

Now uses ragas_groq_utils for:
  • Markdown fence stripping on Groq responses
  • Disk-based LLM response caching
  • Exponential-backoff retry with per-question failure logging
"""

import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Import shared resilience utilities ─────────────────────────────────
from ragas_groq_utils import (  # noqa: E402
    MODEL,
    GROQ_BASE_URL,
    make_groq_judge,
    make_embeddings,
    resilient_evaluate,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Example A  –  GOOD  (same as Step 50)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION = (
    "What is the minimum net owned fund requirement for an NBFC to "
    "obtain a certificate of registration from the RBI?"
)

GOOD_CONTEXTS = [
    "As per the RBI Master Direction on NBFCs, every NBFC shall have a "
    "minimum net owned fund (NOF) of ₹2 crore to be eligible for a "
    "certificate of registration (CoR). The NOF requirement was raised "
    "from ₹25 lakh to ₹2 crore effective April 2022 for new registrations."
]

GOOD_ANSWER = (
    "The minimum net owned fund (NOF) requirement for an NBFC to obtain "
    "a certificate of registration from the RBI is ₹2 crore."
)

REFERENCE_ANSWER = (
    "An NBFC must have a minimum net owned fund of ₹2 crore to obtain "
    "a certificate of registration from the RBI."
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Example B  –  BAD
#   • Answer fabricates a wrong figure (₹10 crore) and invents a 15%
#     annual growth clause that is nowhere in the context.
#   • Context is only partially relevant: one chunk talks about NBFCs
#     but gives asset-size classification rules, not the NOF
#     registration requirement.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BAD_CONTEXTS = [
    # Partially relevant: mentions NBFCs and classification, but NOT the
    # minimum NOF for registration.
    "NBFCs are classified into different categories based on their asset "
    "size. NBFCs with asset size of ₹500 crore and above are classified "
    "as NBFC-Upper Layer under the Scale Based Regulation framework "
    "introduced in October 2022.",
    # Irrelevant chunk about Priority Sector Lending
    "Priority Sector Lending (PSL) guidelines require commercial banks "
    "to lend at least 40% of their Adjusted Net Bank Credit to priority "
    "sectors including agriculture, micro and small enterprises, and "
    "weaker sections.",
]

BAD_ANSWER = (
    "The minimum net owned fund requirement for an NBFC to obtain a "
    "certificate of registration from the RBI is ₹10 crore, and the "
    "NBFC must also demonstrate at least 15% annual growth in its "
    "managed assets for three consecutive years."
)

# ── Build the RAGAS dataset with both examples ─────────────────────────
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402

sample_a = SingleTurnSample(
    user_input=QUESTION,
    retrieved_contexts=GOOD_CONTEXTS,
    response=GOOD_ANSWER,
    reference=REFERENCE_ANSWER,
)

sample_b = SingleTurnSample(
    user_input=QUESTION,
    retrieved_contexts=BAD_CONTEXTS,
    response=BAD_ANSWER,
    reference=REFERENCE_ANSWER,
)

eval_dataset = EvaluationDataset(samples=[sample_a, sample_b])

# ── Configure Groq judge + embeddings via shared utils ─────────────────
judge_llm = make_groq_judge()
ragas_embeddings = make_embeddings()

# ── Import metrics ─────────────────────────────────────────────────────
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]

metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]

# ── Run resilient evaluation ───────────────────────────────────────────
print(f"Model       : {MODEL}")
print(f"Groq URL    : {GROQ_BASE_URL}")
print(f"Embeddings  : BAAI/bge-small-en-v1.5 (local)")
print(f"Cache       : eval/.ragas_cache/ (disk)")
print(f"Question    : {QUESTION[:80]}…")
print(f"Examples    : A (GOOD) + B (BAD)")
print("-" * 68)
print("Running RAGAS evaluation (2 samples × 4 metrics)…\n")

result = resilient_evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=ragas_embeddings,
)

# ── Extract per-row scores ─────────────────────────────────────────────
scores_a = result.scores[0]  # Example A (GOOD)
scores_b = result.scores[1]  # Example B (BAD)

# ── Print side-by-side comparison ──────────────────────────────────────
COL_W = 14  # width for each score column

print("\n" + "=" * 68)
print("  RAGAS CONTRAST SCORES (Step 51)")
print("=" * 68)
header = f"  {'Metric':<42s} {'A (GOOD)':>{COL_W}s} {'B (BAD)':>{COL_W}s}  {'Δ (A−B)':>{COL_W}s}"
print(header)
print("  " + "-" * (42 + COL_W * 3 + 6))

any_failed = False
for m in METRIC_NAMES:
    sa = scores_a.get(m, float("nan"))
    sb = scores_b.get(m, float("nan"))

    def fmt(v):
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return f"{v:.4f}"

    delta = sa - sb if not (math.isnan(sa) or math.isnan(sb)) else float("nan")

    if math.isnan(sa) or math.isnan(sb):
        any_failed = True

    print(f"  {m:<42s} {fmt(sa):>{COL_W}s} {fmt(sb):>{COL_W}s}  {fmt(delta):>{COL_W}s}")

print("=" * 68)

# ── Summary ────────────────────────────────────────────────────────────
if any_failed:
    print("\n⚠  One or more metrics returned NaN.")
    print("   Raw scores for debugging:")
    print(f"   A: {scores_a}")
    print(f"   B: {scores_b}")
else:
    print("\n✅ All metrics computed for both examples.")
    print("   Expected: A scores ≥ B scores across the board.")

print("\nDone.")
