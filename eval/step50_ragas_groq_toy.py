"""
Step 50 – RAGAS + Groq Toy Example (resilient).

Single hardcoded example: one question, one answer, one context list, and
one reference answer.  Uses Groq (openai/gpt-oss-120b) as the RAGAS LLM
judge via the OpenAI-compatible endpoint, and local HuggingFace embeddings
for AnswerRelevancy.

Now uses ragas_groq_utils for:
  • Markdown fence stripping on Groq responses
  • Disk-based LLM response caching
  • Exponential-backoff retry with per-question failure logging

Computes:
  • faithfulness
  • answer_relevancy
  • context_precision  (with reference)
  • context_recall
"""

import math
import sys
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

# ── Hardcoded toy example ──────────────────────────────────────────────
# A simple, clearly-correct example so we expect near-perfect scores.
QUESTION = (
    "What is the minimum net owned fund requirement for an NBFC to "
    "obtain a certificate of registration from the RBI?"
)

CONTEXTS = [
    "As per the RBI Master Direction on NBFCs, every NBFC shall have a "
    "minimum net owned fund (NOF) of ₹2 crore to be eligible for a "
    "certificate of registration (CoR). The NOF requirement was raised "
    "from ₹25 lakh to ₹2 crore effective April 2022 for new registrations."
]

ANSWER = (
    "The minimum net owned fund (NOF) requirement for an NBFC to obtain "
    "a certificate of registration from the RBI is ₹2 crore."
)

REFERENCE_ANSWER = (
    "An NBFC must have a minimum net owned fund of ₹2 crore to obtain "
    "a certificate of registration from the RBI."
)

# ── Build the RAGAS dataset ────────────────────────────────────────────
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402

sample = SingleTurnSample(
    user_input=QUESTION,
    retrieved_contexts=CONTEXTS,
    response=ANSWER,
    reference=REFERENCE_ANSWER,
)
eval_dataset = EvaluationDataset(samples=[sample])

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
print("-" * 60)
print("Running RAGAS evaluation (1 sample, 4 metrics)…\n")

result = resilient_evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=judge_llm,
    embeddings=ragas_embeddings,
)

# ── Print scores ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RAGAS SCORES (Step 50 – Toy Example)")
print("=" * 60)

any_failed = False
for metric_name, score in result._repr_dict.items():
    if isinstance(score, float) and (math.isnan(score) or score < 0):
        status = "⚠ FAILED (NaN)"
        any_failed = True
    else:
        status = f"{score:.4f}"
    print(f"  {metric_name:40s} {status}")

print("=" * 60)

if any_failed:
    print("\n⚠  One or more metrics failed to compute.")
    print("   Raw result dataframe for debugging:\n")
    try:
        df = result.to_pandas()
        print(df.to_string(index=False))
    except Exception:
        print("   Raw scores:", result.scores)
else:
    print("\n✅ All four metrics computed successfully.")

print("\nDone.")
