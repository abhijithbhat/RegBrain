"""
Step 47 – Compare Single-hop vs Multi-hop Decomposed Answers
================================================================
For each of 3 hardcoded multi-hop questions, runs the query through:
  a) answer_query()      — single-hop, direct retrieval
  b) plan_and_answer()   — classify → decompose → fan-out → synthesize

Prints FULL answers side by side so you can judge whether decomposition
adds value over direct retrieval for each question.
"""

import json
import os
import sys
import time
import textwrap

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv                             # noqa: E402
from generation.answer_query import answer_query           # noqa: E402
from query_planner.plan_and_answer import plan_and_answer  # noqa: E402

load_dotenv()

# ── Rate-limit sleep between Groq-heavy calls ──────────────────────
RATE_LIMIT_SLEEP = 12

# ── Hardcoded questions ─────────────────────────────────────────────
QUESTIONS = [
    "What are the KYC requirements and capital adequacy norms for NBFCs?",
    "What are the governance guidelines for commercial bank boards and what are the credit facility limits for NBFCs?",
    "What are the minimum capital adequacy ratio rules for NBFCs and what are the specific loan outflow limits for households under NBFC-MFIs?",
]

SEP = "═" * 72
SUB_SEP = "─" * 72


def fmt_answer(label: str, result: dict) -> str:
    """Format one result block with full answer text."""
    status = result.get("status", "?")
    confidence = result.get("confidence", 0.0)
    citations = result.get("citations", [])
    answer = result.get("answer", result.get("reason", "(no answer)"))
    decomposed = result.get("was_decomposed", None)

    lines = [
        f"  ┌─ {label}",
        f"  │ Status      : {status}",
        f"  │ Confidence  : {confidence}",
        f"  │ Citations   : {len(citations)} claim(s)",
    ]
    if decomposed is not None:
        lines.append(f"  │ Decomposed? : {decomposed}")

    lines.append("  │")
    lines.append("  │ Answer:")

    # Wrap the answer text at 68 chars, indented under the box
    wrapped = textwrap.fill(answer, width=68)
    for line in wrapped.splitlines():
        lines.append(f"  │   {line}")

    lines.append("  │")
    lines.append("  │ Citations detail:")
    if not citations:
        lines.append("  │   (none)")
    else:
        for i, c in enumerate(citations, 1):
            clause = c.get("cited_clause_id", c.get("clause_id", "?"))
            text = c.get("text", "")
            short = textwrap.shorten(text, width=90, placeholder="…")
            lines.append(f"  │   [{i}] clause {clause}: {short}")

    lines.append(f"  └{'─' * 70}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(SEP)
    print("  Step 47 – Single-hop vs Multi-hop Comparison")
    print(SEP)

    for qi, question in enumerate(QUESTIONS, 1):
        print(f"\n{SEP}")
        print(f"  QUESTION {qi}/{len(QUESTIONS)}:")
        print(f"  \"{question}\"")
        print(SEP)

        # ── A) Single-hop direct ────────────────────────────────────
        print(f"\n  Running answer_query() (single-hop) …")
        single_result = answer_query(question)

        # Normalise keys to match the plan_and_answer output shape
        single_norm = {
            "status": single_result.get("status", "abstain"),
            "answer": single_result.get("answer", single_result.get("reason", "")),
            "citations": single_result.get("citations", []),
            "confidence": single_result.get("confidence", 0.0),
        }

        print(fmt_answer("SINGLE-HOP DIRECT", single_norm))

        # Rate-limit pause
        print(f"\n  ⏳ Sleeping {RATE_LIMIT_SLEEP}s before multi-hop run …\n")
        time.sleep(RATE_LIMIT_SLEEP)

        # ── B) Multi-hop decomposed ─────────────────────────────────
        print(f"  Running plan_and_answer() (multi-hop) …")
        multi_result = plan_and_answer(question)

        print(fmt_answer("MULTI-HOP DECOMPOSED", multi_result))

        # Pause before next question
        if qi < len(QUESTIONS):
            print(f"\n  ⏳ Sleeping {RATE_LIMIT_SLEEP}s before next question …")
            time.sleep(RATE_LIMIT_SLEEP)

    print(f"\n{SEP}")
    print("  ✅ All 3 comparisons complete.")
    print(SEP)
