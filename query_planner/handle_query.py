"""
handle_query.py
Session-aware query handler: follow-up rewrite → plan_and_answer.

Maintains a simple session_state dict with "last_query" and
"last_answer". If a new query is a conversational follow-up to the
previous turn, it is rewritten into a standalone query before being
routed through the planner.

Usage as a module:
    from query_planner.handle_query import handle_query
    state = {}
    result = handle_query("What are the KYC requirements for NBFCs?", state)
    result2 = handle_query("What about capital adequacy?", state)

Usage from the command line (simulates a 2-turn conversation):
    python query_planner/handle_query.py
"""

import json
import logging
import os
import sys
import time

import requests
from dotenv import load_dotenv

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from query_planner.plan_and_answer import plan_and_answer  # noqa: E402

from generation.groq_client import groq_json_completion  # noqa: E402

RATE_LIMIT_SLEEP = 0.2  # seconds between Groq calls

# ── Follow-up rewrite prompt (Strict referential gating) ────────────
REWRITE_PROMPT = """\
You are a precision query classifier and rewriter for a regulatory Q&A system.

You will receive:
  1. The previous user query and the system's answer from the last turn.
  2. A new user query.

STRICT FOLLOW-UP RULES:
  • An input is a follow-up (is_followup: true) ONLY IF it contains EXPLICIT referential or elliptical language that makes it incomplete without previous context (e.g., pronouns like "it", "they", "that", "these", "same rule", or elliptical phrases like "What about X instead?", "And for Y?", "How about Z?").
  • If the new query is already a complete, standalone question with a clear subject and predicate (such as a comparison query like "Compare the KYC requirements between Commercial Banks and NBFCs" or a completely new topic like "What is the capital adequacy ratio for banks?"), you MUST return is_followup: false and return the new query UNCHANGED.
  • Do NOT merge, mix, or blend prior-turn content into complete standalone questions.

Respond with ONLY a JSON object — no markdown, no extra text:
{"is_followup": true/false, "standalone_query": "<the rewritten or original query>"}
"""


def _rewrite_if_followup(
    prev_query: str,
    prev_answer: str,
    new_query: str,
) -> dict:
    """Check if *new_query* is a follow-up; rewrite if needed."""
    user_prompt = (
        f"Previous query: {prev_query}\n"
        f"Previous answer: {prev_answer[:400]}\n\n"
        f"New query: {new_query}"
    )

    result = groq_json_completion(REWRITE_PROMPT, user_prompt)
    if "_error" in result:
        return {"is_followup": False, "standalone_query": new_query}

    return {
        "is_followup": bool(result.get("is_followup", False)),
        "standalone_query": result.get("standalone_query", new_query) or new_query,
    }


# ── Public API ──────────────────────────────────────────────────────
def handle_query(query: str, session_state: dict) -> dict:
    """
    Session-aware query handler.

    1. If session_state has a previous turn, check whether *query*
       is a follow-up and rewrite it into a standalone query.
    2. Pass the (possibly rewritten) query into plan_and_answer().
    3. Update session_state with this turn's query and answer.
    4. Return plan_and_answer()'s result, augmented with rewrite info.

    Parameters
    ----------
    query : str
        The raw user query (may be a follow-up).
    session_state : dict
        Mutable dict with optional keys "last_query" and "last_answer".
        Updated in-place after each call.

    Returns
    -------
    dict
        plan_and_answer() result plus:
        - original_query   : the raw query as typed by the user
        - rewritten_query  : the standalone query actually executed
        - was_rewritten    : bool
    """
    original_query = query
    was_rewritten = False
    standalone_query = query

    # ── 1. Follow-up detection & rewrite ────────────────────────────
    prev_query = session_state.get("last_query")
    prev_answer = session_state.get("last_answer")

    if prev_query and prev_answer:
        rewrite_result = _rewrite_if_followup(prev_query, prev_answer, query)

        if rewrite_result.get("is_followup"):
            standalone_query = rewrite_result.get("standalone_query", query)
            was_rewritten = True
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            standalone_query = query
            was_rewritten = False

    # ── 2. Plan and answer ──────────────────────────────────────────
    result = plan_and_answer(standalone_query)

    # ── 3. Update session state ─────────────────────────────────────
    session_state["last_query"] = original_query
    session_state["last_answer"] = result.get("answer", "")

    # ── 4. Augment result with rewrite info ─────────────────────────
    result["original_query"] = original_query
    result["rewritten_query"] = standalone_query
    result["was_rewritten"] = was_rewritten

    return result


# ── CLI: simulate a 2-turn conversation ─────────────────────────────
def main() -> None:
    print("=" * 64)
    print("handle_query – 2-Turn Conversation Demo")
    print("=" * 64)

    session_state: dict = {}

    # ── Turn 1 ──────────────────────────────────────────────────────
    turn1_query = "What are the KYC requirements for Commercial Banks?"

    print(f"\n{'─' * 64}")
    print(f"  TURN 1: \"{turn1_query}\"")
    print(f"{'─' * 64}\n")

    result1 = handle_query(turn1_query, session_state)

    print(f"  Status       : {result1['status']}")
    print(f"  Confidence   : {result1['confidence']}")
    print(f"  Decomposed?  : {result1['was_decomposed']}")
    print(f"  Rewritten?   : {result1['was_rewritten']}")
    print(f"  Citations    : {len(result1.get('citations', []))} claim(s)")
    print(f"\n  Answer (first 300 chars):")
    print(f"    {result1['answer'][:300]}…")

    # Rate-limit pause before Turn 2
    print(f"\n  ⏳ Sleeping {RATE_LIMIT_SLEEP}s before Turn 2 …")
    time.sleep(RATE_LIMIT_SLEEP)

    # ── Turn 2: follow-up ───────────────────────────────────────────
    turn2_query = "What about for NBFCs instead?"

    print(f"\n{'─' * 64}")
    print(f"  TURN 2: \"{turn2_query}\"")
    print(f"{'─' * 64}\n")

    result2 = handle_query(turn2_query, session_state)

    print(f"  Was rewritten? : {result2['was_rewritten']}")
    if result2['was_rewritten']:
        print(f"  Rewritten to   : \"{result2['rewritten_query']}\"")
    print(f"  Status         : {result2['status']}")
    print(f"  Confidence     : {result2['confidence']}")
    print(f"  Decomposed?    : {result2['was_decomposed']}")
    print(f"  Citations      : {len(result2.get('citations', []))} claim(s)")
    print(f"\n  Answer (first 300 chars):")
    print(f"    {result2['answer'][:300]}…")

    print(f"\n{'=' * 64}")
    print("✅ 2-turn conversation complete.")


if __name__ == "__main__":
    main()
