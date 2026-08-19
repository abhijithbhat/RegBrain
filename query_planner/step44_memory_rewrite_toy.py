"""
Step 44 – Memory Rewrite Toy Example
=======================================
Demonstrates conversational follow-up detection and rewriting.

Hardcodes a "previous turn" (query + answer about Commercial Banks KYC)
and a follow-up query ("What about for NBFCs instead?"). Asks Groq
whether the follow-up depends on the previous turn's context and, if
so, rewrites it into a standalone query.

Forces JSON output:
  {"is_followup": true/false, "standalone_query": "..."}
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "openai/gpt-oss-120b"

# ── Hardcoded previous turn (real output from answer_query) ─────────
PREVIOUS_QUERY = "What are the KYC requirements for Commercial Banks?"

PREVIOUS_ANSWER = (
    "Commercial banks must have a Board\u2011approved KYC policy that contains "
    "four core elements \u2013 a Customer Acceptance Policy, Risk Management, "
    "Customer Identification Procedures (CIP) and Transaction Monitoring. "
    "The policy must prescribe periodic KYC updation, allow exceptional "
    "updation measures (e.g., recent photograph, physical presence, more "
    "frequent updates), require a copy of OVD for address changes, and "
    "enable customers to update KYC at any branch. Additional Board\u2011approved "
    "policies are required for handling mobile\u2011number change requests in "
    "non\u2011face\u2011to\u2011face accounts and for approving cross\u2011border correspondent "
    "banking relationships. Banks must capture each customer\u2019s KYC record "
    "and upload it to the Central KYC Records Registry (CKYCR) within 10 "
    "days of opening the account. A risk\u2011based approach to KYC upkeep is "
    "mandated, with periodic reviews at least every two years for high\u2011risk "
    "customers, every eight years for medium\u2011risk customers and every ten "
    "years for low\u2011risk customers, together with ongoing due\u2011diligence "
    "monitoring of transactions."
)

# ── Hardcoded follow-up ────────────────────────────────────────────
FOLLOWUP_QUERY = "What about for NBFCs instead?"

# ── Rewrite prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a query rewriter for a regulatory Q&A system.

You will receive:
  1. The previous user query and the system's answer from the last turn.
  2. A new user query.

Your task:
  • Determine whether the new query is a follow-up that depends on
    context from the previous turn (e.g. pronouns like "it", "they",
    "that", or elliptical references like "What about X instead?",
    "And for Y?", "Same question but for Z").
  • If it IS a follow-up, rewrite it into a fully self-contained,
    standalone query that can be sent to a retrieval system with no
    conversation history.
  • If it is NOT a follow-up, return the new query unchanged.

Respond with ONLY a JSON object — no markdown, no extra text:
{"is_followup": true/false, "standalone_query": "<the rewritten or original query>"}
"""


def rewrite_followup(
    prev_query: str,
    prev_answer: str,
    new_query: str,
) -> dict:
    """Ask Groq whether *new_query* is a follow-up; rewrite if needed."""
    user_prompt = (
        f"Previous query: {prev_query}\n"
        f"Previous answer: {prev_answer}\n\n"
        f"New query: {new_query}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

    if response.status_code != 200:
        return {
            "is_followup": None,
            "standalone_query": new_query,
            "error": f"Groq API error [{response.status_code}]: {response.text}",
        }

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "is_followup": None,
            "standalone_query": new_query,
            "error": f"Unparseable LLM output: {raw}",
        }


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("Step 44 – Memory Rewrite Toy Example")
    print(f"  Model : {MODEL}")
    print("=" * 64)

    print(f"\n  Previous query  : \"{PREVIOUS_QUERY}\"")
    print(f"  Previous answer : \"{PREVIOUS_ANSWER[:120]}…\"")
    print(f"\n  Follow-up query : \"{FOLLOWUP_QUERY}\"")
    print()

    result = rewrite_followup(PREVIOUS_QUERY, PREVIOUS_ANSWER, FOLLOWUP_QUERY)

    is_followup = result.get("is_followup")
    standalone  = result.get("standalone_query", "")

    print(f"  is_followup      : {is_followup}")
    print(f"  standalone_query : \"{standalone}\"")

    if is_followup:
        print(f"\n  ✅ Rewritten: \"{FOLLOWUP_QUERY}\" → \"{standalone}\"")
    elif is_followup is False:
        print(f"\n  ℹ️  Not a follow-up — query unchanged.")
    else:
        print(f"\n  ⚠ Error: {result.get('error', 'unknown')}")

    print(f"\n{'=' * 64}")
    print("✅ Rewrite complete.")
