"""
Step 41 – Synthesize Sub-answers into a Final Response
========================================================
Takes the sub-query / sub-answer pairs produced by step 40 (hardcoded)
and asks Groq to synthesize one coherent answer to the original
question: "Has the NBFC lending limit changed since 2023?"

Also merges all citations from every sub-answer into one deduplicated
list (by clause_id).
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

# ── Hardcoded inputs (from step 40 output) ──────────────────────────
ORIGINAL_QUERY = "What are the KYC requirements and capital adequacy norms for NBFCs?"

SUB_RESULTS = [
    {
        "sub_query": "KYC requirements for NBFCs as specified in RBI circulars",
        "status": "answered",
        "answer": (
            "NBFCs must perform customer due‑diligence in accordance with the "
            "RBI's Know Your Customer (KYC) standards set out in the RBI "
            "(Non‑Banking Financial Companies – Know Your Customer) Directions, "
            "2025, and must also comply with the related Anti‑Money‑Laundering "
            "(AML), Combating the Financing of Terrorism (CFT), Prevention of "
            "Money‑Laundering Act (PMLA) and Foreign Exchange Management Act "
            "(FEMA) requirements. For NBFC branches or subsidiaries operating "
            "abroad, if the host‑country's KYC/AML rules differ from RBI's, "
            "the NBFC must follow the more stringent of the two regimes."
        ),
        "citations": [
            {
                "text": "NBFCs must carry out customer due‑diligence in line with RBI KYC Directions 2025, AML, CFT, PMLA and FEMA provisions.",
                "cited_clause_id": "A.",
            },
            {
                "text": "For overseas branches/subsidiaries, NBFCs must adopt the more stringent KYC/AML regulation between RBI and the host‑country rules.",
                "cited_clause_id": "E.",
            },
        ],
        "confidence": 100.0,
    },
    {
        "sub_query": "Capital adequacy norms for NBFCs as specified in RBI master directions",
        "status": "answered",
        "answer": (
            "Under the RBI (Non‑Banking Financial Companies – Prudential Norms "
            "on Capital Adequacy) Directions, 2025, an NBFC (including NBFC‑MFI) "
            "must maintain a capital adequacy ratio of at least 15 % of its "
            "aggregate risk‑weighted assets (on‑balance‑sheet) and the "
            "risk‑adjusted value of off‑balance‑sheet items. The total Tier 2 "
            "capital at any time may not exceed 100 % of Tier 1 capital. The "
            "computation of capital and the treatment of on‑balance‑sheet and "
            "off‑balance‑sheet assets, as well as deferred tax assets and "
            "liabilities, must follow the provisions laid out in those Directions."
        ),
        "citations": [
            {
                "text": "An NBFC‑MFI shall maintain a capital adequacy ratio of not less than 15 % of its aggregate risk‑weighted assets of on‑balance sheet and risk‑adjusted value of off‑balance sheet items.",
                "cited_clause_id": "A.",
            },
            {
                "text": "The total of Tier 2 capital at any point of time shall not exceed 100 % of Tier 1 capital.",
                "cited_clause_id": "Chapter II",
            },
            {
                "text": "The treatment to on‑balance and off‑balance sheet assets for capital adequacy shall be as provided in the RBI (Non‑Banking Financial Companies – Prudential Norms on Capital Adequacy) Directions, 2025.",
                "cited_clause_id": "A.",
            },
            {
                "text": "An NBFC‑MFI shall also adhere to provisions of the RBI Directions on treatment of deferred tax assets and deferred tax liabilities for computation of capital.",
                "cited_clause_id": "A.",
            },
        ],
        "confidence": 100.0,
    },
]

# ── Synthesis prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a regulatory compliance assistant synthesizing a final answer
from the results of multiple independent sub-queries.

You will receive:
  1. The original user question.
  2. A list of sub-query / sub-answer pairs (each produced by an
     independent retrieval pipeline over RBI circulars).

Your job:
  • Combine the sub-answers into ONE coherent response that directly
    answers the original question.
  • If one or more sub-queries returned "abstain" (meaning the corpus
    had no grounded evidence), clearly state which parts could not be
    answered and why — do NOT invent information.
  • If ALL sub-queries abstained, say so honestly: the available
    regulatory corpus does not contain the information needed.

Respond with ONLY a JSON object — no markdown, no extra text:
{
  "synthesized_answer": "<your final answer>",
  "answerable": true/false,
  "reasoning": "<one-sentence explanation of how you merged the sub-answers>"
}
"""


def merge_citations(sub_results: list[dict]) -> list[dict]:
    """Collect all citations across sub-answers, deduplicated by clause_id."""
    seen: set[str] = set()
    merged: list[dict] = []
    for sr in sub_results:
        for cite in sr.get("citations", []):
            cid = cite.get("cited_clause_id", cite.get("clause_id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(cite)
    return merged


def synthesize(original_query: str, sub_results: list[dict]) -> dict:
    """Ask Groq to synthesize sub-answers into a final response."""
    # Build the user message with all sub-query/sub-answer pairs
    pairs_text = ""
    for i, sr in enumerate(sub_results, 1):
        pairs_text += (
            f"Sub-query {i}: {sr['sub_query']}\n"
            f"  Status: {sr['status']}\n"
            f"  Answer: {sr['answer']}\n"
            f"  Confidence: {sr['confidence']}\n"
            f"  Citations: {len(sr.get('citations', []))} claim(s)\n\n"
        )

    user_prompt = (
        f"Original question: {original_query}\n\n"
        f"Sub-query results:\n{pairs_text}"
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
            "synthesized_answer": f"Groq API error [{response.status_code}]: {response.text}",
            "answerable": False,
            "reasoning": "API error",
        }

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "synthesized_answer": raw,
            "answerable": False,
            "reasoning": "Unparseable LLM output",
        }


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("Step 41 – Synthesize Sub-answers")
    print(f"  Model : {MODEL}")
    print("=" * 64)

    print(f"\n  Original question: \"{ORIGINAL_QUERY}\"")
    print(f"  Sub-results       : {len(SUB_RESULTS)}")

    # Show input sub-results
    for i, sr in enumerate(SUB_RESULTS, 1):
        print(f"\n  ── Sub-query {i} ──")
        print(f"    Q: \"{sr['sub_query']}\"")
        print(f"    Status     : {sr['status']}")
        print(f"    Confidence : {sr['confidence']}")
        print(f"    Answer     : {sr['answer']}")

    # Merge citations
    merged_citations = merge_citations(SUB_RESULTS)

    # Synthesize
    print(f"\n{'─' * 64}")
    print("  Synthesizing final answer …")
    print(f"{'─' * 64}\n")

    result = synthesize(ORIGINAL_QUERY, SUB_RESULTS)

    answerable = result.get("answerable", False)
    synth_answer = result.get("synthesized_answer", "")
    reasoning = result.get("reasoning", "")

    print(f"  Answerable : {answerable}")
    print(f"  Reasoning  : {reasoning}")
    print()
    print(f"  Synthesized answer:")
    print()
    for line in synth_answer.splitlines():
        print(f"    {line}")

    # Merged citations
    print(f"\n{'─' * 64}")
    print(f"  Merged citations: {len(merged_citations)} unique clause(s)")
    print(f"{'─' * 64}")
    if merged_citations:
        for j, cite in enumerate(merged_citations, 1):
            claim = cite.get("text", cite.get("claim", ""))
            cid = cite.get("cited_clause_id", cite.get("clause_id", "?"))
            print(f"    [{j}] clause {cid}: {claim}")
    else:
        print("    (none — all sub-queries abstained)")

    print(f"\n{'=' * 64}")
    print("✅ Synthesis complete.")
