"""
generate.py
Grounded answer generation: retrieve real chunks → send to Groq → structured JSON.

Usage as a module:
    from generation.generate import generate
    result = generate("What are the KYC requirements for NBFCs?")

Usage from the command line:
    python generation/generate.py "your query here"
"""

import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

# Allow running from the project root (python generation/generate.py …)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retrieve import retrieve  # noqa: E402

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = [
    os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
]
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds exponential backoff base

_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def _error_response(message: str) -> dict:
    """Keep provider failures distinguishable from an evidence abstention."""
    return {"_error": message, "answer": "", "claims": []}

SYSTEM_PROMPT = """\
You are a regulatory compliance assistant.
Answer the user's question using ONLY the provided context chunks.
Do NOT use any prior knowledge.

STRICT GROUNDING RULES:
1. You must NOT supply any specific fact — including numbers, percentages, form/return names, report names, codes, section titles, or named responsibilities — unless that exact fact appears verbatim in a provided context chunk.
2. If any part or sub-part of the user's question is not covered by the provided context chunks, you MUST state plainly in the answer for that specific part: "The available regulatory corpus does not contain grounded information on [topic]". Do NOT invent or supply plausible-sounding information for missing sub-parts.
3. If the context contains no relevant information to answer the question at all, set "answer" to "The available regulatory corpus does not contain grounded information on this topic." and return an empty "claims" list.

CRITICAL FOR CLAIMS: When creating claims, use the EXACT wording and phrases from the context chunks. Quote verbatim from the source text rather than paraphrasing. Each claim must use terminology, numbers, and phrases that appear verbatim in the cited chunk.

Each context chunk has a unique `citation_id`. Use that exact `citation_id` in
`cited_clause_id`; never use the human-readable clause label, because labels
such as "A." and "B." can occur in more than one document. If the supplied
context addresses the question, return at least one atomic, verbatim claim.

Respond with valid JSON in exactly this shape (no markdown fences):
{
  "answer": "<your answer>",
  "claims": [
    {"text": "<atomic claim using verbatim language from the chunk>", "cited_clause_id": "<citation_id from context>"}
  ]
}
"""


# ── Public API ──────────────────────────────────────────────────────
def generate(query: str, chunks: list[dict] | None = None) -> dict:
    """
    End-to-end grounded generation.

    1. Use the supplied chunks, or retrieve the top chunks for *query*.
    2. Send query + those exact chunks to Groq with grounding instructions.
    3. Return a dict with keys ``answer`` and ``claims``.
    """
    if not GROQ_API_KEY:
        sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

    # ── 1. Retrieve ─────────────────────────────────────────────────
    # The caller may already have retrieved and logged a set of chunks.  Reuse
    # it so generation and verification always operate on the same evidence.
    if chunks is None:
        chunks = retrieve(query)

    context_for_prompt = [
        {
            "citation_id": f"C{index:02d}",
            "document_id": c.get("doc_id", ""),
            "clause_label": c.get("clause_id", ""),
            "clause_text": c["clause_text"][:1200] + ("..." if len(c["clause_text"]) > 1200 else "")
        }
        for index, c in enumerate(chunks[:8], start=1)
    ]

    user_prompt = (
        f"Context chunks:\n{json.dumps(context_for_prompt, indent=2)}\n\n"
        f"Question: {query}"
    )

    # ── 2. Call Groq with multi-model failover ──────────────────────
    raw_text = None
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for model_name in GROQ_MODELS:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=25)
                if response.status_code == 200:
                    raw_text = response.json()["choices"][0]["message"]["content"]
                    break
                elif response.status_code == 429:
                    # Instant failover to next model if available
                    if model_name != GROQ_MODELS[-1]:
                        break
                    time.sleep(BASE_DELAY * attempt)
                elif response.status_code in (500, 502, 503, 504):
                    time.sleep(BASE_DELAY * attempt)
            except Exception:
                if attempt < MAX_RETRIES:
                    time.sleep(BASE_DELAY)
        if raw_text is not None:
            break

    if raw_text is None:
        return _error_response("No response from Groq API after model failovers")

    # ── 3. Parse JSON (strip markdown fences if present) ────────────
    cleaned = raw_text.strip()
    fence_match = _FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Defensive compatibility for a provider response with incidental
        # prose around an otherwise valid JSON object.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                result = None
        else:
            result = None

    if not isinstance(result, dict):
        return {
            "answer": raw_text,
            "claims": [],
        }

    # Keep the generator's public contract stable even if the model returns a
    # malformed-but-parseable object.
    result["answer"] = str(result.get("answer", ""))
    result["claims"] = result.get("claims", [])
    if not isinstance(result["claims"], list):
        result["claims"] = []

    result["claims"] = [
        claim
        for claim in result["claims"]
        if isinstance(claim, dict)
        and isinstance(claim.get("text"), str)
        and isinstance(claim.get("cited_clause_id"), str)
    ]

    return result


# ── CLI ─────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python generation/generate.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')
    print("Retrieving chunks …")

    result = generate(query)

    print("\n── Generated answer (JSON) ───────────────────────────")
    print(json.dumps(result, indent=2))
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
