"""
plan_and_answer.py
Unified query planner: classify → (optionally decompose → fan-out →
synthesize) → return a single result.

Combines the logic from steps 38-41 into one entry-point function.

Usage as a module:
    from query_planner.plan_and_answer import plan_and_answer
    result = plan_and_answer("What are the KYC requirements for NBFCs?")

Usage from the command line:
    python query_planner/plan_and_answer.py "your query here"
"""

import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation.answer_query import answer_query  # noqa: E402

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

MAX_RETRIES = 6
BASE_DELAY = 6.0
RATE_LIMIT_SLEEP = 3  # seconds between internal steps (backoff retries handle 429)


# ── Groq helper ─────────────────────────────────────────────────────
def _groq_json_call(system_prompt: str, user_prompt: str) -> dict:
    """Send a request to Groq with JSON mode, return parsed dict."""
    if not GROQ_API_KEY:
        sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                break
            elif response.status_code in (429, 500, 502, 503, 504):
                retry_after = 0.0
                if "retry-after" in response.headers:
                    try:
                        retry_after = float(response.headers["retry-after"])
                    except Exception:
                        pass
                delay = max(retry_after, BASE_DELAY * (2 ** (attempt - 1)))
                if response.status_code == 429 and delay < 15.0:
                    delay = 15.0 * attempt
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    continue
                else:
                    return {"_error": f"Groq API error [{response.status_code}] after {MAX_RETRIES} retries"}
            else:
                return {"_error": f"Groq API error [{response.status_code}]: {response.text}"}
        except Exception as err:
            delay = BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                continue
            return {"_error": f"Groq connection exception: {err}"}

    if response is None or response.status_code != 200:
        return {"_error": "No valid response from Groq API"}

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": f"Unparseable LLM output: {raw}"}


# ── Step 38 logic: classify ─────────────────────────────────────────
CLASSIFY_PROMPT = """\
You are a query planner for a regulatory Q&A system backed by a
vector store of RBI circulars and master directions.

Given a user query, decide whether answering it fully requires:
  • a SINGLE retrieve() call (one topic, one time-frame, one entity lookup), OR
  • MULTIPLE independent lookups (e.g. comparing across entities like Banks vs NBFCs, comparing across time, or combining separate regulations).

Respond with ONLY a JSON object — no markdown, no explanation outside
the JSON:
{"needs_decomposition": true/false, "reasoning": "..."}
"""


def classify_query(query: str) -> dict:
    """Return {"needs_decomposition": bool, "reasoning": str}."""
    return _groq_json_call(CLASSIFY_PROMPT, query)


# ── Step 39 logic: decompose ────────────────────────────────────────
DECOMPOSE_PROMPT = """\
You are a precision query planner for an RBI regulatory retrieval system.

The user's query requires more than one independent lookup. Decompose it into 2-3 independent, factual, concise sub-queries.

Rules:
  • Keep sub-queries clean and factual (e.g. "What are the KYC requirements for Commercial Banks?", "What are the KYC requirements for NBFCs?").
  • Do NOT add conversational boilerplate or meta-phrasing like "according to RBI circulars and master directions" or "search for...".
  • Each sub-query must be fully self-contained (no pronouns like "it" or "the same").
  • Return EXACTLY a JSON object — no markdown, no extra text:
    {"sub_queries": ["...", "..."]}
"""


def decompose_query(query: str) -> list[str]:
    """Return a list of 2-3 independent sub-queries."""
    result = _groq_json_call(DECOMPOSE_PROMPT, query)
    return result.get("sub_queries", [query])


# ── Step 41 logic: merge citations + synthesize ─────────────────────
SYNTHESIZE_PROMPT = """\
You are a strict regulatory summarizer. Your ONLY task is to combine and rephrase the provided VERIFIED SUPPORTED CLAIMS into fluent, concise prose.

ABSOLUTE NEGATIVE CONSTRAINTS (ZERO TOLERANCE):
1. DO NOT USE ANY EXTERNAL KNOWLEDGE OR PARAMETRIC MEMORY.
2. DO NOT ADD ANY NEW FACTS, RULES, PERCENTAGES, PORTAL NAMES (e.g., Sachet, CMS, Complaint Management), CHANNELS (e.g., Digital Banking Units/DBUs, Business Correspondents, Doorstep Banking), REQUIREMENTS, OR DETAILS that are not explicitly stated in the bulleted claims list.
3. Every sentence in your output MUST strictly derive from and map to at least one bullet in the "Verified Supported Claims" list.
4. If only 2-3 brief claims are provided, your output must be equally concise and contain ONLY those 2-3 facts.
5. If abstained topics are listed, state clearly: "Regarding [topic], the available regulatory corpus does not contain grounded information on this."

Respond with ONLY a JSON object:
{
  "synthesized_answer": "<concise synthesis containing ONLY facts from the provided verified claims>",
  "answerable": true,
  "reasoning": "<brief explanation>"
}
"""


def _merge_citations(sub_results: list[dict]) -> list[dict]:
    """Collect all citations across sub-answers, deduplicated by clause_id.

    Only processes sub-results that are already filtered to status == 'answered'.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for sr in sub_results:
        for cite in sr.get("citations", []):
            cid = cite.get("cited_clause_id", cite.get("clause_id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(cite)
    return merged


def _synthesize(
    original_query: str,
    answered_results: list[dict],
    abstained_topics: list[str],
) -> dict:
    """Ask Groq to synthesize only the verified supported claims.

    Abstained topics are listed separately so the LLM can state the
    gap explicitly without being tempted by their (empty) content.
    """
    # Build the answered section using verified claims only
    answered_text = ""
    for i, sr in enumerate(answered_results, 1):
        claims = sr.get("citations", [])
        claims_list_text = "\n".join(
            f"    • {c.get('text', '')}"
            for c in claims
            if isinstance(c, dict) and c.get("text")
        )
        if not claims_list_text:
            claims_list_text = "    • (No specific claim text)"
        answered_text += (
            f"Answered Sub-query {i}: {sr['sub_query']}\n"
            f"  Verified Supported Claims ({len(claims)}):\n"
            f"{claims_list_text}\n\n"
        )

    # Build the abstained section
    if abstained_topics:
        abstained_text = "Abstained topics (NO grounded evidence found):\n"
        for topic in abstained_topics:
            abstained_text += f"  - {topic}\n"
    else:
        abstained_text = "Abstained topics: none\n"

    user_prompt = (
        f"Original user question: {original_query}\n\n"
        f"{answered_text}"
        f"{abstained_text}\n"
        "INSTRUCTION: Combine ONLY the facts from the Verified Supported Claims above into a cohesive answer. "
        "Do NOT add any extra facts, channels, portals, or rules."
    )

    return _groq_json_call(SYNTHESIZE_PROMPT, user_prompt)


def _filter_ungrounded_sentences(synthesized_text: str, citations: list[dict]) -> str:
    """Ensure every sentence in synthesized_text is backed by at least one citation."""
    if not synthesized_text or not citations:
        return synthesized_text

    from generation.verify import _split_into_sentences, _get_nli_model
    sentences = _split_into_sentences(synthesized_text)
    if not sentences:
        return synthesized_text

    citation_texts = [
        c.get("evidence_sentence", c.get("text", ""))
        for c in citations
        if isinstance(c, dict)
    ]
    if not citation_texts:
        return synthesized_text

    citation_blob = " ".join(citation_texts).lower()
    nli_model = _get_nli_model()
    valid_sentences = []

    for sent in sentences:
        # Preserve explicit abstention notices
        if "available regulatory corpus does not contain" in sent.lower():
            valid_sentences.append(sent)
            continue

        sent_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sent.lower()))
        if not sent_words:
            continue
        overlap = len(sent_words.intersection(set(re.findall(r"\b[a-zA-Z]{4,}\b", citation_blob)))) / len(sent_words)

        if overlap >= 0.50:
            valid_sentences.append(sent)
        else:
            pairs = [[ct, sent] for ct in citation_texts]
            if pairs:
                scores = nli_model.predict(pairs)
                best_entailment = max(s[0] for s in scores)
                if best_entailment > 0.0:
                    valid_sentences.append(sent)

    return " ".join(valid_sentences) if valid_sentences else synthesized_text


# ── Public API ──────────────────────────────────────────────────────
def plan_and_answer(query: str) -> dict:
    """
    End-to-end query planner + answerer.

    1. Classify whether the query needs decomposition.
    2a. If single-hop → call answer_query() directly.
    2b. If multi-hop  → decompose → fan-out answer_query() per sub-query
        → synthesize into one final answer with merged citations.

    Returns
    -------
    dict with keys:
        answer          – the final answer text
        citations       – list of grounded citations
        confidence      – float (percentage)
        status          – "answered" | "abstain"
        was_decomposed  – bool
    """
    # ── 1. Classify ─────────────────────────────────────────────────
    classification = classify_query(query)
    if "_error" in classification:
        return classification
    needs_decomp = classification.get("needs_decomposition", False)

    # ── 2a. Single-hop: answer directly ─────────────────────────────
    if not needs_decomp:
        time.sleep(RATE_LIMIT_SLEEP)  # respect rate limit after classify call
        result = answer_query(query)
        if "_error" in result:
            return result
        return {
            "answer": result.get("answer", result.get("reason", "")),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
            "status": result.get("status", "abstain"),
            "was_decomposed": False,
        }

    # ── 2b. Multi-hop: decompose → fan-out → synthesize ─────────────
    time.sleep(RATE_LIMIT_SLEEP)
    sub_queries = decompose_query(query)

    # Fan-out: run each sub-query through the full pipeline
    sub_results: list[dict] = []
    for i, sq in enumerate(sub_queries):
        if i > 0:
            time.sleep(RATE_LIMIT_SLEEP)

        raw = answer_query(sq)
        if "_error" in raw:
            return raw
        sub_results.append({
            "sub_query": sq,
            "status": raw.get("status", "abstain"),
            "answer": raw.get("answer", raw.get("reason", "")),
            "citations": raw.get("citations", []),
            "confidence": raw.get("confidence", 0.0),
        })

    # ── Separate answered vs abstained sub-results ──────────────────
    answered = [sr for sr in sub_results if sr["status"] == "answered"]
    abstained = [sr for sr in sub_results if sr["status"] != "answered"]
    abstained_topics = [sr["sub_query"] for sr in abstained]

    # ── All abstained → return abstain immediately, no synthesis ────
    if not answered:
        return {
            "answer": (
                "The available regulatory corpus does not contain "
                "grounded information to answer this question."
            ),
            "citations": [],
            "confidence": 0.0,
            "status": "abstain",
            "was_decomposed": True,
        }

    # ── Merge citations only from answered sub-results ──────────────
    merged_citations = _merge_citations(answered)

    # ── Confidence from answered sub-results only ───────────────────
    confidences = [sr["confidence"] for sr in answered]
    avg_confidence = sum(confidences) / len(confidences)

    # ── Synthesize (pass only answered content + abstained topics) ──
    time.sleep(RATE_LIMIT_SLEEP)
    synth = _synthesize(query, answered, abstained_topics)
    if "_error" in synth:
        return synth

    synth_answer = synth.get("synthesized_answer", "")
    synth_answer = _filter_ungrounded_sentences(synth_answer, merged_citations)

    return {
        "answer": synth_answer,
        "citations": merged_citations,
        "confidence": avg_confidence,
        "status": "answered",  # at least one sub-query was answered
        "was_decomposed": True,
    }


# ── CLI ─────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query_planner/plan_and_answer.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')
    print("Running plan_and_answer() …\n")

    result = plan_and_answer(query)

    print("── Final result ──────────────────────────────────────")
    print(json.dumps(result, indent=2, default=str))
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
