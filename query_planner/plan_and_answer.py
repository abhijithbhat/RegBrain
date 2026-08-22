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

import concurrent.futures
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

from generation.groq_client import groq_json_completion  # noqa: E402

RATE_LIMIT_SLEEP = 0.2  # seconds between internal steps


def extract_category_scope(query: str) -> str:
    """Classify the query into one of: 'Commercial_Banks', 'NBFC', 'NBFC_HFC', 'UCB', 'cross', or 'general'."""
    q_lower = query.lower().strip()

    entities_present = set()
    if any(k in q_lower for k in ["commercial bank", "commercial banks", "scheduled commercial bank", "scbs", "scb"]):
        entities_present.add("Commercial_Banks")
    elif "bank" in q_lower or "banks" in q_lower:
        if not any(k in q_lower for k in ["cooperative", "co-operative", "ucb", "small finance", "payments"]):
            entities_present.add("Commercial_Banks")

    if any(k in q_lower for k in ["hfc", "hfcs", "housing finance"]):
        entities_present.add("NBFC_HFC")
    elif any(k in q_lower for k in ["nbfc", "nbfcs", "non-banking", "non banking"]):
        entities_present.add("NBFC")

    if any(k in q_lower for k in ["ucb", "ucbs", "urban cooperative", "urban co-operative", "cooperative bank", "co-operative bank", "cooperative banks", "co-operative banks"]):
        entities_present.add("UCB")

    # If comparison words or multiple entity categories mentioned:
    if len(entities_present) > 1 or any(p in q_lower for p in ["compare", "comparative", " vs ", " versus ", "difference between", "differences between"]):
        return "cross"

    if len(entities_present) == 1:
        return list(entities_present)[0]

    return "general"


# ── Step 38 logic: classify ─────────────────────────────────────────
CLASSIFY_PROMPT = """\
You are a query planner for an RBI regulatory Q&A system.
Available regulatory categories:
- Commercial_Banks
- NBFC
- NBFC_HFC (Housing Finance Companies)
- UCB (Urban Co-operative Banks)

Given a user query, decide whether answering it fully requires:
  • a SINGLE lookup (one topic, one entity category), OR
  • MULTIPLE independent lookups (comparing across categories like Banks vs NBFCs vs UCBs, comparing across time, or combining separate regulations).

Respond with ONLY a JSON object:
{"needs_decomposition": true/false, "category": "Commercial_Banks"|"NBFC"|"NBFC_HFC"|"UCB"|"cross"|"general", "reasoning": "..."}
"""


def classify_query(query: str) -> dict:
    """Return {"needs_decomposition": bool, "category": str, "reasoning": str}."""
    q_lower = query.lower().strip()
    cat_scope = extract_category_scope(query)

    # Fast-path comparative check across entities
    if cat_scope == "cross" or any(q_lower.startswith(p) or f" {p}" in q_lower for p in ["compare", "comparative", " vs ", " versus ", "difference between", "differences between"]):
        return {"needs_decomposition": True, "category": "cross", "reasoning": "Comparative inquiry across distinct regulated entities."}

    # Fast-path multi-topic combination ("X requirements and Y rules")
    if " and " in q_lower and any(kw in q_lower for kw in ["dividend", "digital lending", "cryptocurrency", "crypto", "ombudsman", "capital adequacy", "kyc"]):
        return {"needs_decomposition": True, "category": cat_scope, "reasoning": "Multi-topic cross-regulation inquiry."}

    # Single-entity specific fast-path
    if cat_scope in ("Commercial_Banks", "NBFC", "NBFC_HFC", "UCB"):
        return {"needs_decomposition": False, "category": cat_scope, "reasoning": f"Single entity lookup for {cat_scope}."}

    result = groq_json_completion(CLASSIFY_PROMPT, query)
    if "_error" in result:
        return {"needs_decomposition": False, "category": cat_scope, "reasoning": "Fallback to single-hop."}

    result["category"] = result.get("category") or cat_scope
    return result


# ── Step 39 logic: decompose ────────────────────────────────────────
DECOMPOSE_PROMPT = """\
You are a precision query planner for an RBI regulatory retrieval system covering Commercial Banks, NBFCs, Housing Finance Companies (NBFC-HFC), and Urban Co-operative Banks (UCB).

The user's query requires more than one independent lookup. Decompose it into 2-3 independent, factual, concise sub-queries.

Rules:
  • Keep sub-queries clean, direct, and factual (e.g. "What are the capital adequacy requirements for Commercial Banks?", "What are the capital adequacy requirements for Urban Co-operative Banks?").
  • Do NOT add conversational boilerplate, meta-phrasing, or suffix tags like "under RBI guidelines", "according to RBI circulars", "search for...", or "in India".
  • Each sub-query must be fully self-contained (no pronouns like "it" or "the same").
  • Return EXACTLY a JSON object — no markdown, no extra text:
    {"sub_queries": ["...", "..."]}
"""


def decompose_query(query: str) -> list[str]:
    """Return a list of 2-3 independent sub-queries."""
    result = groq_json_completion(DECOMPOSE_PROMPT, query)
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

    return groq_json_completion(SYNTHESIZE_PROMPT, user_prompt)


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

        if overlap >= 0.35:
            valid_sentences.append(sent)
        elif nli_model is not None and hasattr(nli_model, "rerank_pairs"):
            pairs = [[ct, sent] for ct in citation_texts]
            if pairs:
                scores = list(nli_model.rerank_pairs(pairs, batch_size=8))
                if scores and max(scores) > -1.5:
                    valid_sentences.append(sent)
        elif overlap >= 0.20:
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
    category = classification.get("category")

    # ── 2a. Single-hop: answer directly ─────────────────────────────
    if not needs_decomp:
        result = answer_query(query, category=category)
        if "_error" in result:
            return result
        return {
            "answer": result.get("answer", result.get("reason", "")),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
            "status": result.get("status", "abstain"),
            "category": category,
            "was_decomposed": False,
        }

    # ── 2b. Multi-hop: decompose → parallel fan-out → synthesize ────
    sub_queries = decompose_query(query)

    # Fan-out: parallelize sub-query retrieval and processing across threads
    def _execute_sub_query(sq: str) -> dict:
        sq_cat = extract_category_scope(sq)
        raw = answer_query(sq, category=sq_cat)
        return {
            "sub_query": sq,
            "category": sq_cat,
            "status": raw.get("status", "abstain"),
            "answer": raw.get("answer", raw.get("reason", "")),
            "citations": raw.get("citations", []),
            "confidence": raw.get("confidence", 0.0),
            "_error": raw.get("_error"),
        }

    max_workers = min(3, max(1, len(sub_queries)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        sub_results = list(executor.map(_execute_sub_query, sub_queries))

    for sr in sub_results:
        if sr.get("_error"):
            return {"_error": sr["_error"]}

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
