"""
answer_query.py
Single entry-point that combines retrieval → generation → verification.

Usage as a module:
    from generation.answer_query import answer_query
    result = answer_query("What are the KYC requirements for NBFCs?")

Usage from the command line:
    python generation/answer_query.py "your query here"
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation.generate import generate          # noqa: E402
from generation.verify import (                    # noqa: E402
    compute_confidence,
    finalize_response,
    verify_claims,
)
from retrieval.retrieve import retrieve            # noqa: E402


# ── Current-rate safety guard ───────────────────────────────────────
# Some RBI Directions set a statutory *ceiling* and delegate the actual
# operative percentage to a separate Gazette notification.  This applies to
# any rate/ratio requirement, not only SLR or CRR.
_OPERATIVE_VALUE_REQUEST_RE = re.compile(
    r"\b(?:how\s+much|rate|ratio|percentage|requirement|minimum|applicable|"
    r"current(?:ly)?|prevailing|prescribed|effective)\b",
    re.IGNORECASE,
)
_CEILING_INTENT_RE = re.compile(
    r"\b(?:statutory\s+ceiling|upper\s+limit|maximum|not\s+exceeding)\b",
    re.IGNORECASE,
)
_STATUTORY_CEILING_RE = re.compile(
    r"not\s+exceeding\s+\d+(?:\.\d+)?\s*(?:per\s+cent|percent|%)"
    r"[\s\S]{0,240}?(?:notification|official\s+gazette|specify\s+from\s+time\s+to\s+time)",
    re.IGNORECASE,
)
_OPERATIVE_RATE_RE = re.compile(
    r"(?:currently|at\s+present|with\s+effect\s+from|prescribed\s+(?:at|as)|"
    r"applicable\s+(?:(?:rate|ratio|percentage|requirement)\s+)?(?:is|of|at)|"
    r"(?:rate|ratio|percentage|requirement)\s+(?:is|of|at)|"
    r"(?:shall|must)\s+maintain[\s\S]{0,80}?\b(?:at|of))"
    r"\D{0,45}\d+(?:\.\d+)?\s*(?:per\s+cent|percent|%|basis\s+points|bps)",
    re.IGNORECASE,
)


def _requires_current_rate(query: str) -> bool:
    """Whether a question asks for an operative regulatory value, not a ceiling."""
    return bool(
        _OPERATIVE_VALUE_REQUEST_RE.search(query)
        and not _CEILING_INTENT_RE.search(query)
    )


def _has_only_statutory_ceiling(chunks: list[dict]) -> bool:
    """True when retrieved sources delegate the rate but state no operative one."""
    source_texts = [str(chunk.get("clause_text", "")) for chunk in chunks]
    return (
        any(_STATUTORY_CEILING_RE.search(text) for text in source_texts)
        and not any(_OPERATIVE_RATE_RE.search(text) for text in source_texts)
    )


def _current_rate_abstention() -> dict:
    """Return a transparent result without inventing an operative RBI rate."""
    return {
        "status": "abstain",
        "reason": (
            "The indexed RBI Directions establish a statutory ceiling, but do not "
            "include the RBI notification that specifies the currently applicable "
            "rate. No current rate has been asserted."
        ),
        "confidence": 0.0,
    }


# ── Public API ──────────────────────────────────────────────────────
def answer_query(query: str, category: str | None = None) -> dict:
    """
    End-to-end pipeline: retrieve → generate → verify → finalize.

    Returns a dict with keys:
        status      – "answered" | "abstain"
        answer      – the generated answer (only when status == "answered")
        citations   – list of supported claims (only when status == "answered")
        confidence  – percentage of claims that passed verification
        reason      – explanation (only when status == "abstain")
    """
    # 1. Retrieve top chunks with optional category pre-filtering
    chunks = retrieve(query, category=category)

    # Stop before generation when the evidence only supplies a statutory
    # maximum and delegates the real rate to a separate notification.
    if _requires_current_rate(query) and _has_only_statutory_ceiling(chunks):
        return _current_rate_abstention()

    # Clause labels such as "A." are reused across different RBI documents.
    # Give each retrieved source a local, unambiguous ID for the model and
    # verifier, while retaining its readable document/clause metadata.
    chunks_by_clause_id: dict[str, str] = {}
    citation_sources: dict[str, dict] = {}
    for index, chunk in enumerate(chunks, start=1):
        citation_id = f"C{index:02d}"
        chunks_by_clause_id[citation_id] = chunk["clause_text"]
        citation_sources[citation_id] = {
            "doc_id": chunk.get("doc_id", ""),
            "category": chunk.get("category", ""),
            "source_clause_id": chunk.get("clause_id", ""),
        }

    # 2. Generate grounded answer + claims
    # Reuse these same chunks. A second retrieval could return a different
    # ordering or document set and break claim-to-source verification.
    gen_result = generate(query, chunks=chunks)
    if "_error" in gen_result:
        return {"_error": gen_result["_error"]}

    # 3. Verify each claim against its cited chunk
    verified = verify_claims(gen_result["claims"], chunks_by_clause_id)

    # Enrich verified claims for the UI without exposing the internal C01/C02
    # lookup IDs as if they were actual regulatory clause labels.
    for claim in verified:
        source = citation_sources.get(claim.get("cited_clause_id", ""), {})
        claim.update(source)

    # 4. Finalize: keep only supported claims, decide answered/abstain
    return finalize_response(gen_result["answer"], verified)


# ── CLI ─────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python generation/answer_query.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')
    print("Running pipeline: retrieve → generate → verify → finalize …\n")

    result = answer_query(query)

    print("── Final response ────────────────────────────────────")
    print(json.dumps(result, indent=2))
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
