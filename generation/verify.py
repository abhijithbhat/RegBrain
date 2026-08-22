"""
verify.py
Claim-level grounding verifier.

Given the claims produced by generate() and the retrieved chunks, check
whether each claim is actually supported by its cited chunk.

Three checks must ALL pass for a claim to be marked "supported":
  1. SequenceMatcher similarity  ≥ 0.70
  2. Keyword-overlap ratio       ≥ 0.70
  3. Number match — every number in the claim text must appear
     in the cited chunk's text

Usage as a module:
    from generation.verify import verify_claims
    tagged = verify_claims(claims, chunks_by_clause_id)

Usage from the command line:
    python generation/verify.py "your query here"
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generation.generate import generate  # noqa: E402
from retrieval.retrieve import retrieve   # noqa: E402

# ── Thresholds ──────────────────────────────────────────────────────
SEQ_THRESHOLD = 0.35   # sequence similarity gate
KW_THRESHOLD  = 0.60   # keyword overlap gate

# ── Sentence splitting & NLI ────────────────────────────────────────
NLI_MODEL_NAME = os.getenv("NLI_MODEL", "shared")
_NLI_MODEL = None


def _get_nli_model():
    """Return neural cross-encoder instance for claim verification.
    
    If NLI_MODEL is 'shared' or unspecified, reuses the singleton reranker
    instance from retrieval.retrieve (_get_reranker()) to avoid loading extra
    weights into memory while enforcing genuine neural relevance scoring.
    """
    global _NLI_MODEL
    if _NLI_MODEL is None:
        if NLI_MODEL_NAME not in ("shared", "reranker", "disabled", ""):
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
                _NLI_MODEL = TextCrossEncoder(NLI_MODEL_NAME, cache_dir=FASTEMBED_CACHE, threads=1)
            except Exception:
                _NLI_MODEL = None

        if _NLI_MODEL is None and NLI_MODEL_NAME != "disabled":
            try:
                from retrieval.retrieve import _get_reranker
                _NLI_MODEL = _get_reranker()
            except Exception:
                _NLI_MODEL = None
    return _NLI_MODEL


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly."""
    if not text:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return sents if sents else [text]


# ── Scoring helpers (same logic as step30) ──────────────────────────
def _fuzzy_score(claim_text: str, chunk_text: str) -> float:
    """
    Best-window SequenceMatcher score.

    A short claim compared to a long chunk will always score near 0
    with a naive SequenceMatcher.ratio().  Instead we slide a window
    roughly the size of the claim across the chunk and return the
    *maximum* similarity found.  This way, if the claim's content
    genuinely appears somewhere in the chunk, the score stays high.
    """
    claim_low = claim_text.lower()
    chunk_low = chunk_text.lower()

    # If the chunk is shorter or similar in length, compare directly
    if len(chunk_low) <= len(claim_low) * 2:
        return SequenceMatcher(None, claim_low, chunk_low).ratio()

    # Sliding window: step through the chunk in overlapping windows
    window = len(claim_low) * 2   # generous window
    step = max(1, len(claim_low) // 2)
    best = 0.0

    for start in range(0, len(chunk_low) - window + 1, step):
        segment = chunk_low[start : start + window]
        score = SequenceMatcher(None, claim_low, segment).ratio()
        if score > best:
            best = score
            if best > 0.95:
                break  # early exit — good enough

    return best


def _keyword_overlap_score(claim_text: str, chunk_text: str) -> float:
    """Token-level overlap: |intersection| / |claim_tokens|."""
    tokenise = lambda t: set(re.findall(r"\b[\w.%]+\b", t.lower()))
    claim_tokens = tokenise(claim_text)
    chunk_tokens = tokenise(chunk_text)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


# ── Text-form number normalisation ──────────────────────────────────
_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "twenty-five": "25",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "seventy-five": "75", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000",
}

def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric values from text, including text-form numbers
    and Indian numbering (crore, lakh)."""
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    lower = text.lower()
    for word, digit in _WORD_TO_NUM.items():
        if word in lower:
            nums.add(digit)
    # Indian numbering: "2 crore" → 20000000, but we just need
    # to recognise the base number is present
    return nums


def _numbers_match(claim_text: str, chunk_text: str) -> bool:
    """Strict number matching: every number in the claim must appear in the chunk."""
    claim_nums = _extract_numbers(claim_text)
    chunk_nums = _extract_numbers(chunk_text)
    if not claim_nums:
        return True
    missing = claim_nums - chunk_nums
    return len(missing) == 0


def _normalise_text(text: str) -> str:
    """Normalise text for a conservative verbatim-source lookup."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _source_passages(text: str) -> list[str]:
    """Preserve numbered source passages while joining wrapped PDF lines.

    RBI extracts commonly put a clause number on one line and continue its
    sentence on the next.  A sentence splitter turns those labels into bogus
    candidates (for example, ``"A."``), which is unacceptable as visible
    evidence.  This parser keeps each numbered item and its continuation lines
    together; roman sub-items remain with their numeric parent.
    """
    passages: list[str] = []
    current: list[str] = []
    top_level = re.compile(r"^(?:[A-Z]\.|\d+\.)\s+")
    numeric_item = re.compile(r"^\(\d+\)\s+")
    roman_item = re.compile(r"^\([ivxlcdm]+\)\s+", re.IGNORECASE)

    def flush() -> None:
        if current:
            passage = re.sub(r"\s+", " ", " ".join(current)).strip()
            if passage:
                passages.append(passage)
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or re.fullmatch(r"\d{1,3}", line):
            continue

        starts_new_passage = bool(top_level.match(line) or numeric_item.match(line))
        if starts_new_passage:
            flush()
        elif roman_item.match(line) and not current:
            starts_new_passage = True

        if starts_new_passage:
            current.append(line)
        else:
            if current:
                # Repair word-wrap hyphenation without affecting ordinary hyphens.
                if current[-1].endswith("-") and line[:1].islower():
                    current[-1] = current[-1][:-1] + line
                else:
                    current.append(line)
            else:
                current.append(line)

    flush()
    return passages or _split_into_sentences(text)


def _select_evidence_sentence(claim_text: str, chunk_text: str) -> str | None:
    """Return the most directly corresponding source passage for display.

    NLI chooses the sentence with the strongest entailment probability.  That
    score is right for the verification gate, but it is not necessarily the
    sentence a reader expects to see beside a particular claim in a long
    clause.  Keep the NLI winner as telemetry and independently prefer an
    exact or closest lexical source sentence for the visible quotation.
    """
    passages = _source_passages(chunk_text)
    if not passages:
        return None

    claim_normalised = _normalise_text(claim_text)
    if not claim_normalised:
        return passages[0]

    exact_matches = [
        passage
        for passage in passages
        if len(_normalise_text(passage)) >= 24
        and (
            claim_normalised in _normalise_text(passage)
            or _normalise_text(passage) in claim_normalised
        )
    ]
    if exact_matches:
        return min(exact_matches, key=len)

    claim_numbers = _extract_numbers(claim_text)

    def source_score(passage: str) -> tuple[int, float, float]:
        passage_numbers = _extract_numbers(passage)
        numbers_complete = int(not claim_numbers or claim_numbers <= passage_numbers)
        return (
            numbers_complete,
            _keyword_overlap_score(claim_text, passage),
            _fuzzy_score(claim_text, passage),
        )

    return max(passages, key=source_score)


# ── Public API ──────────────────────────────────────────────────────
def verify_claims(
    claims: list[dict],
    chunks_by_clause_id: dict[str, str],
) -> list[dict]:
    """
    Verify each claim against its cited chunk using a 2-gate verifier:
      1. Lexical: kw >= 0.60, seq >= 0.35, strict number matching
      2. Sentence-level NLI: best_sentence_entailment > best_sentence_contradiction

    Parameters
    ----------
    claims : list[dict]
        Each dict has ``text`` and ``cited_clause_id``.
    chunks_by_clause_id : dict[str, str]
        Mapping from ``clause_id`` → ``clause_text``.

    Returns
    -------
    list[dict]
        A copy of each claim augmented with:
        - ``supported``     (bool: lexical_pass AND nli_pass)
        - ``lexical_pass``  (bool)
        - ``nli_pass``      (bool)
        - ``seq_score``     (float)
        - ``kw_score``      (float)
        - ``nums_ok``       (bool)
        - ``evidence_sentence`` (str | None) source excerpt shown with claim
        - ``reason``        (str)  short explanation when unsupported
    """
    results: list[dict] = []
    if not claims:
        return results

    # Pre-collect all (sentence, claim_text) pairs for batch NLI prediction
    all_pairs = []
    claim_sentence_map = []  # (claim_index, sentences)

    for i, claim in enumerate(claims):
        cid = claim.get("cited_clause_id", "")
        chunk_text = chunks_by_clause_id.get(cid)
        if chunk_text is not None:
            sentences = _split_into_sentences(chunk_text)
            # Filter to top candidate sentences based on keyword overlap
            claim_kw_sents = [
                s for s in sentences
                if _keyword_overlap_score(claim["text"], s) >= 0.15
            ]
            cand_sents = claim_kw_sents[:2] if claim_kw_sents else sentences[:2]
            claim_sentence_map.append((i, cand_sents))
            for s in cand_sents:
                all_pairs.append((s, claim["text"]))
        else:
            claim_sentence_map.append((i, []))

    all_scores = None
    if all_pairs:
        nli_model = _get_nli_model()
        if nli_model is not None:
            import gc
            all_scores = list(nli_model.rerank_pairs(all_pairs, batch_size=8))
            gc.collect()

    score_offset = 0

    for i, claim in enumerate(claims):
        cid = claim.get("cited_clause_id", "")
        chunk_text = chunks_by_clause_id.get(cid)

        # Claim cites a clause that wasn't retrieved
        if chunk_text is None:
            results.append({
                **claim,
                "supported": False,
                "lexical_pass": False,
                "nli_pass": False,
                "seq_score": 0.0,
                "kw_score": 0.0,
                "nums_ok": False,
                "best_sentence": None,
                "evidence_sentence": None,
                "best_entailment": -999.0,
                "best_contradiction": 999.0,
                "reason": f"cited clause '{cid}' not found in retrieved chunks",
            })
            continue

        seq = _fuzzy_score(claim["text"], chunk_text)
        kw = _keyword_overlap_score(claim["text"], chunk_text)
        nums_ok = _numbers_match(claim["text"], chunk_text)
        evidence_sentence = _select_evidence_sentence(claim["text"], chunk_text)

        # Gate 1: Lexical Pass
        lexical_pass = (kw >= KW_THRESHOLD) and (seq >= SEQ_THRESHOLD) and nums_ok

        # Gate 2: Sentence-level NLI Pass
        sentences = claim_sentence_map[i][1]
        if sentences and all_scores is not None:
            n_sents = len(sentences)
            scores = all_scores[score_offset : score_offset + n_sents]
            score_offset += n_sents

            # Handle both single-logit relevance (shared cross-encoder) and 3-class NLI
            first_elem = scores[0] if len(scores) > 0 else None
            is_multiclass = hasattr(first_elem, "__len__") and len(first_elem) >= 2

            if is_multiclass:
                # 3-class NLI [contradiction, entailment, neutral]
                best_idx, _ = max(enumerate(scores), key=lambda x: float(x[1][1]))
                best_scores = scores[best_idx]
                best_sentence = sentences[best_idx]
                best_entailment = float(best_scores[1])
                best_contradiction = float(best_scores[0])
                nli_pass = best_entailment > best_contradiction
            else:
                # Single-logit cross-encoder relevance score
                best_idx, _ = max(enumerate(scores), key=lambda x: float(x[1]))
                best_score = float(scores[best_idx])
                best_sentence = sentences[best_idx]
                best_entailment = best_score
                best_contradiction = 0.0
                nli_pass = best_score > -1.5
        else:
            best_sentence = evidence_sentence
            best_entailment = 1.0
            best_contradiction = 0.0
            nli_pass = True

        supported = lexical_pass and nli_pass

        reason = ""
        if not supported:
            parts = []
            if kw < KW_THRESHOLD:
                parts.append(f"low keyword overlap ({kw:.2f})")
            if seq < SEQ_THRESHOLD:
                parts.append(f"low sequence similarity ({seq:.2f})")
            if not nums_ok:
                claim_nums = set(re.findall(r"\d+(?:\.\d+)?", claim["text"]))
                chunk_nums = set(re.findall(r"\d+(?:\.\d+)?", chunk_text))
                missing = claim_nums - chunk_nums
                parts.append(f"numbers {missing} not in chunk")
            if not nli_pass:
                parts.append(f"NLI margin fail (ent={best_entailment:.2f} <= con={best_contradiction:.2f})")
            reason = "; ".join(parts)

        results.append({
            **claim,
            "supported": supported,
            "lexical_pass": lexical_pass,
            "nli_pass": nli_pass,
            "seq_score": round(seq, 4),
            "kw_score": round(kw, 4),
            "nums_ok": nums_ok,
            "best_sentence": best_sentence,
            "evidence_sentence": evidence_sentence,
            "best_entailment": round(best_entailment, 4),
            "best_contradiction": round(best_contradiction, 4),
            "reason": reason,
        })

    return results


def compute_confidence(verified_claims: list[dict]) -> float:
    """
    Return the percentage of claims where ``supported`` is True.

    Returns 0.0 when *verified_claims* is empty.
    """
    if not verified_claims:
        return 0.0
    return sum(1 for c in verified_claims if c["supported"]) / len(verified_claims) * 100


def finalize_response(answer: str, verified_claims: list[dict]) -> dict:
    """
    Produce the final API-style response.

    - Keeps only claims where ``supported`` is True.
    - If confidence < 33% or zero supported claims → ``"status": "abstain"``.
    - Otherwise → ``"status": "answered"`` with the answer and citations.
    """
    confidence = compute_confidence(verified_claims)
    supported = [c for c in verified_claims if c["supported"]]

    if confidence < 33.0 or len(supported) == 0:
        return {
            "status": "abstain",
            "reason": "insufficient grounded evidence",
            "confidence": confidence,
        }

    return {
        "status": "answered",
        "answer": answer,
        "citations": supported,
        "confidence": confidence,
    }


# ── CLI ─────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python generation/verify.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')

    # 1. Retrieve chunks
    print("Retrieving chunks …")
    chunks = retrieve(query)

    # Build lookup: clause_id → full clause_text
    chunks_by_clause_id: dict[str, str] = {}
    for c in chunks:
        cid = c["clause_id"]
        # If multiple chunks share a clause_id, concatenate them
        if cid in chunks_by_clause_id:
            chunks_by_clause_id[cid] += "\n" + c["clause_text"]
        else:
            chunks_by_clause_id[cid] = c["clause_text"]

    # 2. Generate grounded answer
    print("Generating answer …")
    gen_result = generate(query)

    print("\n── Answer ────────────────────────────────────────────")
    print(gen_result["answer"][:500])
    if len(gen_result["answer"]) > 500:
        print("  … (truncated)")
    print("─────────────────────────────────────────────────────\n")

    # 3. Verify each claim
    tagged = verify_claims(gen_result["claims"], chunks_by_clause_id)

    supported_count = sum(1 for t in tagged if t["supported"])
    total = len(tagged)
    confidence = compute_confidence(tagged)

    print(f"── Claim verification  ({supported_count}/{total} supported, "
          f"confidence {confidence:.1f}%) ──\n")

    for i, t in enumerate(tagged, start=1):
        icon = "✅" if t["supported"] else "❌"
        print(f"  {i}. {icon}  [{t['cited_clause_id']}]  {t['text']}")
        print(f"       lexical={'✓' if t['lexical_pass'] else '✗'} (seq={t['seq_score']:.4f}, kw={t['kw_score']:.4f}, nums={'✓' if t['nums_ok'] else '✗'})")
        print(f"       nli={'✓' if t['nli_pass'] else '✗'} (ent={t.get('best_entailment', 0.0):.4f}, con={t.get('best_contradiction', 0.0):.4f})")
        if t["reason"]:
            print(f"       reason: {t['reason']}")
        print()

    # 4. Finalize
    final = finalize_response(gen_result["answer"], tagged)

    print("── Finalized response ────────────────────────────────")
    print(json.dumps(final, indent=2))
    print("─────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
