"""
eval/naive_baseline.py – Naive RAG baseline (dense-only retrieval, no verification).

naive_retrieve : dense vector search only (top-5 cosine) — no BM25, no fusion,
                 no cross-encoder reranker.
naive_generate : same Groq LLM call as generate(), but returns the raw answer
                 with no claim verification and no abstention logic.

Usage:
    python eval/naive_baseline.py "What are the KYC requirements for NBFCs?"
"""

import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client import QdrantClient  # noqa: E402

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "regbrain"
TOP_K = 5

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

QDRANT_URL = (os.getenv("QDRANT_CLUSTER_ENDPOINT") or "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip()

MAX_RETRIES = 5
BASE_DELAY = 5.0

_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)

SYSTEM_PROMPT = """\
You are a regulatory compliance assistant.
Answer the user's question using ONLY the provided context chunks.
Do NOT use any prior knowledge.

Respond with valid JSON in exactly this shape (no markdown fences):
{
  "answer": "Your answer here, citing relevant clause IDs in brackets.",
  "cited_clause_ids": ["clause_id_1", "clause_id_2"]
}
"""

# ── Lazy-loaded singletons ──────────────────────────────────────────────
_embed_model = None
_qdrant_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30
        )
    return _qdrant_client


# ── naive_retrieve ──────────────────────────────────────────────────────
def naive_retrieve(query: str) -> list[dict]:
    """
    Dense vector search only against the "regbrain" Qdrant collection.

    Top-5 by cosine similarity — no BM25, no RRF fusion, no cross-encoder
    reranker.

    Returns a list of dicts with keys:
        doc_id, clause_id, category, effective_date, clause_text, score
    """
    model = _get_embed_model()
    client = _get_qdrant_client()
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
    )

    chunks = []
    for hit in results.points:
        payload = hit.payload if hit.payload is not None else {}
        chunks.append({
            "doc_id": payload.get("doc_id", ""),
            "clause_id": payload.get("clause_id", ""),
            "category": payload.get("category", ""),
            "effective_date": payload.get("effective_date", ""),
            "clause_text": payload.get("clause_text", ""),
            "score": float(hit.score) if hit.score is not None else 0.0,
        })

    return chunks


# ── naive_generate ──────────────────────────────────────────────────────
def naive_generate(query: str, chunks: list[dict]) -> dict:
    """
    Same Groq LLM call as generate(), but returns whatever the LLM says
    directly — no citation verifier, no abstention logic.

    Returns:
        {"answer": "...", "citations": [...clause_ids...],
         "confidence": None, "status": "answered"}
    """
    if not GROQ_API_KEY:
        sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

    context_for_prompt = [
        {
            "clause_id": c["clause_id"],
            "clause_text": c["clause_text"][:2500]
            + ("..." if len(c["clause_text"]) > 2500 else ""),
        }
        for c in chunks
    ]

    user_prompt = (
        f"Context chunks:\n{json.dumps(context_for_prompt, indent=2)}\n\n"
        f"Question: {query}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Call Groq with retry on rate-limit & server error
    raw_text = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                raw_text = response.json()["choices"][0]["message"]["content"]
                break
            elif response.status_code in (429, 500, 502, 503, 504):
                delay = BASE_DELAY * (2 ** (attempt - 1))
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    continue
                else:
                    return {
                        "answer": f"Groq API error [{response.status_code}] after {MAX_RETRIES} retries",
                        "citations": [],
                        "confidence": None,
                        "status": "error",
                    }
            else:
                return {
                    "answer": f"Groq API error [{response.status_code}]: {response.text}",
                    "citations": [],
                    "confidence": None,
                    "status": "error",
                }
        except Exception as err:
            delay = BASE_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                time.sleep(delay)
                continue
            return {
                "answer": f"Groq connection exception: {err}",
                "citations": [],
                "confidence": None,
                "status": "error",
            }

    if raw_text is None:
        return {
            "answer": "No response from Groq API",
            "citations": [],
            "confidence": None,
            "status": "error",
        }

    # Parse JSON (strip markdown fences if present)
    cleaned = raw_text.strip()
    fence_match = _FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Return raw text as-is if JSON parsing fails
        return {
            "answer": raw_text,
            "citations": [c["clause_id"] for c in chunks],
            "confidence": None,
            "status": "answered",
        }

    answer = result.get("answer", raw_text)
    cited_ids = result.get("cited_clause_ids", [])

    return {
        "answer": answer,
        "citations": cited_ids,
        "confidence": None,
        "status": "answered",
    }


# ── CLI ─────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python eval/naive_baseline.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f'Query: "{query}"\n')

    # Step 1: Naive retrieve
    print("── Naive Retrieve (dense-only, top-5) ───────────────────")
    chunks = naive_retrieve(query)
    for i, c in enumerate(chunks, 1):
        text_preview = c["clause_text"][:120].replace("\n", " ")
        print(f"  {i}. [{c['clause_id']}] (score={c['score']:.4f}) {text_preview}")
    print()

    # Step 2: Naive generate
    print("── Naive Generate (no verification) ────────────────────")
    result = naive_generate(query, chunks)
    print(f"  Status    : {result['status']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Citations : {result['citations']}")
    print(f"  Answer    : {result['answer'][:300]}")
    print()

    # Full JSON output
    print("── Full JSON Output ────────────────────────────────────")
    print(json.dumps(result, indent=2, default=str))
    print()


if __name__ == "__main__":
    main()
