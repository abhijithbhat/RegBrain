"""
eval/ragas_groq_utils.py – Resilient RAGAS + Groq integration layer.

Provides four hardening features for running RAGAS with a Groq judge:

  1. **Markdown fence stripping** – Monkey-patches the OpenAI client so
     ```json … ``` wrappers are stripped before RAGAS / instructor parse.
  2. **Exponential-backoff retry** – Wraps evaluate() calls; on transient
     API failures retries up to 3× with 2 s → 4 s → 8 s backoff.
  3. **Per-question failure logging** – After evaluation, any sample that
     still has NaN scores is retried individually.  Samples that fail all
     retries are logged by question text so you can rerun them later.
  4. **Disk-based caching** – Uses RAGAS's DiskCacheBackend so successful
     LLM judge calls are cached to eval/.ragas_cache/ and survive reruns.

Usage from other eval scripts:

    from ragas_groq_utils import make_groq_judge, make_embeddings, resilient_evaluate
    llm = make_groq_judge()
    emb = make_embeddings()
    result = resilient_evaluate(dataset, metrics, llm, emb)
"""

import logging
import math
import os
import re
import sys
import time
import warnings
from typing import Any, List, Optional, Sequence

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Module-level logger ────────────────────────────────────────────────
logger = logging.getLogger("ragas_groq")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ── Environment ────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".ragas_cache")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1.  Markdown fence stripping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing ``json … `` wrappers if present."""
    if not text:
        return text
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _patch_client_strip_fences(client):
    """
    Monkey-patch an OpenAI client so every chat-completion response has
    markdown fences stripped from message.content *before* RAGAS /
    instructor attempts to parse the JSON.
    """
    original_create = client.chat.completions.create

    def _patched_create(*args, **kwargs):
        kwargs["max_tokens"] = 4096
        response = original_create(*args, **kwargs)
        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                raw = choice.message.content
                if raw:
                    cleaned = strip_markdown_fences(raw)
                    if cleaned != raw:
                        logger.debug("Stripped markdown fences from Groq response")
                    choice.message.content = cleaned
        return response

    client.chat.completions.create = _patched_create
    return client


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2.  Disk cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_disk_cache(cache_dir: str = CACHE_DIR):
    """Return a RAGAS DiskCacheBackend, creating the directory if needed."""
    from ragas.cache import DiskCacheBackend

    os.makedirs(cache_dir, exist_ok=True)
    return DiskCacheBackend(cache_dir=cache_dir)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3.  LLM + Embeddings factories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_groq_judge(cache=None):
    """
    Create a RAGAS-ready Groq judge LLM with:
      • Fence-stripping OpenAI client
      • Disk-based response caching (by default)

    Returns an InstructorBaseRagasLLM usable with ragas.evaluate().
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    if not GROQ_API_KEY:
        sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    client = _patch_client_strip_fences(client)

    if cache is None:
        cache = get_disk_cache()

    return llm_factory(MODEL, provider="openai", client=client, cache=cache)


def make_embeddings():
    """
    Create local HuggingFace embeddings (BAAI/bge-small-en-v1.5) wrapped
    for RAGAS's MetricWithEmbeddings interface.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper

    try:
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings as LCHFEmbed,
        )
    except ImportError:
        sys.exit(
            "ERROR: langchain-community HuggingFaceEmbeddings not available.\n"
            "  pip install langchain-community sentence-transformers"
        )

    lc_emb = LCHFEmbed(model_name="BAAI/bge-small-en-v1.5")
    return LangchainEmbeddingsWrapper(lc_emb)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4.  Resilient evaluate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resilient_evaluate(
    dataset,
    metrics,
    llm,
    embeddings,
    max_retries: int = 3,
    base_delay: float = 2.0,
    show_progress: bool = True,
):
    """
    Run ragas.evaluate() with full resilience:

      Phase 1 – Run the batch evaluation.  If the call itself raises an
                exception (rate-limit, timeout, etc.) retry with
                exponential backoff up to *max_retries* times.

      Phase 2 – Scan the results for NaN scores.  For each sample that
                has any NaN metric, retry that individual sample up to
                *max_retries* more times.  Log (never silently swallow)
                any sample that still fails.

    Returns the (potentially patched) EvaluationResult.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.utils import safe_nanmean

    metric_names = [m.name for m in metrics]

    # ── Phase 1: batch evaluate with top-level retry ───────────────────
    result = None
    for attempt in range(1, max_retries + 1):
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
                show_progress=show_progress,
            )
            break
        except Exception as e:
            delay = base_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning(
                    "evaluate() attempt %d/%d failed: %s  — retrying in %.0fs…",
                    attempt, max_retries, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "evaluate() failed after %d attempts: %s",
                    max_retries, e,
                )
                raise

    if result is None:
        raise RuntimeError("evaluate() returned None unexpectedly")

    # ── Phase 2: per-sample NaN retry ──────────────────────────────────
    def _has_nan(scores_dict):
        return any(
            isinstance(scores_dict.get(m, float("nan")), float)
            and math.isnan(scores_dict.get(m, float("nan")))
            for m in metric_names
        )

    failed_indices = [
        i for i, row in enumerate(result.scores) if _has_nan(row)
    ]

    if not failed_indices:
        logger.info("All %d sample(s) scored successfully on first pass.", len(result.scores))
        return result

    logger.info(
        "NaN scores detected in %d sample(s) (indices %s). "
        "Retrying individually…",
        len(failed_indices), failed_indices,
    )

    samples = list(dataset)
    permanently_failed: List[int] = []

    for idx in failed_indices:
        sample = samples[idx]
        q_preview = (sample.user_input or "?")[:80]
        success = False

        for attempt in range(1, max_retries + 1):
            delay = base_delay * (2 ** (attempt - 1))
            logger.info(
                "  ↻ Retry %d/%d for sample %d: '%s…' (wait %.0fs)",
                attempt, max_retries, idx, q_preview, delay,
            )
            time.sleep(delay)

            try:
                mini = EvaluationDataset(samples=[sample])
                mini_result = evaluate(
                    dataset=mini,
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings,
                    raise_exceptions=False,
                    show_progress=False,
                )

                retry_scores = mini_result.scores[0]
                if not _has_nan(retry_scores):
                    result.scores[idx] = retry_scores
                    success = True
                    logger.info("  ✓ Sample %d succeeded on retry %d", idx, attempt)
                    break
                else:
                    nan_metrics = [
                        m for m in metric_names
                        if isinstance(retry_scores.get(m), float)
                        and math.isnan(retry_scores.get(m))
                    ]
                    logger.warning(
                        "  Sample %d retry %d still NaN on: %s",
                        idx, attempt, nan_metrics,
                    )
            except Exception as e:
                logger.warning(
                    "  Sample %d retry %d raised exception: %s",
                    idx, attempt, e,
                )

        if not success:
            permanently_failed.append(idx)
            logger.error(
                "  ✗ PERMANENTLY FAILED (all %d retries) — sample %d: '%s…'",
                max_retries, idx, q_preview,
            )

    # ── Rebuild aggregated score dicts after patching ──────────────────
    result._scores_dict = {
        k: [d.get(k, float("nan")) for d in result.scores]
        for k in result.scores[0].keys()
    }
    result._repr_dict = {
        name: safe_nanmean(result._scores_dict[name])
        for name in result._scores_dict
    }

    if permanently_failed:
        logger.error(
            "SUMMARY: %d sample(s) failed all retries: indices %s",
            len(permanently_failed), permanently_failed,
        )
    else:
        logger.info("All samples recovered after per-sample retries.")

    return result
