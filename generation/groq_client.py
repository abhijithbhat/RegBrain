"""
groq_client.py
Robust Groq LLM client with multi-model failover, adaptive pacing,
token-reset aware rate-limit backoff, and JSON parsing.
"""

import json
import logging
import os
import re
import sys
import threading
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Active models with available token quotas (excluding exhausted 120b)
GROQ_MODELS = [
    os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    "qwen/qwen3.6-27b",
    "allam-2-7b",
    "openai/gpt-oss-safeguard-20b",
]

_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)

_CALL_LOCK = threading.Lock()
_LAST_CALL_TIME = 0.0
MIN_CALL_INTERVAL = 0.30  # seconds between calls to prevent RPM/TPM bursts


def _rate_limit_wait():
    global _LAST_CALL_TIME
    with _CALL_LOCK:
        now = time.time()
        elapsed = now - _LAST_CALL_TIME
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)
        _LAST_CALL_TIME = time.time()


def _parse_wait_time(response: requests.Response) -> float:
    # 1. Retry-After header
    retry_header = response.headers.get("retry-after")
    if retry_header:
        try:
            return min(float(retry_header), 6.0)
        except ValueError:
            pass
    # 2. X-RateLimit-Reset-Tokens header
    reset_tokens = response.headers.get("x-ratelimit-reset-tokens")
    if reset_tokens:
        try:
            if "ms" in reset_tokens:
                return float(reset_tokens.replace("ms", "")) / 1000.0
            elif "s" in reset_tokens:
                return min(float(reset_tokens.replace("s", "")), 6.0)
        except ValueError:
            pass
    # 3. Regex in response body
    try:
        data = response.json()
        msg = data.get("error", {}).get("message", "")
        match = re.search(r"try again in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
        if match:
            return min(float(match.group(1)), 6.0)
    except Exception:
        pass
    return 2.0


def groq_json_completion(
    system_prompt: str,
    user_prompt: str,
    max_retries_per_model: int = 2,
    temperature: float = 0.0,
) -> dict:
    """Execute chat completion with JSON mode, multi-model failover and backoff."""
    if not GROQ_API_KEY:
        return {"_error": "GROQ_API_KEY not found in environment."}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    raw_text = None
    last_error = "No response from Groq"

    # Up to 2 full passes across all available models
    for _ in range(2):
        for model_name in GROQ_MODELS:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }

            for attempt in range(1, max_retries_per_model + 1):
                _rate_limit_wait()
                try:
                    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=25)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_text = data["choices"][0]["message"]["content"]
                        break
                    elif resp.status_code == 429:
                        wait = _parse_wait_time(resp)
                        last_error = f"Rate limit on {model_name}: {resp.text[:150]}"
                        if model_name != GROQ_MODELS[-1]:
                            time.sleep(0.4)
                            break  # Failover to next model
                        else:
                            time.sleep(min(wait, 3.0) * attempt)
                    elif resp.status_code in (500, 502, 503, 504):
                        last_error = f"Server error {resp.status_code} on {model_name}"
                        time.sleep(1.0 * attempt)
                    else:
                        last_error = f"HTTP {resp.status_code} on {model_name}: {resp.text[:150]}"
                        break
                except Exception as ex:
                    last_error = f"Network exception on {model_name}: {str(ex)}"
                    time.sleep(0.5)

            if raw_text is not None:
                break
        if raw_text is not None:
            break

    if raw_text is None:
        return {"_error": f"Groq failover exhausted: {last_error}"}

    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    fence_match = _FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"_error": f"Unparseable LLM output: {raw_text[:200]}"}
