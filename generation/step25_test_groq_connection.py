"""
Step 25 – Smoke-test the Groq API connection.

Loads GROQ_API_KEY from .env, sends a single hardcoded prompt to
the "openai/gpt-oss-120b" model via the Groq REST API, and prints
the raw response text.
"""

import os
import sys
import requests
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────
load_dotenv()  # reads .env from project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("ERROR: GROQ_API_KEY not found in environment. Add it to .env")

# ── Call the Groq chat-completions endpoint ─────────────────────
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

payload = {
    "model": "openai/gpt-oss-120b",
    "messages": [
        {
            "role": "user",
            "content": "Say hello and confirm you're working.",
        }
    ],
}

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

print("Sending request to Groq (model: openai/gpt-oss-120b) …")

response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)

if response.status_code != 200:
    print(f"Request failed  [{response.status_code}]")
    print(response.text)
    sys.exit(1)

data = response.json()
reply = data["choices"][0]["message"]["content"]

print("\n── Raw response ──────────────────────────────────────")
print(reply)
print("─────────────────────────────────────────────────────")
print("\n✅ Groq connection successful.")
