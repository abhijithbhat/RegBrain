import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("API Key exists:", bool(GROQ_API_KEY), flush=True)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello, answer in one word: ready"}],
    "temperature": 0
}
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

print("Calling Groq API...", flush=True)
try:
    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=15)
    print("Status code:", resp.status_code, flush=True)
    print("Response headers:", resp.headers, flush=True)
    print("Response text:", resp.text, flush=True)
except Exception as e:
    print("Exception:", e, flush=True)
