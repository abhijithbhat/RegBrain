import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

for model in ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=10)
    print(f"Model: {model} -> Status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"   Error: {resp.text[:150]}")
