"""
Test which Gemini models actually work with our API key.
Run: python scripts/test_models.py
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Candidate models, in order of preference
CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-3.5-pro",
]

print("Testing each candidate model with a simple prompt...\n")

working = []
for model_name in CANDIDATES:
    try:
        start = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'ok' in one word.",
        )
        elapsed = time.time() - start
        text = (response.text or "").strip()[:40]
        print(f"  ✅ {model_name:30s}  ({elapsed:.2f}s)  → {text!r}")
        working.append(model_name)
    except Exception as e:
        err = str(e)[:80].replace("\n", " ")
        print(f"  ❌ {model_name:30s}  {err}")

print()
if working:
    print(f"Working models: {working}")
    print(f"Best candidate (first that worked): {working[0]}")
else:
    print("No models worked. Check your API key and quota.")