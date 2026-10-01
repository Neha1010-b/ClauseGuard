"""
Phase 6A.0 — Verify the new google-genai SDK works with your API key.
Run: python scripts/test_gemini_sdk.py
"""
import os
import sys
from pathlib import Path

# Load .env from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in environment")
    print("   Did you create .env in the project root with GEMINI_API_KEY=...")
    sys.exit(1)

print(f"✅ API key loaded (length: {len(api_key)}, starts with: {api_key[:6]}...)")

# Import new SDK
from google import genai

print("\n[1/2] Creating client...")
client = genai.Client(api_key=api_key)
print("      Client created")

print("\n[2/2] Sending test prompt to Gemini 2.0 Flash...")
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="In one short sentence, explain why a clause that says 'the Company may terminate at its sole discretion without notice' is risky.",
)
print("\n--- Gemini's response ---")
print(response.text)
print("--- end ---")
print("\n✅ Phase 6A.0 SDK test complete")