"""Quick check that .env is set up correctly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import os

gem = os.getenv("GEMINI_API_KEY")
jwt = os.getenv("JWT_SECRET_KEY")

print("GEMINI_API_KEY:")
if gem:
    print(f"  ✅ Set (length={len(gem)}, starts with '{gem[:6]}...')")
else:
    print("  ❌ MISSING")

print("\nJWT_SECRET_KEY:")
if not jwt:
    print("  ❌ MISSING")
elif jwt.startswith("change_this"):
    print(f"  ❌ Still the placeholder ({jwt[:30]}...)")
else:
    print(f"  ✅ Set (length={len(jwt)}, starts with '{jwt[:6]}...')")

if not gem or not jwt or jwt.startswith("change_this"):
    print("\n⚠️  Fix .env, then restart the uvicorn server before re-running test_auth.py")
    sys.exit(1)
else:
    print("\n✅ Environment looks good")