"""
Phase 7.4a verification — test the SQLite database layer.
Run: python scripts/test_db.py
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.db import (
    init_db, create_user, get_user_by_email, save_document,
    list_documents, get_db, DB_PATH,
)


def main():
    print("=" * 70)
    print("Phase 7.4a — SQLite database verification")
    print("=" * 70)

    # 1. Init schema
    init_db()
    print(f"\n[1] ✅ Database initialized at {DB_PATH}")
    print(f"    Exists: {DB_PATH.exists()}")
    if DB_PATH.exists():
        print(f"    Size: {DB_PATH.stat().st_size} bytes")

    # 2. Create test user
    test_email = "test@example.com"
    # Clean slate if it exists
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE email = ?", (test_email,))

    uid = create_user(test_email, "Test User", "fake_hash_xyz")
    print(f"\n[2] ✅ Test user created: id={uid}, email={test_email}")

    # 3. Verify lookup
    u = get_user_by_email(test_email)
    assert u is not None and u["id"] == uid
    print(f"    Lookup by email works: {u['full_name']}")

    # 4. Save a fake document
    doc_id = str(uuid.uuid4())
    fake_json = '{"fake": "analysis data", "clauses": []}'
    save_document(
        doc_id=doc_id,
        user_id=uid,
        filename="test_contract.pdf",
        format="pdf",
        pages=8,
        chars=12701,
        total_clauses=47,
        high_count=1,
        medium_count=1,
        low_count=41,
        analysis_json=fake_json,
    )
    print(f"\n[3] ✅ Document saved: id={doc_id[:8]}...")

    # 5. List documents
    docs = list_documents(uid)
    print(f"\n[4] ✅ Listed {len(docs)} documents for user {uid}:")
    for d in docs:
        print(f"    {d['filename']}  "
              f"({d['high_count']}H/{d['medium_count']}M/{d['low_count']}L)  "
              f"{d['created_at']}")

    # 6. Round-trip JSON
    from src.api.db import get_document
    fetched = get_document(doc_id, uid)
    assert fetched is not None
    import json
    parsed = json.loads(fetched["analysis_json"])
    assert parsed["fake"] == "analysis data"
    print(f"\n[5] ✅ JSON round-trip works")

    # 7. Cleanup
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE email = ?", (test_email,))
    print(f"\n[6] ✅ Cleanup done (test user + cascaded documents removed)")

    print(f"\n{'=' * 70}")
    print("✅ All database operations verified")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()