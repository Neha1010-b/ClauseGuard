"""
Phase 7.4c verification — test document persistence endpoints end-to-end.
Run: python scripts/test_documents.py
Requires: FastAPI server on http://localhost:8000
"""
import sys
import json
import uuid
from http.cookiejar import CookieJar
from urllib import request as urlreq
from urllib.error import HTTPError


BASE = "http://localhost:8000"


def _make_opener(cookie_jar):
    return urlreq.build_opener(urlreq.HTTPCookieProcessor(cookie_jar))


def _request(method, path, body=None, cookie_jar=None, is_json=True):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        if is_json:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            data = body
    req = urlreq.Request(url, data=data, method=method, headers=headers)
    opener = _make_opener(cookie_jar) if cookie_jar is not None else urlreq.build_opener()
    try:
        with opener.open(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def main():
    print("=" * 70)
    print("Phase 7.4c — Document persistence test")
    print("=" * 70)

    # 1. Sign up a fresh user
    email = f"doctest_{uuid.uuid4().hex[:8]}@example.com"
    jar = CookieJar()
    status, body = _request(
        "POST", "/auth/signup",
        body={"email": email, "full_name": "Doc Test", "password": "secret123"},
        cookie_jar=jar,
    )
    print(f"\n[1] Signup  -> {status}")
    assert status == 200, f"Signup failed: {body}"
    print(f"    user_id = {body['id']}")

    # 2. List documents (empty)
    status, body = _request("GET", "/documents", cookie_jar=jar)
    print(f"\n[2] GET /documents (empty)  -> {status}")
    assert status == 200 and len(body["documents"]) == 0, f"Expected empty list: {body}"

    # 3. Analyze the real contract
    print(f"\n[3] Analyzing real_contract.pdf (this takes ~30s)...")
    from pathlib import Path
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "real_contract.pdf"
    if not pdf_path.exists():
        print(f"SKIP: {pdf_path} not found")
        return

    # Multipart upload
    boundary = "----boundary" + uuid.uuid4().hex
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="generate_explanations"\r\n\r\n')
    parts.append(b"false\r\n")   # skip LLM for speed
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"\r\n'.encode()
    )
    parts.append(b"Content-Type: application/pdf\r\n\r\n")
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body_bytes = b"".join(parts)

    req = urlreq.Request(BASE + "/analyze/quick", data=body_bytes, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    opener = _make_opener(jar)
    with opener.open(req, timeout=180) as resp:
        analysis = json.loads(resp.read().decode("utf-8"))
    print(f"    Analysis done: {analysis['summary']}")

    # 4. Save it as a document
    status, saved = _request(
        "POST", "/documents",
        body={"filename": pdf_path.name, "analysis": analysis},
        cookie_jar=jar,
    )
    print(f"\n[4] POST /documents  -> {status}")
    assert status == 200, f"Save failed: {saved}"
    doc_id = saved["id"]
    print(f"    Saved with id = {doc_id[:12]}...")
    print(f"    Summary: {saved['high_count']}H / {saved['medium_count']}M / {saved['low_count']}L")

    # 5. List documents (should be 1)
    status, body = _request("GET", "/documents", cookie_jar=jar)
    print(f"\n[5] GET /documents  -> {status}")
    assert status == 200 and len(body["documents"]) == 1, f"Expected 1 doc: {body}"
    print(f"    Found: {body['documents'][0]['filename']}")

    # 6. Get full document
    status, body = _request("GET", f"/documents/{doc_id}", cookie_jar=jar)
    print(f"\n[6] GET /documents/{doc_id[:12]}...  -> {status}")
    assert status == 200
    assert body["analysis"]["summary"]["total_clauses"] == analysis["summary"]["total_clauses"]
    print(f"    Round-trip works: {body['analysis']['summary']['total_clauses']} clauses")

    # 7. Get nonexistent doc -> 404
    status, body = _request("GET", f"/documents/{'0'*32}", cookie_jar=jar)
    print(f"\n[7] GET /documents/nonexistent  -> {status}")
    assert status == 404

    # 8. Delete document
    status, body = _request("DELETE", f"/documents/{doc_id}", cookie_jar=jar)
    print(f"\n[8] DELETE /documents/{doc_id[:12]}...  -> {status}")
    assert status == 200

    # 9. List is empty again
    status, body = _request("GET", "/documents", cookie_jar=jar)
    print(f"\n[9] GET /documents (after delete)  -> {status}")
    assert status == 200 and len(body["documents"]) == 0

    # 10. Unauth user gets 401
    status, body = _request("GET", "/documents")
    print(f"\n[10] GET /documents (no auth)  -> {status}")
    assert status == 401

    print(f"\n{'=' * 70}")
    print("✅ All 10 document tests passed")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()