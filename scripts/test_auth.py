"""
Phase 7.4b verification — test auth endpoints end-to-end.
Run: python scripts/test_auth.py
Requires: the FastAPI server running on http://localhost:8000
"""
import sys
import json
import uuid
from http.cookiejar import CookieJar
from urllib import request as urlreq
from urllib.error import HTTPError


BASE = "http://localhost:8000"


def _make_opener(cookie_jar):
    """Build an opener that always uses the given cookie jar."""
    handler = urlreq.HTTPCookieProcessor(cookie_jar)
    return urlreq.build_opener(handler)


def _request(method, path, body=None, cookie_jar=None, headers=None):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urlreq.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)

    # If we have a cookie jar, use an opener that wires it in.
    if cookie_jar is not None:
        opener = _make_opener(cookie_jar)
    else:
        opener = urlreq.build_opener()

    try:
        with opener.open(req, timeout=15) as resp:
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


def _dump_cookies(jar, label):
    """Debug helper — print cookies in the jar."""
    print(f"    ({label} cookies: {[(c.name, c.domain, c.path) for c in jar]})")


def main():
    print("=" * 70)
    print("Phase 7.4b — Auth endpoints test")
    print("=" * 70)

    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "secret123"

    # 1. GET /auth/me without session -> 401
    status, body = _request("GET", "/auth/me")
    print(f"\n[1] GET /auth/me (no session)  -> {status} {body}")
    assert status == 401, "Expected 401 without session"

    # 2. Signup
    jar = CookieJar()
    status, body = _request(
        "POST", "/auth/signup",
        body={"email": email, "full_name": "Test User", "password": password},
        cookie_jar=jar,
    )
    print(f"\n[2] POST /auth/signup  -> {status} {body}")
    _dump_cookies(jar, "after signup")
    assert status == 200, f"Signup failed: {body}"

    # 3. GET /auth/me with cookie -> returns user
    status, body = _request("GET", "/auth/me", cookie_jar=jar)
    print(f"\n[3] GET /auth/me (with cookie)  -> {status} {body}")
    _dump_cookies(jar, "after me")
    assert status == 200 and body["email"] == email, f"Auth failed: {body}"

    # 4. Duplicate signup -> 409
    status, body = _request(
        "POST", "/auth/signup",
        body={"email": email, "full_name": "Dup", "password": "anything"},
    )
    print(f"\n[4] POST /auth/signup (duplicate)  -> {status} {body}")
    assert status == 409

    # 5. Signout
    status, body = _request("POST", "/auth/signout", cookie_jar=jar)
    print(f"\n[5] POST /auth/signout  -> {status} {body}")
    _dump_cookies(jar, "after signout")
    assert status == 200

    # 6. GET /auth/me after signout -> 401
    status, body = _request("GET", "/auth/me", cookie_jar=jar)
    print(f"\n[6] GET /auth/me (after signout)  -> {status} {body}")
    assert status == 401

    # 7. Signin with wrong password -> 401
    status, body = _request(
        "POST", "/auth/signin",
        body={"email": email, "password": "wrong_password"},
    )
    print(f"\n[7] POST /auth/signin (wrong pass)  -> {status} {body}")
    assert status == 401

    # 8. Signin with correct password -> 200
    jar2 = CookieJar()
    status, body = _request(
        "POST", "/auth/signin",
        body={"email": email, "password": password},
        cookie_jar=jar2,
    )
    print(f"\n[8] POST /auth/signin (correct)  -> {status} {body}")
    _dump_cookies(jar2, "after signin")
    assert status == 200

    # 9. Verify we're logged in
    status, body = _request("GET", "/auth/me", cookie_jar=jar2)
    print(f"\n[9] GET /auth/me (new session)  -> {status} {body}")
    assert status == 200

    print(f"\n{'=' * 70}")
    print("✅ All 9 auth tests passed")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()