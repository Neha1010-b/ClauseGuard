"""
SQLite database layer — Phase 7.4a
Handles user accounts and saved document analyses.

Design:
- Single SQLite file at data/app.db (gitignored)
- WAL mode for better concurrency
- Connection per request (simple, safe, fast enough for our scale)
- No ORM — raw SQL with a tiny helper. Keep it transparent.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..utils.config import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "data" / "app.db"


# ============================================================
# Connection management
# ============================================================
def _connect() -> sqlite3.Connection:
    """Create a connection with sane defaults."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row       # access columns by name
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def get_db():
    """Context manager for a connection — commits on success, rolls back on error."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# Schema initialization
# ============================================================
def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                format TEXT NOT NULL,
                pages INTEGER NOT NULL,
                chars INTEGER NOT NULL,
                total_clauses INTEGER NOT NULL,
                high_count INTEGER NOT NULL,
                medium_count INTEGER NOT NULL,
                low_count INTEGER NOT NULL,
                analysis_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_documents_user_created
                ON documents(user_id, created_at DESC);
        """)


# ============================================================
# User operations
# ============================================================
def create_user(email: str, full_name: str, password_hash: str) -> int:
    """Insert a user, return the new user id. Raises on duplicate email."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, full_name, password_hash) VALUES (?, ?, ?)",
            (email.lower().strip(), full_name.strip(), password_hash),
        )
        return cur.lastrowid


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


# ============================================================
# Document operations
# ============================================================
def save_document(
    doc_id: str,
    user_id: int,
    filename: str,
    format: str,
    pages: int,
    chars: int,
    total_clauses: int,
    high_count: int,
    medium_count: int,
    low_count: int,
    analysis_json: str,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO documents
            (id, user_id, filename, format, pages, chars,
             total_clauses, high_count, medium_count, low_count, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, user_id, filename, format, pages, chars,
             total_clauses, high_count, medium_count, low_count, analysis_json),
        )


def list_documents(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Return summary info for the user's documents, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, format, pages, chars,
                   total_clauses, high_count, medium_count, low_count, created_at
            FROM documents
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_document(doc_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Get a single document, ensuring it belongs to the user."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        ).fetchone()
        return dict(row) if row else None


def delete_document(doc_id: str, user_id: int) -> bool:
    """Delete a document. Returns True if a row was deleted."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        return cur.rowcount > 0


def count_user_documents(user_id: int) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM documents WHERE user_id = ?", (user_id,)
        ).fetchone()
        return int(row["n"])