# Chat history management

import os
import uuid
import psycopg2
import psycopg2.extras

from contextlib import contextmanager
from typing import Generator
from psycopg2.pool import SimpleConnectionPool
from config import DATABASE_URL, HISTORY_TURNS

_pool: SimpleConnectionPool | None = None

def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "Add DATABASE_URL=postgresql://user:xxxxx@host/db?sslmode=required in .env"
            )
        _pool = SimpleConnectionPool(minconn=1, maxconn=3, dsn=DATABASE_URL)
    return _pool
    
@contextmanager
def _conn() -> Generator[psycopg2.extensions.connection, None, None]:
    pool = _get_pool()
    conn = pool.getconn()
    try: 
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

def create_session() -> str:
    sid = str(uuid.uuid4())
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO sessions (session_id) VALUES (%s)", (sid,))
    return sid

def save_message(session_id: str, role: str, content: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )

def get_history(session_id: str) -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM messages
                WHERE  session_id = %s
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (session_id, HISTORY_TURNS),
            )
            rows = cur.fetchall()
            return list(reversed([dict(r) for r in rows]))
        
def delete(session_id: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))