import os
import sqlite3
import time
from typing import Iterable, Dict, Any, List, Optional


FTS_CREATE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    text,
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    path UNINDEXED,
    filename UNINDEXED,
    chunk_start UNINDEXED,
    chunk_end UNINDEXED,
    mtime UNINDEXED,
    page_numbers UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks'")
    exists = cur.fetchone() is not None
    if exists:
        cur = conn.execute("PRAGMA table_info(fts_chunks)")
        columns = [row[1] for row in cur.fetchall()]
        if "page_numbers" not in columns:
            cur = conn.execute(
                "SELECT text, chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime FROM fts_chunks"
            )
            existing_rows = cur.fetchall()
            conn.execute("DROP TABLE IF EXISTS fts_chunks")
            conn.execute(FTS_CREATE_SQL)
            if existing_rows:
                conn.executemany(
                    """
                    INSERT INTO fts_chunks (text, chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime, page_numbers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, '')
                    """,
                    existing_rows,
                )
            conn.commit()
            return
    conn.execute(FTS_CREATE_SQL)
    conn.commit()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def ensure_fts(db_path: str) -> None:
    _ensure_parent_dir(db_path)
    last_err = None
    conn = None
    for attempt in range(3):
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            _ensure_schema(conn)
            return
        except sqlite3.OperationalError as ex:
            last_err = ex
            if "unable to open database file" in str(ex).lower():
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
        finally:
            if conn is not None:
                conn.close()
    # If we reach here, retries exhausted
    raise last_err


class FTSWriter:
    """Single-connection FTS writer for reliable, fast bulk upserts.

    Use as a context manager:

        with FTSWriter(path, recreate=True) as w:
            w.upsert_many(rows)
    """

    def __init__(self, db_path: str, recreate: bool = False):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        _ensure_parent_dir(db_path)
        # Open with retries
        last_err = None
        for attempt in range(3):
            try:
                self.conn = sqlite3.connect(db_path)
                self.conn.execute("PRAGMA journal_mode=WAL;")
                self.conn.execute("PRAGMA busy_timeout=5000;")
                _ensure_schema(self.conn)
                break
            except sqlite3.OperationalError as ex:
                last_err = ex
                if "unable to open database file" in str(ex).lower():
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
        if self.conn is None:
            raise last_err

        cur = self.conn.cursor()
        if recreate:
            cur.execute("DROP TABLE IF EXISTS fts_chunks")
        cur.execute(FTS_CREATE_SQL)
        self.conn.commit()

    def upsert_many(self, rows: Iterable[Dict[str, Any]]) -> int:
        rows = [r for r in rows if r.get("text")]
        if not rows:
            return 0
        cur = self.conn.cursor()
        # Delete existing by chunk_id
        cur.executemany(
            "DELETE FROM fts_chunks WHERE chunk_id = ?",
            [(str(r.get("chunk_id")),) for r in rows],
        )
        # Insert many
        cur.executemany(
            """
            INSERT INTO fts_chunks (text, chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime, page_numbers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.get("text", ""),
                    str(r.get("chunk_id")),
                    str(r.get("doc_id")),
                    str(r.get("path")),
                    str(r.get("filename")),
                    int(r.get("chunk_start", 0) or 0),
                    int(r.get("chunk_end", 0) or 0),
                    int(r.get("mtime", 0) or 0),
                    str(r.get("page_numbers", "") or ""),
                )
                for r in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def __enter__(self) -> "FTSWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def upsert_chunks(db_path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """Upsert chunk rows into FTS, identified by chunk_id. Returns inserted count.

    Expected row keys: text, chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime
    """
    # Ensure DB exists and open with simple retries to avoid transient
    # "unable to open database file" issues on some systems.
    ensure_fts(db_path)
    last_err = None
    conn = None
    for attempt in range(3):
        try:
            conn = sqlite3.connect(db_path)
            # Improve resilience on busy filesystems
            conn.execute("PRAGMA busy_timeout=5000;")
            break
        except sqlite3.OperationalError as ex:
            last_err = ex
            # Retry only for open failures
            if "unable to open database file" in str(ex).lower():
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    if conn is None:
        # Re-raise the last error if we failed all attempts
        raise last_err
    try:
        cur = conn.cursor()
        count = 0
        cur.execute("BEGIN")
        for r in rows:
            if not r.get("text"):
                continue
            # Delete any existing entry for this chunk_id, then insert
            cur.execute("DELETE FROM fts_chunks WHERE chunk_id = ?", (str(r.get("chunk_id")),))
            cur.execute(
                """
                INSERT INTO fts_chunks (text, chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime, page_numbers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("text", ""),
                    str(r.get("chunk_id")),
                    str(r.get("doc_id")),
                    str(r.get("path")),
                    str(r.get("filename")),
                    int(r.get("chunk_start", 0) or 0),
                    int(r.get("chunk_end", 0) or 0),
                    int(r.get("mtime", 0) or 0),
                    str(r.get("page_numbers", "") or ""),
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def search(db_path: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search the FTS index with BM25 ranking. Returns list of rows."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Order by bm25() ASC for best first
        cur.execute(
            """
            SELECT chunk_id, doc_id, path, filename, chunk_start, chunk_end, mtime, page_numbers, text,
                   bm25(fts_chunks) AS bm25
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
            ORDER BY bm25 LIMIT ?
            """,
            (query, int(limit)),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
