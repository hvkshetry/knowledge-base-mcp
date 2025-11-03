import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SUMMARY_DB_PATH = os.getenv("SUMMARY_DB_PATH", "data/summary.db")


def _ensure_summary(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            collection TEXT,
            doc_id TEXT,
            section_path TEXT,
            summary TEXT,
            element_ids TEXT,
            metadata TEXT,
            PRIMARY KEY (collection, doc_id, section_path)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_summaries_collection ON summaries(collection)"
    )
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info('summaries')")}
        if "metadata" not in cols:
            conn.execute("ALTER TABLE summaries ADD COLUMN metadata TEXT")
    except sqlite3.DatabaseError:
        pass
    return conn


def _concise_summary(texts: List[str], max_chars: int = 600) -> str:
    combined = " ".join(texts)
    if len(combined) <= max_chars:
        return combined
    sentences = combined.split(". ")
    out = []
    total = 0
    for sentence in sentences:
        addition = sentence.strip()
        if not addition:
            continue
        out.append(addition)
        total += len(addition) + 2
        if total >= max_chars:
            break
    return ". ".join(out)[:max_chars]


def upsert_summaries(
    collection: str,
    doc_id: str,
    chunks: Iterable[Dict],
    summary_path: str = SUMMARY_DB_PATH,
) -> None:
    if not collection:
        return
    conn = _ensure_summary(Path(summary_path))
    cur = conn.cursor()
    groups: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)
    for chunk in chunks:
        section = tuple(chunk.get("section_path") or [])
        groups[section].append(chunk)
    for section_path, chunk_list in groups.items():
        texts = [c.get("text", "") for c in chunk_list if c.get("text")]
        if not texts:
            continue
        summary = _concise_summary(texts)
        element_ids = []
        for c in chunk_list:
            ids = c.get("element_ids") or []
            element_ids.extend(ids)
        cur.execute(
            """
            INSERT OR REPLACE INTO summaries(collection, doc_id, section_path, summary, element_ids, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                collection,
                doc_id,
                json.dumps(section_path, ensure_ascii=False),
                summary,
                json.dumps(element_ids, ensure_ascii=False),
                None,
            ),
        )
    conn.commit()
    conn.close()


def query_summaries(
    collection: str,
    topic: str,
    limit: int = 5,
    summary_path: str = SUMMARY_DB_PATH,
) -> List[Dict[str, str]]:
    conn = _ensure_summary(Path(summary_path))
    cur = conn.cursor()
    like = f"%{topic.lower()}%"
    cur.execute(
        """
        SELECT doc_id, section_path, summary, element_ids, metadata
        FROM summaries
        WHERE collection = ?
          AND lower(summary) LIKE ?
        LIMIT ?
        """,
        (collection, like, limit),
    )
    rows = cur.fetchall()
    results: List[Dict[str, str]] = []
    for doc_id, section_path, summary, element_ids, metadata in rows:
        results.append(
            {
                "doc_id": doc_id,
                "section_path": json.loads(section_path),
                "summary": summary,
                "element_ids": json.loads(element_ids),
                "metadata": json.loads(metadata) if metadata else None,
            }
        )
    conn.close()
    return results


def upsert_summary_entry(
    collection: str,
    doc_id: str,
    section_path: Sequence[str],
    summary: str,
    element_ids: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    summary_path: str = SUMMARY_DB_PATH,
) -> None:
    if not collection:
        return
    conn = _ensure_summary(Path(summary_path))
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO summaries(collection, doc_id, section_path, summary, element_ids, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            collection,
            doc_id,
            json.dumps(tuple(section_path), ensure_ascii=False),
            summary,
            json.dumps(list(element_ids or []), ensure_ascii=False),
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
        ),
    )
    conn.commit()
    conn.close()
