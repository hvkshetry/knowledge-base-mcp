#!/usr/bin/env python3
"""
Generate a retrieval gold set by sampling chunks from an FTS database.

The script emits JSONL records consumable by eval.py:
{
  "query": "...",
  "collection": "daf_kb",
  "relevance": [
    {"doc_id": "...", "chunk_id": "...", "path_contains": "...", "gain": 1.0}
  ],
  "metadata": {...}
}

Each synthetic query references an exact snippet from the chunk, which ensures
lexical and dense pipelines can locate the source while remaining deterministic.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


MIN_WORD_COUNT = 12
DEFAULT_SAMPLE_FACTOR = 6


def _has_column(conn: sqlite3.Connection, column: str) -> bool:
    cur = conn.execute("PRAGMA table_info('fts_chunks')")
    return any(row[1] == column for row in cur.fetchall())


def _load_candidates(conn: sqlite3.Connection, limit: int, min_words: int) -> List[Dict[str, Any]]:
    has_section = _has_column(conn, "section_path")
    has_pages = _has_column(conn, "pages")
    columns = ["doc_id", "chunk_id", "path", "text"]
    if has_section:
        columns.append("section_path")
    if has_pages:
        columns.append("pages")
    sql = f"""
        SELECT {", ".join(columns)}
        FROM fts_chunks
        WHERE text IS NOT NULL
          AND LENGTH(text) >= ?
        ORDER BY doc_id, chunk_start
        LIMIT ?
    """
    cur = conn.execute(sql, (min_words * 6, limit))
    rows: List[Dict[str, Any]] = []
    for fetched in cur.fetchall():
        # Map fetch tuple back into dict using selected columns
        row_map = dict(zip(columns, fetched))
        text = row_map.get("text")
        if not text:
            continue
        rows.append(
            {
                "doc_id": str(row_map.get("doc_id")),
                "chunk_id": str(row_map.get("chunk_id")),
                "path": str(row_map.get("path")),
                "section_path": row_map.get("section_path"),
                "text": text,
                "pages": row_map.get("pages"),
            }
        )
    return rows


WORD_RE = re.compile(r"[A-Za-z0-9%°μ/+-]{2,}")


def _normalise_words(words: Sequence[str]) -> Sequence[str]:
    return [w.lower() for w in words if len(w) >= 2]


def _select_snippet(text: str, max_words: int = 12) -> str:
    words = WORD_RE.findall(text)
    if len(words) < max_words:
        snippet = " ".join(words)
    else:
        snippet = " ".join(words[:max_words])
    return snippet.strip()


def _build_query(snippet: str, section: Sequence[str], doc_name: str) -> str:
    section_label = ""
    if section:
        section_label = " > ".join(str(part) for part in section if part)
    quoted = snippet[:120]
    if section_label:
        return f'Which passage in "{section_label}" discusses "{quoted}" in {doc_name}?'
    return f'Where in {doc_name} is the phrase "{quoted}" explained?'


def _parse_section(section_raw: Any) -> List[str]:
    if not section_raw:
        return []
    if isinstance(section_raw, list):
        return [str(x) for x in section_raw if x]
    if isinstance(section_raw, str):
        try:
            data = json.loads(section_raw)
            if isinstance(data, list):
                return [str(x) for x in data if x]
        except json.JSONDecodeError:
            parts = [part.strip() for part in section_raw.split(">")]
            return [p for p in parts if p]
    return []


def generate_records(
    candidates: Iterable[Dict[str, Any]],
    collection: str,
    limit: int,
    seed: int,
    min_words: int,
) -> List[Dict[str, Any]]:
    subset = list(candidates)
    random.Random(seed).shuffle(subset)

    records: List[Dict[str, Any]] = []
    seen_snippets: set[str] = set()

    for entry in subset:
        if len(records) >= limit:
            break
        text = entry["text"]
        snippet = _select_snippet(text)
        if len(_normalise_words(snippet.split())) < min_words:
            continue
        key = snippet.lower()
        if key in seen_snippets:
            continue
        seen_snippets.add(key)

        section = _parse_section(entry.get("section_path"))
        doc_name = Path(entry["path"]).name if entry.get("path") else entry["doc_id"]
        query = _build_query(snippet, section, doc_name)

        relevance = {
            "doc_id": entry["doc_id"],
            "chunk_id": entry["chunk_id"],
            "path_contains": doc_name[:80],
            "gain": 1.0,
        }
        metadata = {
            "snippet": snippet,
            "section_path": section,
            "pages": entry.get("pages"),
        }
        records.append(
            {
                "query": query,
                "collection": collection,
                "relevance": [relevance],
                "metadata": metadata,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation gold sets from an FTS database.")
    parser.add_argument("--fts", required=True, help="Path to the SQLite FTS database.")
    parser.add_argument("--collection", required=True, help="Collection slug for the generated records.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--limit", type=int, default=160, help="Number of queries to generate.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for deterministic sampling.")
    parser.add_argument("--sample-factor", type=int, default=DEFAULT_SAMPLE_FACTOR, help="Multiplier for candidate sampling.")
    parser.add_argument("--min-words", type=int, default=MIN_WORD_COUNT, help="Minimum unique words required per snippet.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.fts)
    conn.execute("PRAGMA temp_store=MEMORY;")
    try:
        candidate_limit = max(args.limit * max(args.sample_factor, 2), args.limit)
        candidates = _load_candidates(conn, candidate_limit, args.min_words)
    finally:
        conn.close()

    if not candidates:
        raise SystemExit("No candidate chunks found; check FTS path or re-ingest first.")

    records = generate_records(
        candidates,
        args.collection,
        max(1, args.limit),
        args.seed,
        max(3, args.min_words),
    )

    if not records:
        raise SystemExit("Unable to generate records with the requested constraints.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
