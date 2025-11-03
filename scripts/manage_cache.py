#!/usr/bin/env python3
"""Utility helpers for managing ingestion caches and summary stores.

Usage examples:

    # Show current cache / DB sizes
    python scripts/manage_cache.py status

    # Clear the Docling page cache
    python scripts/manage_cache.py clear-ingest-cache

    # Remove graph.db and summary.db (they will be rebuilt on next ingest)
    python scripts/manage_cache.py clear-graph --graph-db data/graph.db
    python scripts.manage_cache.py clear-summary --summary-db data/summary.db

The script is intentionally conservative: it only deletes the requested files/dirs
and prints what changed.
"""

import argparse
import os
import shutil
from pathlib import Path

DEFAULT_INGEST_CACHE = Path(os.getenv("INGEST_CACHE_DIR", ".ingest_cache"))
DEFAULT_GRAPH_DB = Path(os.getenv("GRAPH_DB_PATH", "data/graph.db"))
DEFAULT_SUMMARY_DB = Path(os.getenv("SUMMARY_DB_PATH", "data/summary.db"))


def _human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def cmd_status(args: argparse.Namespace) -> None:
    print(f"Ingest cache: {args.ingest_cache.resolve()} ({_human_size(args.ingest_cache)})")
    print(f"Graph DB    : {args.graph_db.resolve()} ({_human_size(args.graph_db)})")
    print(f"Summary DB  : {args.summary_db.resolve()} ({_human_size(args.summary_db)})")


def _safe_remove(path: Path) -> None:
    if not path.exists():
        print(f"Nothing to remove: {path}")
        return
    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    print(f"Removed {path}")


def cmd_clear_ingest_cache(args: argparse.Namespace) -> None:
    _safe_remove(args.ingest_cache)


def cmd_clear_graph(args: argparse.Namespace) -> None:
    _safe_remove(args.graph_db)


def cmd_clear_summary(args: argparse.Namespace) -> None:
    _safe_remove(args.summary_db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage ingestion caches and summary stores")
    sub = parser.add_subparsers(dest="command", required=True)

    base_defaults = {
        "ingest_cache": DEFAULT_INGEST_CACHE,
        "graph_db": DEFAULT_GRAPH_DB,
        "summary_db": DEFAULT_SUMMARY_DB,
    }

    status = sub.add_parser("status", help="Show cache/DB sizes")
    status.add_argument("--ingest-cache", type=Path, default=base_defaults["ingest_cache"])
    status.add_argument("--graph-db", type=Path, default=base_defaults["graph_db"])
    status.add_argument("--summary-db", type=Path, default=base_defaults["summary_db"])
    status.set_defaults(func=cmd_status)

    clear_cache = sub.add_parser("clear-ingest-cache", help="Delete the Docling/page cache directory")
    clear_cache.add_argument("--ingest-cache", type=Path, default=base_defaults["ingest_cache"])
    clear_cache.set_defaults(func=cmd_clear_ingest_cache)

    clear_graph = sub.add_parser("clear-graph", help="Remove the graph.db file (will be regenerated)")
    clear_graph.add_argument("--graph-db", type=Path, default=base_defaults["graph_db"])
    clear_graph.set_defaults(func=cmd_clear_graph)

    clear_summary = sub.add_parser("clear-summary", help="Remove the summary.db file (will be regenerated)")
    clear_summary.add_argument("--summary-db", type=Path, default=base_defaults["summary_db"])
    clear_summary.set_defaults(func=cmd_clear_summary)

    args = parser.parse_args()
    args.func(args)


+if __name__ == "__main__":
+    main()
