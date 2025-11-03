#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
from typing import Any, Dict, List

from qdrant_client import QdrantClient

# Ensure repo root is on sys.path so we can import lexical_index when invoked as a file
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qdrant-url', default=os.getenv('QDRANT_URL', 'http://localhost:6333'))
    ap.add_argument('--qdrant-api-key', default=os.getenv('QDRANT_API_KEY'))
    ap.add_argument('--collection', required=True)
    ap.add_argument('--fts-db', default=os.getenv('FTS_DB_PATH', 'data/fts.db'))
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0, help='Stop after N points (0 = all)')
    ap.add_argument('--min-words', type=int, default=0)
    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    from lexical_index import ensure_fts, upsert_chunks

    ensure_fts(args.fts_db)

    next_page = None
    total = 0
    while True:
        points, next_page = client.scroll(
            collection_name=args.collection,
            with_payload=True,
            with_vectors=False,
            limit=args.batch,
            offset=next_page,
        )
        if not points:
            break
        rows = []
        for p in points:
            pl: Dict[str, Any] = p.payload or {}
            text = (pl.get('text') or '').strip()
            if not text:
                continue
            if args.min_words:
                import re
                if len(re.findall(r'[A-Za-z]{2,}', text)) < args.min_words:
                    continue
            page_numbers = pl.get('page_numbers') or pl.get('pages')
            section_path = pl.get('section_path')
            element_ids = pl.get('element_ids')
            bboxes = pl.get('bboxes')
            types = pl.get('types')
            source_tools = pl.get('source_tools')
            table_headers = pl.get('table_headers')
            table_units = pl.get('table_units')
            chunk_profile = pl.get('chunk_profile')
            plan_hash = pl.get('plan_hash')
            doc_metadata = pl.get('doc_metadata')
            model_version = pl.get('model_version')
            prompt_sha = pl.get('prompt_sha')
            rows.append({
                'text': text,
                'chunk_id': str(p.id),
                'doc_id': pl.get('doc_id'),
                'path': pl.get('path'),
                'filename': pl.get('filename'),
                'chunk_start': pl.get('chunk_start'),
                'chunk_end': pl.get('chunk_end'),
                'mtime': pl.get('mtime'),
                'page_numbers': page_numbers,
                'pages': pl.get('pages'),
                'section_path': section_path,
                'element_ids': element_ids,
                'bboxes': bboxes,
                'types': types,
                'source_tools': source_tools,
                'table_headers': table_headers,
                'table_units': table_units,
                'chunk_profile': chunk_profile,
                'plan_hash': plan_hash,
                'model_version': model_version,
                'prompt_sha': prompt_sha,
                'doc_metadata': doc_metadata,
            })
        if rows:
            upsert_chunks(args.fts_db, rows)
            total += len(rows)
        if args.limit and total >= args.limit:
            break
        if next_page is None:
            break
    print(f"Backfilled {total} chunks into FTS at {args.fts_db}")


if __name__ == '__main__':
    main()
