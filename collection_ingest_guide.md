# Technical KB Ingestion Playbook

This guide mirrors the successful `aerobic_treatment_kb` ingestion and shows how to build additional hybrid (semantic + lexical) collections for large technical PDF sets like `ix_db` (ion exchange) and `ro_kb` (reverse osmosis).

## Preflight Checklist
- Activate the project virtual environment:
  - Linux/WSL: `source .venv/bin/activate`
  - Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Ensure services are available:
  - Qdrant: `docker compose up -d qdrant`
  - Ollama: `ollama serve` and `ollama pull snowflake-arctic-embed:xs`
- Optional but recommended: set a Hugging Face cache for Docling layout models once per repo: `export HF_HOME="$PWD/.cache/hf"`.
- Work from repo root so relative paths like `data/*.db` resolve. Create the folder if needed: `mkdir -p data`.

## Structured Metadata Snapshot
- A triage pass labels each PDF page; light pages stay on MarkItDown while table/multi-column/scan pages route through Docling for heading/table/caption extraction.
- Chunks in Qdrant now include pages, section breadcrumbs, element IDs, bounding boxes, source tool, table headers/units, and more. Thin-payload mode can drop `text` to keep the vector store governance-safe.
- The SQLite FTS index stores the same metadata so lexical search and reranking can access provenance fields.
- Lightweight knowledge-graph (`data/graph.db`) and summary (`data/summary.db`) stores are refreshed on every run; graph nodes link docs → sections → chunks → heuristic entities.

## Bulk Run Template (Linux/WSL)
```bash
./.venv/bin/python ingest.py \
  --root "/mnt/g/KnowledgeBases/ion_exchange" \
  --qdrant-collection ix_db \
  --fts-db "data/ix_db_fts.db" \
  --ext ".pdf,.docx,.pptx,.md,.txt" \
  --changed-only --delete-changed \
  --min-words 50 \
  --batch-size 32 \
  --parallel 4 \
  --ollama-threads 8 \
  --ollama-keepalive 2h \
  --ollama-timeout 600 \
  --extractor auto \
  --thin-payload \
  --embed-robust
```
Replace the `--root`, `--qdrant-collection`, and `--fts-db` arguments for other topics (for example, point at `ro_kb` with `data/ro_kb_fts.db`).

## Bulk Run Template (Windows PowerShell)
```powershell
.\.venv\Scripts\python.exe .\ingest.py `
  --root "G:\KnowledgeBases\ion_exchange" `
  --qdrant-collection ix_db `
  --fts-db ".\data\ix_db_fts.db" `
  --ext ".pdf,.docx,.pptx,.md,.txt" `
  --changed-only --delete-changed `
  --min-words 50 `
  --batch-size 32 `
  --parallel 4 `
  --ollama-threads 8 `
  --ollama-keepalive 2h `
  --ollama-timeout 600 `
  --extractor auto `
  --thin-payload `
  --embed-robust
```

## Single-Document Recovery Flow
When very large PDFs keep timing out, process them one at a time so progress is durable.
1. (Optional) Remove existing vectors for the file if you want a clean slate:
   ```bash
   ./.venv/bin/python - <<'PY'
   import pathlib, uuid
   from ingest import file_uri
   from qdrant_client import QdrantClient, models

   path = pathlib.Path("ix_kb/<file.pdf>")
   doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_uri(path)))
   client = QdrantClient(url="http://localhost:6333")
   filt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
   client.delete(collection_name="ix_db", points_selector=models.FilterSelector(filter=filt))
   PY
   ```
2. Reingest the single document with conservative settings:
   ```bash
   ./.venv/bin/python ingest.py \
     --root "./ix_kb" \
     --include "*/<file.pdf>" \
     --qdrant-collection ix_db \
     --fts-db "data/ix_db_fts.db" \
     --ext ".pdf" \
     --skip "*:Zone.Identifier" \
     --min-words 50 \
     --max-chars 800 \
     --overlap 100 \
     --batch-size 1 \
     --parallel 1 \
     --ollama-per-item \
     --ollama-timeout 1800 \
     --embed-robust \
     --embed-window-size 1 \
     --ollama-threads 8 \
     --ollama-keepalive 4h \
     --max-docs-per-run 1 \
     --qdrant-timeout 600
   ```
These parameters embed one chunk at a time, trading speed for reliability.

## Rebuild Lexical Index After Doc-by-Doc Runs
After several targeted retries the FTS database can accumulate stale chunks. Rebuild it once when the vector ingest is complete:
```bash
./.venv/bin/python ingest.py \
  --root "./ix_kb" \
  --qdrant-collection ix_db \
  --fts-db "data/ix_db_fts.db" \
  --ext ".pdf" \
  --skip "*:Zone.Identifier" \
  --min-words 50 \
  --max-chars 800 \
  --overlap 100 \
  --fts-only \
  --fts-rebuild
```
This drops and recreates `fts_chunks`, ensuring the lexical index matches the latest chunk boundaries.

## Post-Run Validation
- Check the terminal for `SUMMARY processed_docs=... processed_chunks=... errors=0`.
- Confirm the collection exists: `curl http://localhost:6333/collections | jq '.result.collections[].name'`.
- Inspect graph/summary stores if needed:
  ```bash
  sqlite3 data/graph.db 'SELECT type, COUNT(*) FROM nodes GROUP BY type'
  sqlite3 data/summary.db 'SELECT COUNT(*) FROM summaries'
  ```
- Count vectors per document with the Python client (update the filename list as needed):
  ```bash
  ./.venv/bin/python - <<'PY'
  from qdrant_client import QdrantClient, models
  client = QdrantClient(url='http://localhost:6333')
  filenames = [
      '454.pdf',
      '8. Ion Exchange Design Proced.pdf',
      'Brian_Windsor_calculation.pdf',
      'IER-Fundamentals-TechFact-45-D01462-en.pdf',
      'Ion Exchange -- Friedrich G_ Helfferich.pdf',
      'Joe_Woolley_process_design.pdf',
      'Mass Transfer and Kinetics of Ion Exchange.pdf',
  ]
  for name in filenames:
      filt = models.Filter(must=[models.FieldCondition(key='filename', match=models.MatchValue(value=name))])
      count = client.count(collection_name='ix_db', count_filter=filt, exact=True).count
      print(f"{name}: {count}")
  print('Total points', client.count(collection_name='ix_db', exact=True).count)
  PY
  ```
- Inspect the lexical index directly with sqlite:
  ```bash
  ./.venv/bin/python - <<'PY'
  import sqlite3
  conn = sqlite3.connect('data/ix_db_fts.db')
  cur = conn.cursor()
  cur.execute('SELECT filename, COUNT(*) FROM fts_chunks GROUP BY filename ORDER BY filename')
  for filename, count in cur.fetchall():
      print(filename, count)
  cur.execute('SELECT COUNT(*) FROM fts_chunks')
  print('Total rows', cur.fetchone()[0])
  conn.close()
  PY
  ```
- Run spot queries with your retrieval tooling (for example, `python -m tools.sample_query --collection ix_db --query "anion resin regen"`).
- Optionally run the CI-friendly evaluation harness: `./.venv/bin/python eval.py --gold eval/gold_sets/ix_db.jsonl --mode auto --fts-db data/ix_db_fts.db --min-ndcg 0.85 --min-recall 0.8 --max-latency 3000`.

## Recommended Next Steps
1. Keep the single-document command handy for stubborn PDFs; it is safe to rerun because chunk IDs are deterministic.
2. After any batch of retries, rebuild the FTS index so lexical search stays in sync with Qdrant.
3. Schedule periodic `--changed-only` bulk ingests so newly added source files are picked up automatically.
