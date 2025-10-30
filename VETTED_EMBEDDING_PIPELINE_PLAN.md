# Embedding + Vector Store Plan (Vetted & Improved)

This document vets the original plan and provides an improved, executable plan for building a local ingestion + embeddings + vector store pipeline using Ollama embeddings + TEI reranker with Qdrant.

---

## 0) Executive Summary

- Cost: Self‑hosted Qdrant is free; Qdrant Cloud has a small free tier (verify current limits). API key on self‑hosted Qdrant is for auth only, not metering.
- Store: Qdrant with cosine/dot/euclid metrics and HNSW ANN.
- Extraction: Use Docling for high‑fidelity PDFs/OCR; MarkItDown for broad formats. Both run locally.
- Pipeline: Walk directory → extract → chunk → embed with Ollama (`snowflake-arctic-embed:xs`) → upsert to Qdrant → store `path` and offsets for downstream tools; optional TEI reranker for cross-encoder re-ranking.
- Includes concrete fixes (dynamic embedding dim probe, Qdrant collection creation, path handling, batching, and validation steps).

---

## 1) Vetting Notes (Accuracy Check)

- Qdrant API key: Correct. Self‑hosted: optional and free, static token via env/config. No metering.
- Qdrant Cloud: The “free” tier historically grants a small instance (e.g., ~1 GB). Confirm current pricing/limits before relying on it.
- Licensing: Qdrant OSS Apache‑2.0.
- Extractors: MarkItDown and Docling claims match their docs (wide format support, OCR, layout, Markdown export).
- Ollama embedding: use `/api/embed` with `{model, input}`; supports batch lists and returns `embeddings`. Prefer CPU-friendly models like `snowflake-arctic-embed:xs`.
- Script correctness:
  - File paths: use `Path(...).resolve().as_posix()` for WSL-compatible paths; if URIs are needed elsewhere, `Path(...).resolve().as_uri()` can be derived.
  - Qdrant: Ensure the collection exists (right size/metric) before upsert; handle (re)creation or mismatch gracefully.
  - Chunking: Ensure forward progress with `i += max_chars - overlap`.
  - Batching: Embed in batches to avoid oversize payloads/timeouts.
  - Metrics: Only L2‑normalize embeddings when using cosine distance.

---

## 2) Improved Plan (Step‑by‑Step)

1) Environment & Dependencies
- Create venv and install: `markitdown[all]`, `docling`, `qdrant-client`, `requests`, `numpy`, `tqdm`.
- Install and run Ollama locally; pull `snowflake-arctic-embed:xs`.
- Start Qdrant: `docker run --rm -p 6333:6333 -e QDRANT__SERVICE__API_KEY=<key_optional> qdrant/qdrant`
- Optional: start TEI reranker `BAAI/bge-reranker-base` on :8087.

2) Storage Schema (Qdrant)
- One collection per scope (e.g., `snowflake_kb`), `size=<model dim>`, `distance` according to your metric.
- Optional: scalar/product quantization after initial load to reduce RAM.
- Consider aliases for zero‑downtime swaps.

3) Ingestion Script Improvements
- Extraction
  - Default extractor: MarkItDown. Auto mode: Docling for PDFs/scans/images, MarkItDown for Office/HTML/CSV/etc.
  - Allow `--ext` allowlist and a `--skip` pattern.
- Chunking
  - Implement `i += max_chars - overlap` with bounds checks; avoid empty/duplicate chunks.
- Embedding
  - Use Ollama `/api/embed` with batch lists; add `--batch-size` (e.g., 32–128) and retry with backoff.
  - Normalize on output iff metric is cosine.
- Qdrant Path
  - On first use, probe Ollama for embedding dimension and check/create collection with specified `size`/`distance`.
  - Use `upsert` with payload including `doc_id`, `path`, `chunk_start`, `chunk_end`, `text`, plus optional `filename`, `mtime`, `mime`.
- Metadata & IDs
  - Deterministic `doc_id` via `uuid5(NAMESPACE_URL, Path.as_uri())`.
  - IDs as `f"{doc_id}:{start}-{end}"` are fine (Qdrant supports string IDs; Postgres uses text primary key).
- File URIs
  - Use `Path(path).resolve().as_uri()`.
- Logging & Resume (optional)
  - Add `--skip-existing` to avoid re‑embedding files already present (look up by `doc_id`).

4) Validation & Smoke Tests
- After ingest of a small folder, run a test search via client and ensure non‑empty hits with sensible payload fields.
- Verify payload includes `path` and offsets so downstream tools can open the source document.

5) Operations
- Qdrant: take snapshots/backups, monitor RAM; consider quantization for RAM savings; use aliases to roll collections.
- TEI: pin image tag/sha, set appropriate concurrency; monitor latency.

6) Security & Cost
- Qdrant API key on self‑host is free; restrict network access. For Cloud, confirm current free tier limits.

---

## 3) Concrete Artifacts

### 3.1 Ollama Embeddings

Use the local Ollama server:

```bash
ollama pull snowflake-arctic-embed:xs
# API: POST /api/embed with {"model":"snowflake-arctic-embed:xs","input":[...texts...]}
```

### 3.2 Qdrant: Create/Ensure Collection (Python)

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", api_key=None)
name = "snowflake_kb"
size = 1024

try:
    info = client.get_collection(name)
    if info.config.params.vectors.size != size or info.config.params.vectors.distance != models.Distance.COSINE:
        # recreate if mismatch
        client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
        )
except Exception:
    client.recreate_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )
```

### 3.3 Ingestion Script – Key Fixes to Apply

- Use `Path(...).resolve().as_uri()` for file URIs.
- Create/ensure Qdrant collection before `upsert`.
- Batch `/embed` requests with `--batch-size` and retry.
- Implement chunking as `i += max_chars - overlap`.
- Only normalize embeddings for cosine metric.

(If you want, I can patch your `ingest.py` accordingly.)

---

## 4) Quickstart Commands

```bash
# 1) Python env
python -m venv .venv && source .venv/bin/activate
pip install "markitdown[all]" docling qdrant-client requests numpy tqdm

# 2) Start services
# Qdrant
docker run --rm -p 6333:6333 qdrant/qdrant
# Optional: TEI reranker on 8087
docker run --rm -p 8087:80 ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id BAAI/bge-reranker-base

# 3) Run ingestion (Ollama embeddings)
ollama pull snowflake-arctic-embed:xs
python ingest.py --root ./docs --ollama-url http://localhost:11434 --ollama-model snowflake-arctic-embed:xs \
  --extractor auto \
  --qdrant-url http://localhost:6333 --qdrant-collection snowflake_kb
```

---

## 5) Retrieval Sanity Checks

- Qdrant (Python):
  - `client.search(collection_name, query_vector=embed([query])[0], limit=5)` and inspect `payload` fields.

---

## 6) Notes & Trade‑offs

- For large TEI models (e.g., Stella‑en‑1.5B‑v5), CPU performance is slow; prefer Ollama `snowflake-arctic-embed:xs` on CPU for speed.
- Quantization in Qdrant reduces RAM substantially; apply after correctness validation.
- Keeping `path` + offsets in payload enables your agents/MCP servers to fetch exact source context.

---

## 7) Next Steps (Optional)

- Wire ingestion to your MCP `add_documents_*` tools instead of direct DB writes for full agent observability.
- Add a lightweight resume/index audit command (`--dry-run`, `--skip-existing`).
- Add a small search CLI for manual validation.
