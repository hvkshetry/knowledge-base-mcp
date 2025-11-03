# Usage Guide

Comprehensive guide for ingesting documents and searching your knowledge base.

## Table of Contents

- [Document Ingestion](#document-ingestion)
- [MCP Ingestion Workflow](#mcp-ingestion-workflow)
- [Search Modes](#search-modes)
- [Multi-Collection Setup](#multi-collection-setup)
- [Advanced Features](#advanced-features)
- [MCP Playbooks](#mcp-playbooks)
- [Operational Acceptance Checks](#operational-acceptance-checks)
- [Performance Tuning](#performance-tuning)
- [Best Practices](#best-practices)

## Document Ingestion

The `ingest.py` script processes documents and creates searchable indexes.

### Basic Ingestion

```bash
python ingest.py \
  --root /path/to/documents \
  --collection my_collection \
  --ext .pdf,.docx,.txt
```

### Complete Parameter Reference

```bash
python ingest.py \
  # Required
  --root /path/to/documents           # Directory to scan
  --collection my_collection          # Qdrant collection name

  # Document Selection
  --ext .pdf,.docx,.txt               # File extensions (comma-separated)
  --include "*/important/*"           # Include glob patterns
  --skip "*/drafts/*,*.tmp"           # Exclude glob patterns
  --max-file-mb 64                    # Max file size in MB
  --min-words 50                      # Min words to index document
  --max-walk-depth 10                 # Directory traversal depth

  # Extraction
  --extractor auto                    # Primary extractor for text fallback
  --sparse-expander splade            # Optional sparse expansion (none|basic|splade)
  --colbert-url http://localhost:7000 # Optional ColBERT service endpoint
  --thin-payload                      # Store metadata-only payloads in Qdrant
  --graph-db data/graph.db            # Override graph storage (optional)
  --summary-db data/summary.db        # Override summary storage (optional)

  # Chunking
  --chunk-size 1800                   # Characters per chunk
  --chunk-overlap 150                 # Overlap between chunks

  # Embedding
  --ollama-url http://localhost:11434 # Ollama API endpoint
  --ollama-model snowflake-arctic-embed:xs  # Embedding model
  --embed-batch-size 48               # Embeddings per batch
  --embed-robust                      # Skip failed chunks, continue
  --workers 4                         # Parallel embedding workers

  # Qdrant
  --qdrant-url http://localhost:6333  # Qdrant endpoint
  --qdrant-api-key YOUR_KEY           # Optional authentication
  --metric cosine                     # cosine | euclidean | dot

  # FTS (Full-Text Search)
  --fts-db data/my_collection_fts.db  # SQLite FTS database path

  # Incremental Processing
  --skip-existing                     # Skip if doc_id exists
  --changed-only                      # Only process changed files
  --delete-changed                    # Delete old chunks before reingest
  --max-docs-per-run 100              # Batch processing limit
```

### Extraction & Triage

- **Page-level triage is automatic**. Each PDF page is scored for tables, multi-column layouts, or scans. Light pages stay on MarkItDown; heavy pages route through Docling for table/figure-aware blocks with bounding boxes and captions.
- **Docling caching**. Page-level extraction artifacts are cached under `.ingest_cache/` so re-ingests reuse prior work. Set `CACHE_DIR` if you want a different location.
- **Fallback extractor (`--extractor`)** still matters for non-PDF formats and for Docling failures—it controls the initial text extraction used for hashing and as a safety net.
- **Hugging Face cache (`HF_HOME`)** should point to a writable directory so Docling can download its layout models.

## MCP Ingestion Workflow

The MCP server exposes deterministic ingestion tools that mirror the CLI pipeline while producing auditable artifacts. Each call writes JSON under `data/ingest_artifacts/<doc_id>/` and keeps the canonical plan in `data/ingest_plans/<doc_id>.plan.json`.

| Tool | Purpose | Key Outputs |
| ---- | ------- | ----------- |
| `ingest.analyze_document(path)` | Run triage to decide Docling vs. MarkItDown per page and seed a plan hash. | plan, route counts |
| `ingest.extract_with_strategy(path, plan)` | Extract structural blocks using the stored or provided plan override. | blocks artifact path |
| `ingest.chunk_with_guidance(artifact_ref, profile)` | Apply a deterministic chunker (`auto`, `heading_based`, `procedure_block`, `table_row`, `fixed_window`). | chunk artifact, sample chunk previews |
| `ingest.generate_metadata(doc_id)` | Produce bounded metadata (<=320-char summary, <=8 concepts, typed entities) with guardrails. | doc metadata, metadata bytes/calls |
| `ingest.assess_quality(doc_id)` | Inspect chunk counts, table-row coverage, metadata status, and warnings. | quality metrics, plan hash |
| `ingest.enhance(doc_id, op)` | Reserved for future safe adjustments (currently returns `not_implemented`). | — |

### Guardrails

- `INGEST_MODEL_VERSION` / `INGEST_PROMPT_SHA` annotate every plan and artifact (defaults: `structured_ingest_v1`, `sha_deterministic_chunking_v1`).
- `MAX_METADATA_BYTES` (default `8192`) rejects metadata that exceeds the byte budget and reports `metadata_bytes_exceeded`.
- `MAX_METADATA_CALLS_PER_DOC` (default `2`) prevents unbounded metadata regeneration; exceeding it yields `metadata_budget_exceeded`.

### Typical Flow

1. `ingest.analyze_document` to review triage routes.
2. `ingest.extract_with_strategy` to cache structural blocks.
3. `ingest.chunk_with_guidance` (`profile="auto"` for most docs, `"table_row"` for tabular-heavy content).
4. `ingest.generate_metadata` to attach deterministic metadata (fast-fails if size/call limits are hit).
5. `ingest.assess_quality` to verify chunk density, table coverage, and metadata status before calling the CLI ingest or a custom upsert routine.

Artifacts are plain JSON, so CI can diff plans, inspect `plan_hash` stability, or replay the same strategy across environments.

### Incremental Ingestion Patterns

#### First-Time Ingestion
Process all documents:

```bash
python ingest.py \
  --root /path/to/documents \
  --collection my_kb \
  --ext .pdf,.docx
```

#### Daily Updates (Changed Files Only)
Only process new or modified files:

```bash
python ingest.py \
  --root /path/to/documents \
  --collection my_kb \
  --ext .pdf,.docx \
  --changed-only \
  --delete-changed
```

**How it works**:
1. Computes SHA256 hash of extracted text
2. Compares with existing chunks in Qdrant
3. If hash differs, deletes old chunks and reingests
4. If hash matches, skips

#### Skip Existing Documents
Never reprocess documents that are already indexed:

```bash
python ingest.py \
  --root /path/to/documents \
  --collection my_kb \
  --ext .pdf,.docx \
  --skip-existing
```

Uses `doc_id` (UUID5 from file path) to check existence.

#### Batch Processing (Memory-Constrained)
Process large collections in batches:

```bash
python ingest.py \
  --root /large/collection \
  --collection big_kb \
  --ext .pdf \
  --max-docs-per-run 50 \
  --skip-existing
```

Run repeatedly until all documents are processed.

### File Filtering

#### Include Only Specific Patterns
```bash
python ingest.py \
  --root /documents \
  --collection kb \
  --include "*/2024/*,*/important/*"
```

#### Exclude Patterns
```bash
python ingest.py \
  --root /documents \
  --collection kb \
  --skip "*/drafts/*,*/archive/*,*.tmp"
```

**Default exclusions** (automatic):
- Hidden files: `*/.*`
- Temp files: `*/~$*`, `*.tmp`, `*.temp`
- System files: `*/Thumbs.db`, `*/Desktop.ini`
- Office temp: `~/\$*`

#### Size and Content Filters
```bash
python ingest.py \
  --root /documents \
  --collection kb \
  --max-file-mb 100 \          # Skip files > 100MB
  --min-words 100              # Skip documents < 100 words
```

### Multi-Format Example

Process a mixed document collection:

```bash
python ingest.py \
  --root /company_docs \
  --collection company_kb \
  --ext .pdf,.docx,.xlsx,.pptx,.html,.txt \
  --extractor auto \
  --skip "*/~\$*,*/drafts/*" \
  --max-file-mb 50 \
  --changed-only \
  --delete-changed \
  --fts-db data/company_kb_fts.db
```

## Search Modes

### Semantic Search (`mode="semantic"`)

**Method**: Pure vector similarity using dense embeddings.

**Use Cases**: Conceptual queries, exploratory research, related-idea discovery.

**Pros**: High recall, finds semantically similar passages even without shared keywords.

**Cons**: May miss exact term matches.

### Rerank Search (`mode="rerank"`, default)

**Method**: Dense retrieval followed by TEI cross-encoder reranking.

**Pros**: Strong precision with moderate latency; ideal general-purpose mode.

**Cons**: Recall limited to the dense candidate pool.

### Hybrid Search (`mode="hybrid"`)

**Method**: Reciprocal Rank Fusion between dense and BM25, then rerank.

**Pros**: Strongest balance of recall + precision for technical queries; handles part numbers, abbreviations, and concepts.

**Cons**: Slowest (dense + BM25 + rerank).

### Sparse Search (`mode="sparse"`)

**Method**: BM25 with domain alias expansion; no dense retrieval.

**Pros**: Fast, perfect for short keyword/ID searches or as a fallback when dense misses.

**Cons**: No semantic understanding.

### Auto Planner (`mode="auto"`)

The default MCP mode. Heuristics pick a route based on query shape:

1. Dense or hybrid retrieval runs first.
2. If the top result underperforms, the planner triggers a HyDE (hypothetical passage) rerun and, if needed, a sparse retry before abstaining.

| Query Type | Suggested Mode |
|------------|----------------|
| Short keyword / part number | `sparse` or `auto` |
| Conceptual (“how does…”) | `semantic` or `auto` |
| Detailed question | `rerank` or `auto` |
| Technical spec with IDs + prose | `hybrid` or `auto` |

### Parameter Tuning Examples

- **Quick search:** `mode=rerank`, `retrieve_k=12`, `return_k=8`, `top_k=5`.
- **High-recall technical search:** `mode=hybrid`, `retrieve_k=48`, `return_k=16`, `top_k=10`.
- **Sparse fallback:** `mode=sparse`, `retrieve_k=32`, `top_k=8` for terse part numbers.
  "retrieve_k": 32,
  "return_k": 12,
  "top_k": 3
}
```

## Multi-Collection Setup

Organize documents into separate knowledge bases with dedicated search tools.

### Configuration

Edit your MCP config (`.mcp.json` or Claude Desktop config):

```json
{
  "env": {
    "NOMIC_KB_SCOPES": "{
      \"technical\": {
        \"collection\": \"engineering_docs\",
        \"title\": \"Engineering Documentation\"
      },
      \"legal\": {
        \"collection\": \"legal_docs\",
        \"title\": \"Legal Research\"
      },
      \"sales\": {
        \"collection\": \"sales_kb\",
        \"title\": \"Sales Knowledge Base\"
      }
    }"
  },
  "autoApprove": [
    "search_technical",
    "search_legal",
    "search_sales"
  ]
}
```

This creates three MCP tools:
- `search_technical`
- `search_legal`
- `search_sales`

### Ingesting Multiple Collections

```bash
# Ingest engineering documents
python ingest.py \
  --root /docs/engineering \
  --collection engineering_docs \
  --ext .pdf,.docx \
  --fts-db data/engineering_fts.db

# Ingest legal documents
python ingest.py \
  --root /docs/legal \
  --collection legal_docs \
  --ext .pdf,.docx \
  --fts-db data/legal_fts.db

# Ingest sales materials
python ingest.py \
  --root /docs/sales \
  --collection sales_kb \
  --ext .pdf,.pptx \
  --fts-db data/sales_fts.db
```

### Collection-Specific FTS Databases

Each collection should have its own FTS database for hybrid search:

```json
{
  "env": {
    "NOMIC_KB_SCOPES": "...",
    "FTS_DB_PATH": "data/default_fts.db"
  }
}
```

The server automatically looks for `data/{collection_name}_fts.db` first, then falls back to `FTS_DB_PATH`.

**Directory structure**:
```
data/
├── engineering_docs_fts.db
├── legal_docs_fts.db
└── sales_kb_fts.db
```

## Advanced Features

### Structurally-Aware Ingestion

- Per-page triage automatically routes heavy PDF pages (tables, multi-column, scans) to Docling. Tune sensitivity via `ROUTE_HEAVY_FRACTION`, `SMALL_DOC_DOCLING`, and `MULTICOL_GAP_FACTOR`.
- `DOCLING_TIMEOUT` guards against pathological pages; timed-out pages fall back to MarkItDown output.
- Page artifacts are cached under `.ingest_cache/` (override with `CACHE_DIR`).
- Set `HF_HOME` to a writable cache directory so Docling’s layout models download once.

### Sparse Expansion (SPLADE / Basic)

- Enable via CLI `--sparse-expander splade` (defaults to env `SPARSE_EXPANDER`), falling back to a lightweight TF-style expander when SPLADE isn’t available.
- During ingest each chunk stores a sparse term-weight dictionary in the FTS sidecar (`fts_chunks_sparse`).
- Query-time, call `kb.sparse_splade_{slug}` (or let auto-route pick it) to use the expanded terms; MCP clients can mix with BM25 by calling both `kb.sparse_{slug}` and `kb.sparse_splade_{slug}`.
- Scores surface as `sparse_score` within the standard `scores` bucket so quality gates continue to work.

### ColBERT Late Interaction

- Configure a ColBERT service (e.g., `colbertv2` + Faiss) and set `COLBERT_URL` (and optional `COLBERT_TIMEOUT`).
- `kb.colbert_{slug}` invokes the service with `POST {COLBERT_URL}/query` and expects `{"results": [{"chunk_id": ..., "doc_id": ..., "score": ...}, ...]}`.
- Auto routing (`mode="auto"`) will prefer ColBERT for question-style queries when the service is available; call it explicitly when you want late-interaction reranking.
- Returned rows include `colbert_score` under `scores.dense`, so quality gates can reason about confidence alongside other routes.

### Docling Operations & GPU Tuning

- `python scripts/manage_cache.py status` shows the size of `.ingest_cache/`, `graph.db`, and `summary.db`; use `clear-ingest-cache`, `clear-graph`, or `clear-summary` to remove them safely.
- Set `DOCLING_DEVICE` (`cpu` or `cuda`) and `DOCLING_BATCH_SIZE` before ingestion to control Docling execution; default is CPU with batch size 1.
- `INGEST_CACHE_DIR` overrides the cache location if you prefer a faster disk or shared volume.

### Neighbor Context Expansion

Automatically includes adjacent chunks for better context.

**Configuration** (environment variable):
```bash
NEIGHBOR_CHUNKS=1  # Include 1 chunk before and after (3 total)
```

**How it works**:
1. Search finds best matching chunk
2. Retrieves N chunks before and after from same document
3. Concatenates into single result

**Use Cases**:
- Improve context understanding
- Avoid truncated information
- Better for Claude to understand full topic

**Trade-offs**:
- Increases result length
- May include less relevant content
- Slightly slower

### Time Decay Scoring

Boost recent documents in search results.

**Configuration**:
```bash
DECAY_HALF_LIFE_DAYS=180  # Documents lose 50% boost after 180 days
DECAY_STRENGTH=0.3        # 0.0-1.0, weight of recency vs relevance
```

**Formula**:
```
final_score = (1 - strength) × relevance_score + strength × time_decay_score
```

**Use Cases**:
- Prefer recent versions of updated documents
- Prioritize current regulations/standards
- Time-sensitive information (news, research)

**Disable** (default):
```bash
DECAY_HALF_LIFE_DAYS=0
DECAY_STRENGTH=0.0
```

### Answerability Threshold

Filter out low-confidence results.

**Configuration**:
```bash
ANSWERABILITY_THRESHOLD=0.5  # Require rerank score ≥ 0.5
```

**Use Cases**:
- Avoid irrelevant results
- High-precision applications
- When "no results" is better than wrong results

**Trade-off**: May return empty results for marginal queries

### Reranker Constraints

Control what's sent to the reranker:

```bash
RERANK_MAX_CHARS=700   # Max characters per chunk
RERANK_MAX_ITEMS=16    # Max number of items to rerank
```

**Why?**
- Cross-encoders have token limits
- Very long chunks may be truncated
- More items = slower reranking

**Tuning**:
- Increase `RERANK_MAX_CHARS` for long-form content
- Increase `RERANK_MAX_ITEMS` with powerful hardware
- Decrease both for faster searches

### Graph & Summary Indices

Two lightweight stores are built during ingest:

- `GRAPH_DB_PATH` – documents, sections, chunks, and heuristic entity nodes linked by `contains`/`mentions` edges. Use `entities_{slug}` to browse by type, `linkouts_{slug}` to jump straight to supporting chunks, and `graph_{slug}` for raw neighbor walks.
- `SUMMARY_DB_PATH` – RAPTOR-style section synopses with element IDs. Use `summary_{slug}` to retrieve.

**Entity tips**
- Table rows add parameter nodes automatically; `linkouts_*` returns the chunk ids so you can follow up with `open_*`.
- Frequent co-occurrences between entities are stored as `co_occurs` edges, giving agent planners a cheap signal for likely relationships (e.g., equipment ↔ operating parameter).
- Numeric readings matched in chunks (e.g., “MLSS = 8000 mg/L”) become `measurement` nodes linked to their parameter via `has_measurement`, so you can query values directly from the graph.

### MCP Utility Tools

- `open_{slug}`: fetch a specific chunk by `chunk_id`/`element_id`, optionally slicing text.
- `neighbors_{slug}`: pull BM25 neighbors around a chunk for more context.
- `summary_{slug}`: retrieve stored section summaries + citations.
- `entities_{slug}`: list graph entities for the collection, filterable by `type` and substring.
- `linkouts_{slug}`: enumerate chunks/sections that reference an entity node (pair with `open_*`).
- `graph_{slug}`: inspect the neighbourhood in the knowledge graph for any node id.
- `sparse_splade_{slug}`: run sparse expansion retrieval (SPLADE/basic) on demand.
- `colbert_{slug}`: route a query through the ColBERT late-interaction service when configured.
- `ingest.assess_quality`: returns chunk stats plus canary query results (configured under `config/canaries/`).
- `ingest.enhance`: supports safe incremental fixes (`add_synonyms`, `link_crossrefs`, `fix_table_pages`) without full re-ingest.

All tools hydrate snippets through the ACL-enforcing document store.

## MCP Playbooks

Use the composable tools above to let the MCP client drive retrieval and critique loops. See [`MCP_PLAYBOOKS.md`](MCP_PLAYBOOKS.md) for ready-to-run sequences:

- **Retrieve → Assess → Refine**: combine `kb.hybrid`, `kb.quality`, `kb.open`, and `kb.hint` / `kb.hyde` to iterate safely.
- **Table QA**: pair `kb.table` with `kb.open` and `kb.quality` to guarantee grounded tabular answers.
- **Graph Walks**: `kb.entities` → `kb.linkouts` → `kb.graph` uncovers multi-hop evidence.
- **Ingestion QA**: `ingest.analyze_document` → `ingest.extract_with_strategy` → `ingest.chunk_with_guidance` → `ingest.assess_quality` ensures deterministic plans.

These playbooks can be embedded directly into Claude prompts, Codex CLI scripts, or other MCP clients.

### Evaluation Guardrails

Run gold-set evaluations locally or in CI:

```bash
python eval.py \
  --gold eval/gold_sets/my_kb.jsonl \
  --mode auto \
  --fts-db data/my_kb_fts.db \
  --min-ndcg 0.85 --min-recall 0.80 --max-latency 3000
```

`eval.py` exits non-zero when any threshold fails, making it easy to wire into CI/CD pipelines.

### RRF Configuration

Tune Reciprocal Rank Fusion for hybrid search:

```bash
HYBRID_RRF_K=60  # Default: 60
```

**RRF Formula**:
```
score(doc) = Σ [1 / (K + rank_i(doc))]
```

**Effect of K**:
- **Lower K** (20-40): More weight to top-ranked results
- **Higher K** (80-100): More uniform weighting across ranks

**When to tune**:
- If vector search dominates: Decrease K
- If lexical search dominates: Increase K
- Default (60) works well for most cases

## Operational Acceptance Checks

### Stage 0 – Provenance & Thin Payload

1. **Backfill provenance into FTS**
   ```bash
   python scripts/backfill_fts_from_qdrant.py \
     --collection daf_kb \
     --fts-db data/daf_kb_fts.db
   ```
   Confirm `page_numbers`, `section_path`, `element_ids`, `table_headers`, and `table_units` survive a rebuild:
   ```bash
   sqlite3 data/daf_kb_fts.db 'SELECT doc_id, page_numbers, section_path, types, element_ids FROM fts_chunks LIMIT 3;'
   ```
2. **Toggle thin payload mode**
   ```bash
   THIN_PAYLOAD=true python ingest.py \
     --root ./daf_kb \
     --collection daf_kb \
     --fts-only
   ```
   Inspect a Qdrant point (via the UI or the API) to confirm payloads only carry provenance while `open_{slug}` still dereferences text.

### Stage 1 – Deterministic Ingestion Plans

1. **Plan replay**
   ```bash
   python -m fastmcp.client call ingest.analyze_document path=/abs/path/to/file.pdf
   python -m fastmcp.client call ingest.extract_with_strategy path=/abs/path/to/file.pdf plan=@last.json
   python -m fastmcp.client call ingest.chunk_with_guidance artifact_ref=@blocks.json profile=auto
   ```
   Re-run the sequence—`plan_hash`, chunk previews, and `data/ingest_plans/<doc>.plan.json` should be identical.
2. **Table indexing sanity check**
   ```bash
   python -m fastmcp.client call table_daf_kb query="design MLSS" limit=3
   ```
   Rows return `type="table_row"` with headers/units. Follow up with `open_daf_kb` to fetch the exact span.
3. **Metadata budget + quality**
   ```bash
   python -m fastmcp.client call ingest.generate_metadata doc_id=<uuid>
   python -m fastmcp.client call ingest.assess_quality doc_id=<uuid>
   ```
   `metadata_calls`, `plan_hash`, and any `metadata_rejects` are persisted under `data/ingest_plans/` for auditing.

## Performance Tuning

### Ingestion Performance

#### Faster Ingestion
```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --embed-batch-size 64 \      # Larger batches
  --workers 8 \                 # More parallel workers
  --extractor markitdown        # Keep fallback fast; triage will still escalate complex pages
```

**Speed**: 10-15 pages/second

#### Higher Quality (Slower)
```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --embed-batch-size 16 \       # Smaller batches (more stable)
  --workers 2 \                  # Fewer workers
  --extractor docling           # Force Docling even for light pages
```

**Speed**: 2-5 pages/second

#### Memory-Constrained
```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --embed-batch-size 8 \
  --workers 1 \
  --max-docs-per-run 20
```

### Search Performance

#### Faster Searches
```json
{
  "mode": "rerank",
  "retrieve_k": 12,
  "return_k": 8,
  "top_k": 5
}
```
**Latency**: ~100ms

#### Balanced
```json
{
  "mode": "rerank",
  "retrieve_k": 24,
  "return_k": 12,
  "top_k": 8
}
```
**Latency**: ~150ms

#### Best Quality
```json
{
  "mode": "hybrid",
  "retrieve_k": 48,
  "return_k": 16,
  "top_k": 10
}
```
**Latency**: ~300ms

### Hardware Optimization

#### CPU-Bound (Typical)
Optimize embedding and reranking:

```bash
# docker-compose.yml
environment:
  OMP_NUM_THREADS: 8           # Match your CPU cores
```

Use CPU-friendly models:
- Embedding: `snowflake-arctic-embed:xs` (384-dim)
- Reranker: `BAAI/bge-reranker-base`

#### Memory-Limited
Reduce batch sizes and chunk size:

```bash
python ingest.py \
  --embed-batch-size 8 \
  --chunk-size 1200
```

#### Storage-Limited
Use smaller chunk sizes and overlap:

```bash
python ingest.py \
  --chunk-size 1200 \
  --chunk-overlap 100
```

**Typical storage**:
- 1000 pages: ~5-10MB (Qdrant + FTS)
- 10,000 pages: ~50-100MB
- 100,000 pages: ~500MB-1GB

## Best Practices

### Ingestion

1. **Start with MarkItDown**, upgrade to Docling if needed
2. **Use incremental ingestion** for regular updates (`--changed-only --delete-changed`)
3. **Filter aggressively**: Use `--skip` to exclude temp files, drafts
4. **Batch large collections**: Use `--max-docs-per-run` for memory management
5. **One collection per topic**: Separate engineering, legal, sales, etc.
6. **Consistent chunk sizes**: Keep `--chunk-size` same within collection

### Search

1. **Start with rerank mode**: Best balance for most queries
2. **Use hybrid for technical queries**: When exact terms matter
3. **Tune retrieve_k**: Higher for exploratory, lower for precision
4. **Enable neighbor expansion**: Better context for Claude (default: 1)
5. **Test different modes**: Run `validate_search.py` with each mode

### Configuration

1. **Separate FTS databases**: One per collection
2. **Use absolute paths**: In MCP configs (avoid relative paths)
3. **Auto-approve search tools**: Add to `autoApprove` for better UX
4. **Set environment variables**: Use `.env` file for complex configs
5. **Version your config**: Keep `.mcp.json` in version control (template)

### Maintenance

1. **Monitor disk usage**: Qdrant + FTS databases grow with content
2. **Periodic reingestion**: Run `--changed-only` weekly/monthly
3. **Check service health**: `docker-compose ps` and check logs
4. **Update models**: New Ollama models improve quality
5. **Backup Qdrant**: `docker cp qdrant:/qdrant/storage ./backup`

### Troubleshooting

1. **No results**: Check collection name, verify documents ingested
2. **Poor quality results**: Try hybrid mode, increase retrieve_k
3. **Slow searches**: Reduce retrieve_k, use semantic mode
4. **High memory**: Reduce batch sizes, fewer workers
5. **Extraction failures**: Use `--embed-robust` to skip problematic chunks

## Example Workflows

### Workflow 1: Technical Documentation Library

```bash
# Ingest
python ingest.py \
  --root /technical_docs \
  --collection tech_kb \
  --ext .pdf,.docx \
  --extractor auto \
  --changed-only --delete-changed \
  --fts-db data/tech_kb_fts.db

# Search (hybrid for technical terms)
# Use MCP client with mode="hybrid"
```

### Workflow 2: Legal Document Research

```bash
# Ingest
python ingest.py \
  --root /legal_docs \
  --collection legal_kb \
  --ext .pdf,.docx \
  --extractor docling \              # High quality for legal docs
  --chunk-size 2400 \                # Larger chunks for legal context
  --changed-only --delete-changed \
  --fts-db data/legal_kb_fts.db

# Search (hybrid for case citations and statutes)
# Use MCP client with mode="hybrid", higher retrieve_k
```

### Workflow 3: Personal Research Library

```bash
# Ingest
python ingest.py \
  --root ~/Documents/Research \
  --collection research_kb \
  --ext .pdf,.docx,.txt \
  --extractor auto \
  --skip-existing \
  --fts-db data/research_kb_fts.db

# Search (semantic for exploratory research)
# Use MCP client with mode="semantic"
```

## Security & Governance

- `ALLOWED_DOC_ROOTS` (path list separated by `:`) restricts dereferencing to known directories. Retrieval tools such as `open_*` and `neighbors_*` refuse snippets outside these roots.
- `AUDIT_LOG_PATH` writes JSONL audit entries for every retrieval event (`search`, `open`, `neighbors`, `table_lookup`, etc.) with hashed subject identifiers and telemetry for downstream monitoring.
- Enabling `THIN_PAYLOAD` strips raw text from vector payloads; snippets are rehydrated from the FTS database after ACL checks so sensitive content is only materialized when explicitly requested.

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for technical deep dive
- Check [FAQ.md](FAQ.md) for common questions
- Explore [examples/](examples/) for sample scripts
- Read [CONTRIBUTING.md](CONTRIBUTING.md) to contribute improvements
