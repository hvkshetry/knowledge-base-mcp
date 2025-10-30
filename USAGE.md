# Usage Guide

Comprehensive guide for ingesting documents and searching your knowledge base.

## Table of Contents

- [Document Ingestion](#document-ingestion)
- [Search Modes](#search-modes)
- [Multi-Collection Setup](#multi-collection-setup)
- [Advanced Features](#advanced-features)
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
  --extractor auto                    # auto | markitdown | docling

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

### Extraction Modes

#### Auto Mode (Recommended)
Automatically selects the best extractor for each file type.

```bash
python ingest.py --root docs/ --collection kb --extractor auto
```

- **PDFs**: Tries PyMuPDF → MarkItDown → Docling (if previous fails)
- **Office docs** (.docx, .xlsx, .pptx): MarkItDown
- **HTML, CSV**: MarkItDown
- **Images in PDFs**: Docling with OCR

#### MarkItDown Mode
Fast extraction for most document types.

```bash
python ingest.py --root docs/ --collection kb --extractor markitdown
```

**Pros**: Fast, handles many formats
**Cons**: Basic PDF extraction, no OCR

**Best for**: Office documents, HTML, clean PDFs

#### Docling Mode
High-fidelity PDF processing with OCR and layout preservation.

```bash
python ingest.py --root docs/ --collection kb --extractor docling
```

**Pros**: Excellent PDF handling, OCR for scanned documents, layout-aware
**Cons**: Slower (10-20x), higher memory usage

**Best for**: Technical PDFs with diagrams, scanned documents, complex layouts

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

The MCP server provides three search modes with different trade-offs.

### Semantic Search

**Method**: Pure vector similarity using dense embeddings

**Use Cases**:
- Conceptual queries without specific keywords
- Finding related content
- Exploratory research

**Example**:
```json
{
  "query": "How do microorganisms break down organic matter?",
  "mode": "semantic",
  "top_k": 8
}
```

Will find content about biological degradation, even if it uses terms like "biodegradation," "microbial metabolism," or "decomposition."

**Parameters**:
- `retrieve_k`: Initial vector search results (default: 24)
- `top_k`: Final results returned (default: 8)

**Pros**: Great recall, finds conceptually similar content
**Cons**: May miss exact keyword matches, can return loosely related results

### Rerank Search (Default)

**Method**: Vector retrieval + cross-encoder reranking

**Use Cases**:
- Most general-purpose searches
- Balance of speed and accuracy
- When you need better precision than pure semantic search

**Example**:
```json
{
  "query": "stainless steel corrosion resistance",
  "mode": "rerank",
  "retrieve_k": 24,
  "return_k": 12,
  "top_k": 5
}
```

**Parameters**:
- `retrieve_k`: Initial vector retrieval size (default: 24)
- `return_k`: Results to rerank (default: 8)
- `top_k`: Final results after reranking (default: 8)

**How it works**:
1. Retrieve `retrieve_k` results via vector search
2. Take top `return_k` results
3. Rerank with cross-encoder (query-document interaction)
4. Return top `top_k`

**Pros**: Better precision than semantic, fast
**Cons**: Limited to vector search recall

### Hybrid Search

**Method**: RRF fusion of vector + BM25 lexical search + reranking

**Use Cases**:
- Complex queries with both conceptual and specific terms
- Technical searches with jargon or model numbers
- Highest quality results (at cost of speed)

**Example**:
```json
{
  "query": "FilmTec BW30-400 membrane cleaning pH 11.5",
  "mode": "hybrid",
  "retrieve_k": 48,
  "return_k": 16,
  "top_k": 8
}
```

Combines:
- **Semantic**: Understands "membrane cleaning" concept
- **Lexical**: Exact match on "BW30-400", "pH 11.5"
- **Reranking**: Selects best overall results

**Parameters**:
- `retrieve_k`: Results from EACH ranker (vector + BM25) (default: 24)
- `return_k`: Fused results to rerank (default: 8)
- `top_k`: Final results (default: 8)

**Pros**: Best result quality, combines semantic + keyword matching
**Cons**: Slowest (~2x rerank mode), requires FTS database

### Search Mode Comparison

| Mode | Speed | Recall | Precision | Best For |
|------|-------|--------|-----------|----------|
| **Semantic** | Fast | High | Medium | Exploratory, conceptual queries |
| **Rerank** | Medium | Medium | High | General-purpose, most queries |
| **Hybrid** | Slow | Highest | Highest | Complex technical queries |

### Parameter Tuning Examples

#### Quick Search (Speed Priority)
```json
{
  "mode": "rerank",
  "retrieve_k": 12,
  "return_k": 8,
  "top_k": 5
}
```

#### Comprehensive Search (Quality Priority)
```json
{
  "mode": "hybrid",
  "retrieve_k": 48,
  "return_k": 16,
  "top_k": 10
}
```

#### High Precision (Few Very Relevant Results)
```json
{
  "mode": "hybrid",
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

## Performance Tuning

### Ingestion Performance

#### Faster Ingestion
```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --embed-batch-size 64 \      # Larger batches
  --workers 8 \                 # More parallel workers
  --extractor markitdown        # Skip Docling
```

**Speed**: 10-15 pages/second

#### Higher Quality (Slower)
```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --embed-batch-size 16 \       # Smaller batches (more stable)
  --workers 2 \                  # Fewer workers
  --extractor docling           # Best PDF extraction
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

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for technical deep dive
- Check [FAQ.md](FAQ.md) for common questions
- Explore [examples/](examples/) for sample scripts
- Read [CONTRIBUTING.md](CONTRIBUTING.md) to contribute improvements
