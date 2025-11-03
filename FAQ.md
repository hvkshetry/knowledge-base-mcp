# Frequently Asked Questions (FAQ)

## General Questions

### What is this project?

A production-grade MCP (Model Context Protocol) server that provides hybrid semantic search over private document collections. It combines vector embeddings, lexical search (BM25), and cross-encoder reranking for high-quality results, all running locally. **Key benefit: zero-cost embeddings and reranking** - no per-document or per-query charges.

### Why use local embeddings and reranking?

- **Zero Embedding Cost**: Use Ollama locally - no per-document charges for embeddings during ingestion
- **Zero Reranking Cost**: Local TEI reranking - no per-query charges for result reranking
- **Unlimited Scale**: Ingest and search unlimited documents without incremental API costs
- **Speed**: <300ms search latency on our reference hardware (hybrid route; actual latency depends on your CPU and rerank configuration) - no API roundtrips for embeddings or reranking
- **Control**: Full customization of embedding models, search parameters, and chunking

### What document formats are supported?

- **PDFs**: Including scanned documents with OCR
- **Microsoft Office**: .docx, .xlsx, .pptx
- **Text files**: .txt, .md, .csv
- **Web**: .html, .htm
- **Images** (in PDFs): OCR via Docling

### How does it compare to cloud solutions?

| Feature | This Project | Cloud (e.g., Pinecone + OpenAI) |
|---------|-------------|----------------------------------|
| **Embedding Cost** | $0 (local Ollama) | $0.0001-0.0004 per 1K tokens |
| **Reranking Cost** | $0 (local TEI) | $0.002-0.01 per query |
| **Storage Cost** | Local hardware only | $0.25-0.50 per GB/month |
| **Total Cost** | Claude subscription + hardware | Claude + per-document + per-query + storage |
| **Search Latency** | <300ms | 500-2000ms |
| **Internet Required** | Yes (for Claude API) | Yes |
| **Scale** | Limited by local hardware | Unlimited (cloud infrastructure) |

## Installation & Setup

### What are the hardware requirements?

**Minimum**:
- 4GB RAM
- 2 CPU cores
- 5GB disk space

**Recommended**:
- 8GB RAM
- 4+ CPU cores
- 10GB+ disk space

**For large collections** (10,000+ pages):
- 16GB+ RAM
- 8+ CPU cores
- 50GB+ disk space

### Can I use GPU acceleration?

Currently optimised for CPU out of the box. GPU support for Ollama embeddings is possible (not enabled in the default docker-compose):

```bash
# Use Ollama with GPU (requires NVIDIA GPU + CUDA)
# Ollama automatically detects and uses GPU if available
ollama pull snowflake-arctic-embed:xs
```

For reranking, you'd need to modify `docker-compose.yml` to use TEI GPU image (requires NVIDIA Docker).

### Do I need Docker?

Yes, Docker is currently required for:
- **Qdrant**: Vector database
- **TEI Reranker**: Cross-encoder reranking

*Future enhancement*: Optional local alternatives without Docker.

### How do I add another knowledge base (collection)?

Edit the `NOMIC_KB_SCOPES` environment variable (or the corresponding entry in `.mcp.json` / `.codex/config.toml`) and add a new slug-to-collection mapping, for example:

```json
{"kb": {"collection": "daf_kb", "title": "DAF Knowledge Base"}, "ro": {"collection": "ro_kb", "title": "Reverse Osmosis"}}
```

Only `daf_kb` is configured by default so that the repository ships without proprietary data. After updating `NOMIC_KB_SCOPES`, ingest documents into the new collection and pass `collection="<slug>"` when calling the `kb.*` tools.

### Can I run this on Windows?

Yes, via **WSL2** (Windows Subsystem for Linux). Native Windows support is possible but untested. We recommend WSL2 for best compatibility.

### How much disk space do I need?

Approximate storage per 1000 pages:
- **Qdrant vectors**: 3-5MB
- **SQLite FTS**: 2-4MB
- **Total**: ~5-10MB per 1000 pages

For 100,000 pages: ~500MB-1GB

## Ingestion

### How long does ingestion take?

Depends on hardware and extractor:

**MarkItDown** (fast):
- 10-15 pages/second
- 1000 pages: ~2-3 minutes
- 10,000 pages: ~20-30 minutes

**Docling** (high quality):
- 2-5 pages/second
- 1000 pages: ~5-10 minutes
- 10,000 pages: ~1-2 hours

### What if ingestion fails partway through?

Use incremental flags:

```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --skip-existing  # Skip already processed files
```

Or for changed-only detection:

```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --changed-only --delete-changed
```

### Can I ingest while searching?

Yes! Ingestion and search are independent. You can search existing documents while ingesting new ones.

### How do I update documents?

Use incremental ingestion:

```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --changed-only --delete-changed
```

This detects changed files by content hash and reingests only those.

### What if a document fails to extract?

Use `--embed-robust` mode:

```bash
python ingest.py \
  --root /docs \
  --collection kb \
  --embed-robust  # Skip failed chunks, continue processing
```

This processes as much as possible, skipping problematic sections.

## Search

### Which search mode should I use?

**Quick guide**:
- **Most queries**: Use `rerank` (default)
- **Exploratory/conceptual**: Use `semantic`
- **Technical with specific terms**: Use `hybrid`

See [USAGE.md](USAGE.md) for detailed comparison.

### Why am I getting poor results?

**Checklist**:
1. **Collection name correct?** Verify matches ingestion
2. **Documents ingested?** Check: `curl http://localhost:6333/collections/{collection}`
3. **Try different mode**: Test semantic, rerank, and hybrid
4. **Increase retrieve_k**: More candidates = better chance of finding match
5. **Check FTS database**: For hybrid mode, ensure FTS DB exists

### Can I search across multiple collections?

Not in a single query. Each MCP tool searches one collection. However, Claude can query multiple tools and synthesize results.

### How can I improve search quality?

1. **Use hybrid mode**: Best overall quality
2. **Increase retrieve_k**: More candidates (e.g., 48)
3. **Increase return_k**: More reranking (e.g., 16)
4. **Better chunking**: Adjust `--chunk-size` for your content
5. **Enable neighbor expansion**: Provides more context (default: 1)

### Why is hybrid search slow?

Hybrid mode performs:
1. Vector search (Qdrant)
2. Lexical search (SQLite FTS)
3. RRF fusion
4. Cross-encoder reranking

Each step adds latency. Typical: ~200-300ms (vs ~100ms for rerank).

## Client Orchestration

### Why doesn't the server generate summaries or HyDE hypotheses?

Because the MCP client (Claude) already has best-in-class generation capabilities at no marginal cost. The server stays deterministic—extracting text, running search, and storing results—while the client reads evidence and produces summaries or hypotheses.

### How does hybrid per-page routing work without server LLM calls?

The server supplies page-level heuristics and confidence scores (`ingest.analyze_document`). For high-confidence pages the suggested extractor is used as-is; for low-confidence pages the client inspects metadata (`table_tokens`, `text_density`, `sample_text`) and chooses the extractor before calling `ingest.extract_with_strategy`.

### What is client orchestration?

Client orchestration means Claude plans the ingestion/search strategy—generating HyDE passages, selecting extractors, and writing summaries—while the server executes deterministically. This keeps the system reproducible, avoids extra API costs, and ensures the highest-quality model is making decisions.

### Does this add latency or cost?

No. Claude is already in the loop; examining metadata for the ~10% low-confidence pages takes seconds. Avoiding server-side LLM calls removes the need to run additional LLM infrastructure on the backend.

## Configuration

### How do I add a new collection?

1. **Ingest documents**:
```bash
python ingest.py --root /docs --collection new_kb --fts-db data/new_kb_fts.db
```

2. **Update MCP config** (`.mcp.json`):
```json
{
  "env": {
    "NOMIC_KB_SCOPES": "{
      \"existing_kb\": {...},
      \"new\": {
        \"collection\": \"new_kb\",
        \"title\": \"New Knowledge Base\"
      }
    }"
  },
  "autoApprove": ["search_existing_kb", "search_new"]
}
```

3. **Restart MCP client** (Claude Desktop or Codex)

### Can I use different embedding models?

Yes! Any Ollama model that supports embeddings:

```bash
# Pull alternative model
ollama pull nomic-embed-text

# Update MCP config
"OLLAMA_MODEL": "nomic-embed-text"
```

**Note**: Changing models requires reingesting documents.

### Can I use different rerankers?

Yes, modify `docker-compose.yml`:

```yaml
reranker:
  image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.8
  command: --model-id BAAI/bge-reranker-v2-m3  # Different model
```

### How do I backup my data?

**Qdrant** (vectors):
```bash
# Create snapshot
docker exec qdrant /qdrant/qdrant snapshot create

# Copy from container
docker cp qdrant:/qdrant/storage ./backup/qdrant_storage
```

**SQLite FTS** (lexical index):
```bash
cp data/*.db ./backup/
```

**Re-create from source**: Keep source documents; reingestion recreates everything.

## Performance

### How can I speed up ingestion?

```bash
python ingest.py \
  --embed-batch-size 64 \      # Larger batches (if memory allows)
  --parallel 8 \                # More workers (match CPU cores)
  --extractor markitdown        # Faster than Docling
```

### How can I speed up search?

1. **Use semantic mode**: Fastest (~100ms)
2. **Reduce retrieve_k**: Fewer candidates (e.g., 12)
3. **Reduce return_k**: Less reranking (e.g., 8)
4. **Disable neighbor expansion**: Set `NEIGHBOR_CHUNKS=0`

### My system is running out of memory

**During ingestion**:
```bash
python ingest.py \
  --embed-batch-size 16 \       # Smaller batches
  --parallel 2 \                 # Fewer workers
  --max-docs-per-run 50          # Process in batches
```

**During search**: Reduce `retrieve_k` and `return_k`

### Can I use this for millions of documents?

Yes, but hardware requirements scale:
- **100K pages**: 8GB RAM sufficient
- **1M pages**: 16GB+ RAM recommended
- **10M pages**: 32GB+ RAM, consider distributed Qdrant

Qdrant supports horizontal scaling for very large collections.

## Troubleshooting

### Services won't start

**Qdrant**:
```bash
# Check if port is in use
lsof -i :6333

# Check logs
docker-compose logs qdrant
```

**Reranker**:
```bash
# Check if port is in use
lsof -i :8087

# Check logs
docker-compose logs reranker

# Verify health
curl http://localhost:8087/health
```

### Ollama connection refused

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (macOS/Linux)
ollama serve

# Check if model is available
ollama list
```

### MCP server not connecting

1. **Check Python path** in `.mcp.json` is absolute and correct
2. **Verify virtual environment** is activated
3. **Test server directly**:
```bash
source .venv/bin/activate
python server.py stdio
# Try sending a test MCP message
```
4. **Check Claude Desktop logs** (macOS): `~/Library/Logs/Claude/mcp*.log`

### Search returns no results

```bash
# Verify collection exists
curl http://localhost:6333/collections/{collection_name}

# Check point count (should be > 0)
curl http://localhost:6333/collections/{collection_name} | jq '.result.points_count'

# Verify FTS database (for hybrid mode)
sqlite3 data/{collection_name}_fts.db "SELECT COUNT(*) FROM fts_chunks;"
```

### Embeddings are very slow

1. **Check Ollama is using GPU** (if available):
```bash
ollama ps
# Should show GPU usage if available
```

2. **Reduce batch size**:
```bash
python ingest.py --embed-batch-size 16 ...
```

3. **Use smaller model**:
```bash
ollama pull snowflake-arctic-embed:xs  # Smallest, fastest
```

### Extraction fails for some PDFs

Try different extractors:

```bash
# Try Docling (best for complex PDFs)
python ingest.py --extractor docling ...

# Or use auto mode (tries multiple extractors)
python ingest.py --extractor auto ...

# Or skip problematic files
python ingest.py --embed-robust ...
```

## Advanced Topics

### Can I customize the chunking strategy?

Yes, via command-line args:

```bash
python ingest.py \
  --chunk-size 2400 \      # Larger chunks for more context
  --overlap 200            # More overlap for continuity
```

**Guidelines**:
- **Technical docs**: 1800-2400 chars
- **Legal docs**: 2400-3000 chars (need more context)
- **Short content**: 1200-1500 chars

### Can I use custom distance metrics?

Yes, specify during ingestion:

```bash
python ingest.py \
  --metric cosine          # Default, best for normalized embeddings
  # OR
  --metric euclidean       # Alternative
  # OR
  --metric dot             # For non-normalized embeddings
```

**Note**: Cannot change metric after ingestion; must recreate collection.

### How does time decay work?

Configure via environment variables:

```bash
DECAY_HALF_LIFE_DAYS=180   # Documents lose 50% boost after 180 days
DECAY_STRENGTH=0.3         # 30% weight to recency vs relevance
```

Recent documents get a score boost. Useful for:
- Frequently updated documentation
- Time-sensitive content
- Version control (prefer latest)

### Can I run multiple MCP servers?

Yes! Each collection can have its own server instance, or one server can handle multiple collections (via `NOMIC_KB_SCOPES`).

### How does RRF fusion work?

Reciprocal Rank Fusion combines rankings from vector and lexical search:

```
score(doc) = Σ [1 / (K + rank_i(doc))]
```

Where `K=60` (default). This normalizes different score scales and combines rankings fairly.

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Areas needing help:
- Testing framework
- Additional document extractors
- Performance optimization
- Documentation improvements
- Example use cases

### I found a bug, what should I do?

1. **Check if it's already reported**: Search GitHub Issues
2. **Gather information**: Logs, environment details, reproduction steps
3. **Create issue**: Use bug report template in [CONTRIBUTING.md](CONTRIBUTING.md)

### I have a feature idea

Create a GitHub Issue with:
- Use case description
- Proposed solution
- Why it would be useful

---

## Still have questions?

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and ideas
- **Documentation**: Check [README.md](README.md), [INSTALLATION.md](INSTALLATION.md), [USAGE.md](USAGE.md), [ARCHITECTURE.md](ARCHITECTURE.md)
