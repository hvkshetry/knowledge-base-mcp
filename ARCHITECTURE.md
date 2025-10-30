# Architecture Deep Dive

This document provides a technical deep dive into the system architecture, algorithms, and design decisions.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Search Pipeline](#search-pipeline)
- [Algorithms](#algorithms)
- [Design Decisions](#design-decisions)
- [Performance Characteristics](#performance-characteristics)

## System Overview

The Semantic Search MCP Server is a hybrid RAG (Retrieval-Augmented Generation) system combining multiple retrieval strategies for high-quality document search.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Client Layer                      │
│              (Claude Desktop / Codex CLI)                │
└────────────────────┬────────────────────────────────────┘
                     │ stdio (MCP Protocol)
                     ↓
┌─────────────────────────────────────────────────────────┐
│                 MCP Server (server.py)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │          Multi-Scope Search Router                 │  │
│  │   (search_kb, search_collection1, search_...)     │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │             Search Mode Dispatcher                 │  │
│  │        (semantic / rerank / hybrid)                │  │
│  └───────────────────────────────────────────────────┘  │
└─────┬───────────────┬───────────────┬──────────────┬────┘
      │               │               │              │
      ↓               ↓               ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Qdrant  │   │ SQLite   │   │  Ollama  │   │   TEI    │
│  Vector  │   │   FTS5   │   │Embeddings│   │ Reranker │
│    DB    │   │  (BM25)  │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
      ↑               ↑               ↑              ↑
      │               │               │              │
      └───────────────┴───────────────┴──────────────┘
                      │
              ┌───────┴────────┐
              │  ingest.py     │
              │  (Ingestion    │
              │   Pipeline)    │
              └───────┬────────┘
                      │
              ┌───────┴────────┐
              │   Documents    │
              │  (PDF, DOCX,   │
              │   TXT, etc.)   │
              └────────────────┘
```

## Component Architecture

### 1. MCP Server (server.py)

**Framework**: FastMCP (MCP protocol implementation)

**Key Components**:
- **Scope Registry**: Maps collection names to search functions
- **Search Dispatcher**: Routes requests to appropriate search mode
- **Result Processor**: Formats and enriches results
- **Neighbor Expander**: Retrieves adjacent chunks for context

**Responsibilities**:
- Handle MCP protocol communication
- Manage multiple collection scopes
- Execute search strategies
- Aggregate and format results

### 2. Ingestion Pipeline (ingest.py)

**Stages**:
1. **Discovery**: Walk directory tree, filter files
2. **Extraction**: Convert documents to plain text
3. **Chunking**: Split text into overlapping segments
4. **Embedding**: Generate vector representations
5. **Storage**: Persist to Qdrant and SQLite FTS

**Key Features**:
- Incremental ingestion with change detection
- Multi-format document support
- Parallel processing
- Robust error handling
- Batch memory management

### 3. Vector Store (Qdrant)

**Purpose**: Store and search dense vector embeddings

**Schema**:
```python
{
    "id": str,                # UUID4 chunk identifier
    "vector": List[float],    # Embedding (384-dim for snowflake:xs)
    "payload": {
        "doc_id": str,        # UUID5 from file URI
        "path": str,          # Absolute POSIX path
        "chunk_start": int,   # Character offset
        "chunk_end": int,     # Character offset
        "text": str,          # Chunk content
        "filename": str,      # File basename
        "mtime": int,         # File modification timestamp
        "content_hash": str   # SHA256 of full document text
    }
}
```

**Index**: HNSW (Hierarchical Navigable Small World)
- Fast approximate nearest neighbor search
- Trade-off: Speed vs accuracy (configurable)

### 4. Lexical Index (SQLite FTS5)

**Purpose**: Traditional keyword search with BM25 ranking

**Schema**:
```sql
CREATE VIRTUAL TABLE fts_chunks USING fts5(
    text,                    -- Full-text indexed
    chunk_id UNINDEXED,      -- Chunk UUID
    doc_id UNINDEXED,        -- Document UUID
    path UNINDEXED,          -- File path
    filename UNINDEXED,      -- File basename
    chunk_start UNINDEXED,   -- Character offset
    chunk_end UNINDEXED,     -- Character offset
    mtime UNINDEXED,         -- Modification time
    tokenize = 'unicode61 remove_diacritics 2'
);
```

**Features**:
- BM25 ranking algorithm
- Unicode normalization
- Diacritic removal for better matching

### 5. Embedding Service (Ollama)

**Purpose**: Generate vector embeddings for text

**Model**: snowflake-arctic-embed:xs (default)
- **Dimensions**: 384
- **Context**: 512 tokens
- **Performance**: ~10ms per chunk on CPU
- **Quality**: Good for technical documents

**API**: Batch endpoint (`/api/embed`)
- Process multiple chunks in single request
- Reduces overhead
- Configurable batch size

### 6. Reranking Service (TEI)

**Purpose**: Re-score candidates using cross-encoder

**Model**: BAAI/bge-reranker-base (default)
- **Type**: Cross-encoder (processes query + document together)
- **Context**: 512 tokens
- **Performance**: ~90ms for 16 candidates on CPU
- **Quality**: Significantly improves precision

**Why Cross-Encoder**:
- Bi-encoders (for vector search): Fast but less accurate
- Cross-encoders (for reranking): Slower but much more accurate
- Two-stage retrieval gets best of both worlds

## Ingestion Pipeline

### Document Discovery

```python
for root, dirs, files in os.walk(args.root):
    # Apply depth limit
    # Apply skip patterns (globs)
    # Filter by extension
    # Check file size
    # Yield discovered files
```

**Filtering**:
- Extension whitelist: `--ext .pdf,.docx,...`
- Skip patterns: `--skip "*/drafts/*,*.tmp"`
- Size limit: `--max-file-mb 64`
- Depth limit: `--max-walk-depth 10`

### Text Extraction

**Extractors**:

1. **PyMuPDF** (fitz):
   - Fast PDF text extraction
   - First attempt for PDFs
   - Falls back if fails or returns little text

2. **MarkItDown**:
   - Multi-format support (Office, HTML, CSV, PDF)
   - Fast, lightweight
   - Good for most documents

3. **Docling**:
   - High-fidelity PDF extraction
   - OCR for scanned documents
   - Layout-aware (preserves tables, lists)
   - Slower, more memory

**Extractor Selection** (auto mode):
```
IF PDF:
    TRY PyMuPDF
    IF insufficient_text OR error:
        TRY MarkItDown
        IF insufficient_text OR error:
            TRY Docling (if not disabled)
ELSE:
    USE MarkItDown
```

### Chunking Strategy

**Algorithm**: Sliding window with character-based splitting

```python
chunk_size = 1800      # Characters per chunk
overlap = 150          # Overlap between chunks

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append({
        "text": chunk,
        "chunk_start": i,
        "chunk_end": i + len(chunk)
    })
```

**Why Character-Based**:
- Simple and deterministic
- Works for all languages
- No dependency on tokenizer
- Consistent across document types

**Why Sliding Window**:
- Ensures no information lost at boundaries
- Overlap provides context continuity
- Small overlap (8.3%) minimizes duplication

**Trade-offs**:
- May split mid-sentence
- No semantic boundaries
- Alternative: Sentence-based chunking (more complex)

### Embedding Generation

**Batch Processing**:
```python
batch_size = 48
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    embeddings = ollama_embed_batch(batch)
```

**Normalization**:
```python
if metric == "cosine":
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

Only normalize for cosine distance (already normalized in dot product space).

**Robust Mode**:
- Process chunks in smaller windows
- Skip failed chunks
- Continue processing rest of document
- Useful for large/problematic documents

### Storage

**Qdrant**:
```python
qdrant_client.upsert(
    collection_name=collection,
    points=[
        PointStruct(
            id=chunk_id,
            vector=embedding,
            payload=metadata
        )
        for chunk_id, embedding, metadata in zip(...)
    ]
)
```

**SQLite FTS**:
```python
cursor.executemany(
    "INSERT INTO fts_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
    [(chunk.text, chunk_id, doc_id, path, ...) for chunk in chunks]
)
```

## Search Pipeline

### Semantic Search Mode

**Algorithm**:
```
1. Embed query with Ollama
2. Search Qdrant for nearest neighbors
3. Retrieve top_k results
4. Optionally expand with neighbor chunks
5. Apply time decay (if enabled)
6. Return results
```

**Complexity**: O(log N) for HNSW index

### Rerank Search Mode

**Algorithm**:
```
1. Embed query with Ollama
2. Search Qdrant for nearest neighbors (retrieve_k results)
3. Truncate to return_k results
4. Rerank with cross-encoder
5. Take top_k results
6. Optionally expand with neighbor chunks
7. Apply time decay (if enabled)
8. Return results
```

**Why Two-Stage**:
- Stage 1 (vector): Fast, high recall
- Stage 2 (rerank): Slow, high precision
- Best results without searching entire collection

### Hybrid Search Mode

**Algorithm**:
```
1. PARALLEL:
   a. Embed query and search Qdrant (retrieve_k results)
   b. Search SQLite FTS (retrieve_k results)

2. Reciprocal Rank Fusion (RRF):
   For each document:
     rrf_score = Σ [1 / (K + rank_i)]
   Where i ∈ {qdrant_rank, fts_rank}

3. Take top return_k by RRF score

4. Rerank with cross-encoder

5. Take top_k results

6. Optionally expand with neighbor chunks

7. Apply time decay (if enabled)

8. Return results
```

**Complexity**: O(log N) for both searches + O(retrieve_k log retrieve_k) for sorting

## Algorithms

### Reciprocal Rank Fusion (RRF)

**Purpose**: Combine rankings from multiple retrieval systems

**Formula**:
```
RRF(d, K) = Σ [1 / (K + r_i(d))]
```

Where:
- `d` = document
- `K` = constant (default 60)
- `r_i(d)` = rank of document `d` in ranker `i`
- Sum over all rankers

**Example**:
```
Document A:
- Vector rank: 3
- BM25 rank: 1
- RRF score: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323

Document B:
- Vector rank: 1
- BM25 rank: 5
- RRF score: 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

Winner: Document A (better combined ranking)
```

**Why RRF**:
- No score normalization needed (works with raw ranks)
- Robust to outliers
- Fair to all rankers
- Simple and effective

**Alternative Considered**: Linear combination of scores
- Requires score normalization
- Sensitive to score scales
- More tuning needed

### Neighbor Context Expansion

**Purpose**: Include adjacent chunks for better context

**Algorithm**:
```python
def expand_with_neighbors(chunk, n_neighbors=1):
    # Find chunks from same document within offset range
    neighbors_before = find_chunks(
        doc_id=chunk.doc_id,
        chunk_end <= chunk.chunk_start,
        limit=n_neighbors
    )
    neighbors_after = find_chunks(
        doc_id=chunk.doc_id,
        chunk_start >= chunk.chunk_end,
        limit=n_neighbors
    )

    # Concatenate
    return "\n\n".join([
        *[n.text for n in neighbors_before],
        chunk.text,
        *[n.text for n in neighbors_after]
    ])
```

**Benefits**:
- Provides fuller context
- Avoids fragmented information
- Improves downstream LLM comprehension

**Trade-offs**:
- Increased result length
- May include less relevant content
- Slight performance overhead

### Time Decay Scoring

**Purpose**: Boost recent documents in results

**Formula**:
```
decay_factor(t) = 2^(-age_days / half_life_days)

final_score = (1 - strength) × relevance_score + strength × decay_factor
```

Where:
- `t` = document modification time
- `age_days` = days since modified
- `half_life_days` = configured half-life (e.g., 180)
- `strength` = weight of recency (0.0-1.0)

**Example**:
```
half_life_days = 180
strength = 0.3

Document A: relevance=0.9, age=30 days
- decay = 2^(-30/180) = 0.89
- final = 0.7×0.9 + 0.3×0.89 = 0.897

Document B: relevance=0.85, age=365 days
- decay = 2^(-365/180) = 0.24
- final = 0.7×0.85 + 0.3×0.24 = 0.667

Winner: Document A (recency boost)
```

**Use Cases**:
- Prefer updated versions
- Time-sensitive content
- Frequently changing documentation

## Design Decisions

### Why Qdrant?

**Alternatives Considered**:
- Faiss: No built-in metadata filtering
- Weaviate: Heavier, more complex
- Pinecone: Cloud-only, not local
- ChromaDB: Less mature at time of design

**Qdrant Advantages**:
- Excellent local deployment
- Rich metadata filtering
- HNSW index (fast + accurate)
- Good documentation
- Active development

### Why SQLite FTS5?

**Alternatives Considered**:
- Elasticsearch: Too heavy for local deployment
- Whoosh: Pure Python, slower
- Tantivy: Rust-based, more complex integration

**SQLite FTS5 Advantages**:
- Built-in to Python
- Fast BM25 implementation
- Minimal overhead
- Single-file database
- Battle-tested

### Why Snowflake Arctic Embed?

**Alternatives Considered**:
- OpenAI ada-002: Cloud-only, costs money
- all-MiniLM-L6-v2: Lower quality
- Stella-en-1.5B-v5: Too slow on CPU
- nomic-embed-text: Good alternative

**Snowflake Advantages**:
- CPU-friendly (small model)
- Good quality for technical docs
- Fast inference (~10ms/chunk)
- Well-documented
- Open source

### Why Cross-Encoder Reranking?

**Alternative**: Use only bi-encoder (vector search)

**Why Reranking Wins**:
- 10-30% improvement in precision
- Marginal latency cost (~90ms)
- Catches nuanced relevance
- Better for complex queries

**Why Not Rerank Everything**:
- O(N) complexity to rerank all documents
- Cross-encoders are slower
- Two-stage gives best speed/quality trade-off

### Why Character-Based Chunking?

**Alternatives**:
- Sentence-based: Requires sentence tokenizer
- Token-based: Model-specific
- Semantic: Complex, slow, subjective

**Character Advantages**:
- Simple, deterministic
- Language-agnostic
- No dependencies
- Consistent behavior

**Trade-off**: May split mid-sentence, but overlap mitigates this.

### Why Multi-Scope Design?

**Alternative**: Single collection with metadata filtering

**Multi-Scope Advantages**:
- Cleaner organization
- Independent tuning per collection
- Better MCP tool semantics
- Simpler filtering logic

**Trade-off**: More configuration needed

## Performance Characteristics

### Ingestion Performance

**Bottlenecks**:
1. **Text Extraction**: Docling (slow), PyMuPDF (fast)
2. **Embedding**: Ollama API calls
3. **I/O**: Disk writes (minimal with batching)

**Scaling**:
- Linear with document count
- Parallel workers help (4-8 typical)
- Batch size tuning important (16-64)

**Typical Rates**:
- MarkItDown: 10-15 pages/sec
- Docling: 2-5 pages/sec
- Limiting factor: Usually embedding generation

### Search Performance

**Latency Components**:

**Semantic Mode**:
- Query embedding: 10ms
- Qdrant search: 20-50ms
- Neighbor expansion: 10-20ms
- **Total**: ~50-100ms

**Rerank Mode**:
- Query embedding: 10ms
- Qdrant search: 20-50ms
- Reranking: 70-100ms
- Neighbor expansion: 10-20ms
- **Total**: ~110-180ms

**Hybrid Mode**:
- Query embedding: 10ms
- Qdrant search: 20-50ms
- FTS search: 30-60ms (parallel)
- RRF fusion: 5-10ms
- Reranking: 70-100ms
- Neighbor expansion: 10-20ms
- **Total**: ~150-250ms

**Scaling**:
- Qdrant: O(log N) with HNSW
- FTS: O(log N) with indexes
- Reranking: O(k) where k is small (8-16)
- Overall: Sub-linear scaling

### Storage Requirements

**Per 1000 Chunks** (~1000 pages):
- Vectors (384-dim): ~3-5 MB
- Payload JSON: ~1-2 MB
- FTS index: ~2-4 MB
- **Total**: ~6-11 MB

**Scaling**: Linear with content volume

### Memory Usage

**Ingestion**:
- Base: ~500 MB (Ollama model loaded)
- Per document batch: ~50-200 MB
- Peak: ~1-2 GB for typical batches

**Search**:
- Base: ~200 MB (server + models)
- Per query: ~10-50 MB
- Peak: ~500 MB

**Docker Services**:
- Qdrant: ~200-500 MB
- TEI Reranker: ~500 MB - 1 GB
- **Total**: ~1-2 GB

## Future Enhancements

**Potential Improvements**:
1. **GPU Acceleration**: Support TEI GPU image, optimize for CUDA
2. **Semantic Chunking**: Use LLM to split at semantic boundaries
3. **Query Expansion**: Automatically enhance queries
4. **Result Caching**: Cache frequent queries
5. **Distributed Qdrant**: Scale to billions of documents
6. **Multi-Modal**: Support images, audio extraction
7. **Incremental Reranking**: Rerank only changed results
8. **Alternative Embeddings**: Support API-based models (OpenAI, Cohere)

---

This architecture balances quality, performance, and simplicity for local deployment. Each component is swappable for experimentation while maintaining clean interfaces.
