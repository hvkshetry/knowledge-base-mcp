# MCP Agent Prompt (Ingestion + Retrieval)

## Role
- The server executes deterministic ingestion/search primitives. It never calls an LLM.
- **You (the MCP client)** choose chunkers, retry strategies, HyDE hypotheses, and summaries, then call the appropriate MCP tools.
- Always record decisions in `client_decisions` / `client_orchestration` fields so the plan artifacts explain what you changed.

## Ingestion Workflow

### For Large-Scale Ingestion (Recommended)
For bulk ingestion (10+ documents, large PDFs, or production pipelines), recommend the CLI to the user instead of MCP tools to avoid token costs.

#### Full Re-ingestion (Clean Slate)
Use when rebuilding the collection from scratch or when major changes require complete reprocessing:

```bash
.venv/bin/python3 ingest.py --root /path/to/documents --qdrant-collection my_collection --max-chars 700 --batch-size 128 --parallel 1 --ollama-threads 4 --fts-db data/my_collection_fts.db --fts-rebuild
```

**What this does:**
- Processes ALL documents in the directory
- Rebuilds FTS database from scratch (`--fts-rebuild`)
- Upserts vectors to Qdrant (deterministic chunk IDs mean duplicates are replaced, not added)

#### Incremental Update (Upsert Changed/New Only)
Use for daily/weekly updates to add new documents and update changed ones WITHOUT reprocessing everything:

```bash
.venv/bin/python3 ingest.py --root /path/to/documents --qdrant-collection my_collection --max-chars 700 --batch-size 128 --parallel 1 --ollama-threads 4 --fts-db data/my_collection_fts.db --changed-only --delete-changed
```

**What this does:**
- Computes SHA256 hash of each document's extracted text
- Compares with existing chunks in Qdrant
- **Only processes documents that are new or have changed content**
- Deletes old chunks before reinserting changed documents (`--delete-changed`)
- Skips unchanged documents entirely (massive time savings)

**When to use each approach:**
- **Full re-ingestion**: First-time setup, changing chunk size, major refactoring
- **Incremental update**: Daily/weekly maintenance, adding new docs to existing collection

**Critical CLI notes for users:**
- `--fts-db` MUST match collection name (e.g., `data/my_collection_fts.db` for `--qdrant-collection my_collection`)
- Otherwise data goes to default `data/fts.db` causing confusion
- `--extractor` flag removed - Docling is now the only extractor
- `--batch-size 128` (new default) for better embedding throughput vs old default of 32
- `--max-chars 700` recommended for reranker compatibility vs old default 1800
- `--parallel 1` for single-threaded ingestion (safer for large PDFs)
- `--ollama-threads 4` controls Ollama embedding parallelism (adjust based on CPU cores)

### For Interactive MCP-Based Ingestion
Use MCP tools for small-scale interactive work (1-10 documents):

1. **Extract** – `ingest.extract_with_strategy(path=..., plan={})`
   - Docling-only processing (no routing, no triage)
   - Full-document extraction in single call
   - Produces `blocks.json` artifacts per doc_id under `data/ingest_artifacts/`

2. **Chunk** – `ingest.chunk_with_guidance(artifact_ref=..., profile=...)`
   - Profiles: `heading_based`, `procedure_block`, `table_row`, `fixed_window`
   - Output includes `chunk_profile`, `plan_hash`, `headers`, `units`, `element_ids` and raw text

3. **Metadata & Summaries (client authored)**
   - `ingest.generate_metadata(doc_id=..., policy="strict_v1")` only when needed; respect byte/call budgets
   - Generate hierarchical summaries yourself:
     - Group chunks by `section_path` (leaf sections first), summarise 3–5 sentences per section with citations
     - Roll up section summaries into parent levels (chapter → section → subsection) before calling `ingest.generate_summary(...)`
   - Persist each summary via `ingest.generate_summary(...)`, including `model`, `prompt_sha`, and any decision notes in `client_decisions`

4. **Quality Gates** – `ingest.assess_quality(doc_id=...)`
   - Executes configured canaries; abort the run if any required condition fails

5. **Enhance (optional)** – `ingest.enhance(doc_id=..., op=...)`
   - Safe post-processing only: `add_synonyms`, `link_crossrefs`, `fix_table_pages`

6. **Upsert**
   - `ingest.upsert(...)` for a single doc or `ingest.upsert_batch(...)` for small batches
   - Provide `client_decisions` so replay logs show what changed

## Extraction Implementation Details
**Breaking change from previous versions:**
- All extractor routing removed (no `markitdown`, no `pymupdf`, no per-page triage)
- `ingest.extract_with_strategy()` always uses Docling for full-document processing
- Docling processes entire PDF in single call (no per-page splitting overhead)
- ~60-65% faster than old per-page routing approach

**Metadata preservation:**
- `table_headers`: Column names from table structure
- `table_units`: Units parsed from headers (e.g., "μm", "mg/L")
- `bboxes`: Bounding box coordinates for tables and figures
- `types`: Block types (`table_row`, `figure`, `heading`, `para`, `list`)
- `source_tools`: Always `['docling']`
- `section_path`: Document structure hierarchy

All metadata preserved in both Qdrant vector payloads and FTS database.

## Retrieval Workflow
1. **Choose the route**
   - Default to `kb.search(mode="auto", ...)` or `kb.hybrid`
   - Alternatives: `kb.dense`, `kb.sparse`, `kb.rerank`, `kb.colbert` (if configured), `kb.sparse_splade` (needs SPLADE)

2. **Inspect evidence**
   - `kb.open`, `kb.neighbors` for context and citations
   - `kb.table` for row-level answers, `kb.summary` / `kb.outline` if summaries built
   - Graph pivots: `kb.entities`, `kb.linkouts`, `kb.graph`

3. **Quality gating** – `kb.quality(collection=..., min_score=..., require_plan_hash=True, require_table_hit=bool)`
   - If below threshold: rerun with `kb.hint` + `kb.sparse`, rephrase via `kb.batch`, or abstain

4. **HyDE retry (client-side)**
   - No `kb.hyde` tool exists
   - After low-score pass (e.g., best score < 0.35), draft 5–7 sentence hypothetical answer
   - Re-run with `kb.dense(query=hypothesis, retrieve_k=..., return_k=...)` and compare telemetry
   - Adopt if improves recall while meeting answerability gates; otherwise revert or abstain

5. **Session priors**
   - `kb.promote` / `kb.demote` once you've verified document quality in this session

## Best Practices for Context Retrieval with kb.neighbors

### Why You Must Expand Every Search Result
With the recommended chunk size of 700 characters (optimized for reranker compatibility), **complete context is distributed across neighboring chunks**. A single chunk rarely contains the full information needed to answer a query comprehensively.

**CRITICAL WORKFLOW - Apply to ALL searches**:
1. Use `kb.search` or `kb.hybrid` to locate the **most relevant chunk(s)**
2. **ALWAYS use `kb.neighbors(chunk_id=..., n=10)`** to retrieve surrounding context
3. Analyze the expanded context (21 chunks total) to understand the full picture

**Default neighbor radius**: `n=10` (retrieves 10 chunks before and after the reference chunk)

### What Gets Distributed Across Neighbors at Chunk Size 700
- **Tables**: Data rows typically 3-10 chunks away from table captions/headers
- **Procedures**: Multi-step instructions split across multiple chunks
- **Definitions**: Term definition separated from usage examples
- **Arguments**: Claims in one chunk, supporting evidence in neighbors
- **Figures**: Captions separated from figure descriptions
- **Context**: Background information before/after the key point

### Why n=10 is the Recommended Default
- **Table data distribution**: With 700-char chunks, table rows are typically 3-10 chunks away from the table reference
- **Token limit safety**: n=10 returns ~19,000-20,000 tokens (safely under the 25,000 token MCP response limit)
- **n=15 risks timeout**: Exceeds 25,000 token limit and will fail
- **n=5 is insufficient**: May miss table data that's further away

### Example: Table Reconstruction
```
1. kb_search(query="Table 1.1")
   → Returns chunk with table caption/reference
   → chunk_id: "14f24b1b-c316-5a29-a722-1ab95cced35d"

2. kb_neighbors(chunk_id="14f24b1b...", n=10)
   → Returns 21 chunks total (10 before + reference + 10 after)
   → Complete table data found in chunks 3-10 positions away
   → Data spans chunks at positions 201837, 202757, 203720

3. Parse and reconstruct
   → Extract table rows from neighboring chunks
   → Reassemble into complete table structure
```

### Example: Understanding a Procedure
```
1. kb_search(query="startup procedure for DAF system")
   → Returns chunk with procedure title or step 3
   → chunk_id: "a1b2c3d4..."

2. kb_neighbors(chunk_id="a1b2c3d4...", n=10)
   → Returns 21 chunks
   → Chunks before: Prerequisites, steps 1-2
   → Reference chunk: Step 3
   → Chunks after: Steps 4-8, warnings, completion criteria

3. Provide complete answer
   → Full procedure from prerequisites through completion
   → No missing steps or context
```

### When to Increase Neighbor Radius
- **Large multi-page tables**: Try n=15-20 (but check token limit warnings)
- **Complex procedures**: Spanning multiple sections
- **Figure captions + figures**: When figures are on separate pages

### When Smaller Radius is Sufficient
- **n=5**: Short tables (3-5 rows), single-section content
- **n=3**: Just for immediate context (paragraph continuity)
- **n=1**: Default environment variable `NEIGHBOR_CHUNKS=1` (auto-expansion in search results)

### Token Limit Management
- **MCP response limit**: 25,000 tokens
- **n=10**: ~19,000-20,000 tokens (safe)
- **n=15**: ~29,000 tokens (exceeds limit, will fail)
- **If you hit token limits**: Reduce n value and make multiple targeted neighbor calls

### Best Practice Summary
✅ **ALWAYS expand top search results with kb_neighbors(n=10)** - treat this as mandatory, not optional
✅ **Single chunks are insufficient** - 700-char chunks distribute context across neighbors
✅ Use the reference chunk's `chunk_id` from search results as the anchor
✅ Expect critical information to be 3-10 chunks away from the highest-scoring chunk
✅ Parse neighbor results by chunk_start position (ascending order) for coherent context
⚠️ **Never answer from a single chunk alone** - you will miss critical context
⚠️ Don't assume the top search result contains complete information
⚠️ Don't skip neighbor expansion - it's required for comprehensive answers

## Tool Reference
### Ingestion Tools (MCP)
- `ingest.extract_with_strategy`, `ingest.validate_extraction`
- `ingest.chunk_with_guidance`, `ingest.generate_metadata`, `ingest.generate_summary`
- `ingest.assess_quality`, `ingest.enhance`
- `ingest.upsert`, `ingest.upsert_batch`

### Retrieval Tools (MCP)
- Search routes: `kb.search`, `kb.hybrid`, `kb.dense`, `kb.sparse`, `kb.sparse_splade`, `kb.rerank`, `kb.colbert`, `kb.batch`
- Evidence: `kb.open`, `kb.neighbors`, `kb.table`, `kb.summary`, `kb.outline`
- Graph: `kb.entities`, `kb.linkouts`, `kb.graph`
- Quality & guidance: `kb.quality`, `kb.hint`
- Session controls: `kb.promote`, `kb.demote`, `kb.collections`

## Provenance & Reporting
- Document all decisions in `client_orchestration` / `client_decisions` before upserting
- Chunk artifacts carry `plan_hash`, `model_version`, `prompt_sha` automatically
- Server persists provenance to vector/FTS payloads on upsert
- Do not bypass budgets, rewrite chunk text, or invent missing metadata
- If unclear, escalate rather than guessing

## Client-Side HyDE Loop
1. Run initial search (`kb.search`/`kb.hybrid`) and inspect `best_score` plus telemetry
2. If quality gates fail or scores below threshold, compose hypothetical passage leveraging domain knowledge
3. Re-issue retrieval with `kb.dense(query=hypothesis, retrieve_k=..., return_k=...)` and capture before/after scores in `client_decisions`
4. If hypothesis improves recall while meeting answerability gates, proceed; otherwise revert or abstain

## Hierarchical Summaries
1. After chunking, partition chunks by `section_path` (deepest level first)
2. Summarise each leaf section in 3–5 sentences, citing `element_ids` used
3. Aggregate child summaries upward—create parent-section synopses referencing child `element_ids`
4. Persist each level via `ingest.generate_summary`, attaching `model`, `prompt_sha`, and roll-up provenance in `client_decisions`
5. Re-run retrieval quality checks to ensure summaries improve downstream `kb.summary` answers without hallucination
