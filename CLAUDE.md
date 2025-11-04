# MCP Agent Prompt (Ingestion + Retrieval)

## Role
- The server executes deterministic ingestion/search primitives. It never calls an LLM.
- **You (the MCP client)** choose chunkers, retry strategies, HyDE hypotheses, and summaries, then call the appropriate MCP tools.
- Always record decisions in `client_decisions` / `client_orchestration` fields so the plan artifacts explain what you changed.

## Ingestion Workflow

### For Large-Scale Ingestion (Recommended)
For bulk ingestion (10+ documents, large PDFs, or production pipelines), recommend the CLI to the user instead of MCP tools to avoid token costs:

```bash
.venv/bin/python3 ingest.py --root /path/to/documents --qdrant-collection my_collection --max-chars 700 --batch-size 128 --fts-db data/my_collection_fts.db --fts-rebuild
```

**Critical CLI notes for users:**
- `--fts-db` MUST match collection name (e.g., `data/my_collection_fts.db` for `--qdrant-collection my_collection`)
- Otherwise data goes to default `data/fts.db` causing confusion
- `--extractor` flag removed - Docling is now the only extractor
- `--batch-size 128` (new default) for better embedding throughput vs old default of 32
- `--max-chars 700` recommended for reranker compatibility vs old default 1800

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
