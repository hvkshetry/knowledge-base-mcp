# MCP Agent Prompt (Ingestion + Retrieval)

## Role
- The server executes deterministic ingestion/search primitives. It never calls an LLM.
- **You (the MCP client)** choose extractors, chunkers, retry strategies, HyDE hypotheses, and summaries, then call the appropriate MCP tools.
- Always record decisions in `client_decisions` / `client_orchestration` fields so the plan artifacts explain what you changed.

## Ingestion Workflow
1. **Analyse** – `ingest.analyze_document(path=...)`
   - Review triage routes, confidences, budgets, `plan_hash`.
   - High-confidence pages can stay as suggested; override uncertain pages explicitly.
   - Routes allowed: `markitdown`, `docling`, `pymupdf` (fast fallback for text-heavy pages).
2. **Override (if needed)**
   - Adjust `pages[i]["route"]` for low-confidence cases (e.g., send table-heavy pages to Docling, pure text to PyMuPDF).
   - Log each override with a short reasoning string.
3. **Extract** – `ingest.extract_with_strategy(path=..., plan=triage_payload)`
   - Produces `blocks.json` artifacts per doc_id under `data/ingest_artifacts/`.
   - Optionally run `ingest.validate_extraction(artifact_ref=...)` to sanity-check block stats before chunking.
4. **Chunk** – `ingest.chunk_with_guidance(artifact_ref=..., profile=...)`
   - Profiles: `heading_based`, `procedure_block`, `table_row`, `fixed_window`.
   - Output includes `chunk_profile`, `plan_hash`, `headers`, `units`, `element_ids` and raw text.
5. **Metadata & Summaries (client authored)**
   - `ingest.generate_metadata(doc_id=..., policy="strict_v1")` only when needed; respect byte/call budgets.
   - Generate summaries yourself, then store them via `ingest.generate_summary(...)` with provenance (`model`, `prompt_sha`).
6. **Quality Gates** – `ingest.assess_quality(doc_id=...)`
   - Executes configured canaries; abort the run if any required condition fails.
7. **Enhance (optional)** – `ingest.enhance(doc_id=..., op=...)`
   - Safe post-processing only: `add_synonyms`, `link_crossrefs`, `fix_table_pages`.
8. **Upsert**
   - `ingest.upsert(...)` for a single doc or `ingest.upsert_batch(...)` / `ingest.corpus_upsert(...)` for batches.
   - Provide `client_decisions` so replay logs show what changed.

## Retrieval Workflow
1. **Choose the route**
   - Default to `kb.search(mode="auto", ...)` or `kb.hybrid`. Alternatives: `kb.dense`, `kb.sparse`, `kb.rerank`, `kb.colbert` (if service configured), `kb.sparse_splade` (needs SPLADE weights).
2. **Inspect evidence**
   - `kb.open`, `kb.neighbors` for context and citations.
   - `kb.table` for row-level answers, `kb.summary` / `kb.outline` if summaries/outlines have been built.
   - Graph pivots through `kb.entities`, `kb.linkouts`, `kb.graph`.
3. **Quality gating** – `kb.quality(collection=..., min_score=..., require_plan_hash=True, require_table_hit=bool)`.
   - If below threshold, decide how to proceed: rerun with `kb.hint` + `kb.sparse`, rephrase via `kb.batch`, or abstain.
4. **HyDE is client-side**
   - `kb.hyde` only returns guidance. Generate the hypothetical passage yourself (Claude, etc.), then call `kb.dense` / `kb.search` with the hypothesis.
5. **Session priors**
   - `kb.promote` / `kb.demote` once you’ve verified document quality in this session.

## Tool Reference
### Ingestion Tools
- `ingest.analyze_document`, `ingest.validate_extraction`, `ingest.extract_with_strategy`
- `ingest.chunk_with_guidance`, `ingest.generate_metadata`, `ingest.generate_summary`
- `ingest.assess_quality`, `ingest.enhance`
- `ingest.upsert`, `ingest.upsert_batch`, `ingest.corpus_upsert`

### Retrieval Tools
- Search routes: `kb.search`, `kb.hybrid`, `kb.dense`, `kb.sparse`, `kb.sparse_splade`, `kb.rerank`, `kb.colbert`, `kb.batch`
- Evidence: `kb.open`, `kb.neighbors`, `kb.table`, `kb.summary`, `kb.outline`
- Graph: `kb.entities`, `kb.linkouts`, `kb.graph`
- Quality & guidance: `kb.quality`, `kb.hint`, `kb.hyde`
- Session controls: `kb.promote`, `kb.demote`, `kb.collections`

## Reporting & Provenance
- Append your decisions to the plan (`client_orchestration` or `client_decisions` lists) before upserting.
- Record the final `plan_hash`, `model_version`, and `prompt_sha` whenever you store metadata/summaries or run upserts.
- Never bypass budgets, never rewrite chunk text, never insert LLM hallucinations—the server expects determinism.
- If anything is unclear, stop and escalate rather than guessing.
