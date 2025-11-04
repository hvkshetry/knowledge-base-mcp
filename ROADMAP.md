# Roadmap

This roadmap captures the remaining improvements we plan to tackle after the current release. Work is ordered to satisfy the non-negotiable objectives (local-first, deterministic, provenance-rich, MCP-orchestrable) before higher-variance experiments.

## Stage 0 – Instrumentation & Provenance *(completed)*
- Extend `scripts/backfill_fts_from_qdrant.py` to replay page numbers, section paths, element/table metadata, and ship a lightweight migrator for adding missing UNINDEXED columns. *(done)*
- Attach per-hit `scores={bm25,dense,rrf,rerank,decay,final}` plus a `why` field (matched terms, header/table hints) in `server.py`, and persist the same payload in structured logs. *(done)*
- Normalize thin-payload behaviour under a documented `THIN_PAYLOAD` env flag so ingest, search, and backfills stay in sync; add regression coverage. *(done)*
- Document acceptance checks (FTS rebuild retains provenance, thin mode round-trips, score vectors readable) and bake them into smoke scripts. *(done – see “Operational Acceptance Checks” in `USAGE.md`)*

## Stage 1 – Deterministic Ingestion Plans *(completed)*
- Persist per-doc ingest plans with triage routes, chunker profiles, plan hashes, and doc metadata; chunks now carry `plan_hash`/`chunk_profile` across vector + FTS layers. *(done)*
- Enumerated chunkers (`fixed_window`, `heading_based`, `procedure_block`, `table_row`) are auto-selected per document structure and exposed for downstream filtering. *(done)*
- Table rows remain first-class via `type="table_row"`, `table_headers`, `table_units`, and provenance fields in both stores, enabling targeted retrieval. *(done)*
- Bounded metadata schema (summary <=320 chars, <=8 concepts, typed entity buckets, units) validated via pydantic with reject logging; metadata hashes feed plan persistence. *(done)*
- Document Stage‑1 acceptance checks (plan replay smoke test, table-row search example, metadata validation notes). *(done – usage guide now documents the acceptance flow)*

## Stage 2 – MCP-Orchestrated Ingestion
- Expose MCP tools `ingest.analyze_document`, `ingest.extract_with_strategy`, `ingest.chunk_with_guidance`, `ingest.generate_metadata`, `ingest.assess_quality`, and `ingest.enhance`, each returning deterministic artifacts. *(done – endpoints now emit plans/artifacts/metadata summaries)*
- Enforce budget guardrails (`MAX_METADATA_CALLS_PER_DOC`, `MAX_METADATA_BYTES`) and embed `plan_hash`, `model_version`, and `prompt_sha` in artifacts/payloads for auditability. *(done)*
- Surface artifact references and plan hashes back into ingestion payloads so downstream MCP clients can chain steps without guessing filenames. *(done – artifacts stored under `data/ingest_artifacts/<doc_id>/`)*
- Document the ingestion toolchain in `USAGE.md` and provide a minimal MCP client walkthrough covering analyze → extract → chunk → ingest. *(done – see new "MCP Ingestion Workflow" section)*

## Stage 3 – Retrieval Primitives & Planner Telemetry
- Add atomic MCP tools `kb.sparse`, `kb.dense`, `kb.hybrid`, `kb.rerank`, `kb.open`, `kb.neighbors`, `kb.batch`, and `kb.quality` (with `collection=` parameter) to give clients surgical control. *(done)*
- Return explicit score components plus `why` annotations on every hit, and include them in search telemetry so planners can reason about route quality. *(done)*
- Replace `_needs_sparse_retry` heuristics with a bounded critic (rule-based or distilled) that decides on the single HyDE retry; log planner decisions and outcomes. *(done – `_should_retry_sparse` critic + telemetry)*
- Make route-selection weights (`MIX_W_*`) fully env-driven with validation and doc updates, ensuring identical inputs yield identical final scores. *(done)*
- Expose helper tools: `kb.hint` for alias/synonym expansions. HyDE retries are now fully client-authored (no server tool). *(done)*
- Provide `kb.table_lookup` for table/cell retrieval using stored `table_headers`/`table_units` metadata. *(done)*
- Offer `kb.outline` for document TOCs and `kb.promote`/`kb.demote` APIs to adjust per-session priors on doc_ids. *(done)*
- Add optional scope/weight controls to search tools and ensure all tools can return `{ "abstain": true, ... }` with reasons so planners can handle insufficent evidence gracefully. *(done)*

## Stage 4 – Security & Governance Hardening
- Ensure thin payload mode strips full text everywhere (ingest, backfill, `_ensure_row_texts`) while still allowing ACL-checked dereference via `open_*`. *(done)*
- Implement policy-aware dereference connectors (filesystem, SharePoint/Drive placeholders) and return structured deny reasons when ACLs block access. *(done – `ALLOWED_DOC_ROOTS` gating + explicit forbidden responses)*
- Emit hashed JSONL audit records for every search/open/neighbors call, capturing subject hashes, query hashes, doc/element IDs, decision, and latency. *(done – `AUDIT_LOG_PATH`)*
- Refresh governance docs covering ACL config, audit log rotation, and thin-index deployment guidance. *(done – see USAGE.md security section)*

## Stage 5 – Evaluation & CI Rails *(in progress)*
- [x] Remove server-side HyDE retries so evaluation reflects the client-led planner; document the workflow in `eval/gold_sets/README.md`.
- [x] Generate 200 deterministic excerpt-matching queries per collection via `scripts/generate_goldset.py` (e.g., `eval/gold_sets/daf_kb_auto.jsonl`) to seed the expanded gold sets.
- [ ] Curate the generated seeds into 150–300 hand-verified queries per collection spanning narrative/table/multi-hop scenarios, with maintained relevance judgments.
- [ ] Enhance `eval.py` to emit Recall@50, nDCG@10, MRR@10 plus per-stage latency metrics (embed, sparse, hybrid, rerank) and structured outputs for dashboards.
- [ ] Add CI thresholds (`EVAL_MIN_NDCG10`, `EVAL_MIN_RECALL50`, `EVAL_MAX_P90_MS`) so quality or latency regressions fail builds.
- [ ] Publish lightweight reports (CSV/JSON → Grafana/static HTML) to visualise quality and latency deltas per release.

## Stage 6 – Advanced Enhancements *(active)*
- SPLADE/uniCOIL sparse expansion: hooks (`--sparse-expander`, `kb.sparse_splade`) are present, but no SPLADE model is bundled yet.
- MCP playbooks/prompts documented in `MCP_PLAYBOOKS.md` and `MCP_PROMPTS.md`, encouraging the agent to act as critic/self-rerouter.
- `graph_builder.py` currently attaches entity → chunk links; richer relations (`feeds`, `discharges_to`, `located_in`) remain on the backlog.
- `scripts/manage_cache.py` plus new env knobs (`DOCLING_DEVICE`, `DOCLING_BATCH_SIZE`) simplify Docling GPU/caching operations (GPU use remains optional).
- MCP UX surfaces (playbooks + prompt snippets) highlight graph/summary/neighbor actions for richer planning. (Feature maturity mirrors the notes above.)
- Canary-driven quality checks run during `ingest.assess_quality()` when `config/canaries/*.json` is populated (defaults are placeholders).
- `ingest.enhance` implements safe ops (`add_synonyms`, `link_crossrefs`, `fix_table_pages`) for incremental fixes without full re-ingest.

## Known Schema & Tooling Follow-Ups
- Normalize stored arrays/objects: `element_ids`, `bboxes`, `types`, `source_tools`, `table_headers`, `table_units`, and `doc_metadata` are persisted as Python repr strings in FTS/Qdrant payloads; migrate them to proper JSON structures and backfill existing rows.
- Table metadata: ensure table headers/units survive extraction before re-enabling the stronger table lookup guarantees (current MCP `kb.table` often returns empty hits).
- Summary/outline/hint builders: ship the background jobs that populate `summary.db`, outline indices, and richer alias expansions so `kb.summary`, `kb.outline`, and `kb.hint` return real data instead of placeholders.
- Entity provenance: extend the graph pipeline to retain `doc_id` on entity nodes/edges so `kb.entities` can always surface source documents.
- Evaluation telemetry: integrate `eval.py` outputs with CI and dashboards (see Stage 5 tasks above) and automate ingestion of new judgments.

Contributions are welcome—see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
