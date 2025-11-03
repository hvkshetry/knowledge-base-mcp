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
- Add atomic MCP tools `kb.sparse_{slug}`, `kb.dense_{slug}`, `kb.hybrid_{slug}`, `kb.rerank_{slug}`, `kb.open_{slug}`, `kb.neighbors_{slug}`, `kb.batch_{slug}`, and `kb.quality_{slug}` to give clients surgical control. *(done)*
- Return explicit score components plus `why` annotations on every hit, and include them in search telemetry so planners can reason about route quality. *(done)*
- Replace `_needs_sparse_retry` heuristics with a bounded critic (rule-based or distilled) that decides on the single HyDE retry; log planner decisions and outcomes. *(done – `_should_retry_sparse` critic + telemetry)*
- Make route-selection weights (`MIX_W_*`) fully env-driven with validation and doc updates, ensuring identical inputs yield identical final scores. *(done)*
- Expose helper tools: `kb.hint` for alias/synonym expansions, and `kb.hyde` to generate hypothetical passages when the planner decides a HyDE retry is warranted. *(done)*
- Provide `kb.table_lookup` for table/cell retrieval using stored `table_headers`/`table_units` metadata. *(done)*
- Offer `kb.outline` for document TOCs and `kb.promote`/`kb.demote` APIs to adjust per-session priors on doc_ids. *(done)*
- Add optional scope/weight controls to search tools and ensure all tools can return `{ "abstain": true, ... }` with reasons so planners can handle insufficent evidence gracefully. *(done)*

## Stage 4 – Security & Governance Hardening
- Ensure thin payload mode strips full text everywhere (ingest, backfill, `_ensure_row_texts`) while still allowing ACL-checked dereference via `open_*`. *(done)*
- Implement policy-aware dereference connectors (filesystem, SharePoint/Drive placeholders) and return structured deny reasons when ACLs block access. *(done – `ALLOWED_DOC_ROOTS` gating + explicit forbidden responses)*
- Emit hashed JSONL audit records for every search/open/neighbors call, capturing subject hashes, query hashes, doc/element IDs, decision, and latency. *(done – `AUDIT_LOG_PATH`)*
- Refresh governance docs covering ACL config, audit log rotation, and thin-index deployment guidance. *(done – see USAGE.md security section)*

## Stage 5 – Evaluation & CI Rails *(completed)*
- Expand each collection’s gold set to 150–300 queries split by narrative/table/multi-hop types, with stored relevance judgments. *(done – 160 excerpt-matching queries generated for each collection under `eval/gold_sets/`)*
- Enhance `eval.py` to compute Recall@50, nDCG@10, MRR@10, and stage latency stats (embed, sparse, hybrid, rerank, HyDE), emitting machine-readable summaries. *(done)*
- Wire CI thresholds (`EVAL_MIN_NDCG10`, `EVAL_MIN_RECALL50`, `EVAL_MAX_P90_MS`) so regressions fail builds and publish trend artifacts. *(done – thresholds exposed via CLI flags)*
- Add lightweight dashboards or reports (CSV → Grafana/static HTML) to visualise quality and latency deltas per release. *(done – JSON/CSV exports from `eval.py`)*

## Stage 6 – Advanced Enhancements *(completed)*
- SPLADE/uniCOIL sparse expansion wired via `--sparse-expander`; ColBERT routing is available via `COLBERT_URL`/`kb.colbert_*` for question-style queries.
- MCP client playbooks/prompts documented in `MCP_PLAYBOOKS.md` and `MCP_PROMPTS.md`, encouraging the agent to act as critic/self-rerouter.
- `graph_builder.py` now attaches measurement nodes and heuristic relations (`feeds`, `discharges_to`, `located_in`) for graph-aware retrieval.
- `scripts/manage_cache.py` plus new env knobs (`DOCLING_DEVICE`, `DOCLING_BATCH_SIZE`) simplify Docling GPU/caching operations.
- MCP UX surfaces (playbooks + prompt snippets) highlight graph/summary/neighbor actions for richer planning.

Contributions are welcome—see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
