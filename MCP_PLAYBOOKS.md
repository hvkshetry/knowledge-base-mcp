# MCP Agent Playbooks

These playbooks show how an MCP-aware client (Claude Desktop/Code, Codex CLI, etc.) can orchestrate ingestion and retrieval by composing the server’s atomic tools. The goal is to keep the agent in control—planning, critiquing, and refining—while the server guarantees determinism, provenance, and safety.

## 1. Retrieve → Assess → Refine

_First list the configured collections with `kb.collections` if you need to confirm slugs before routing._

1. **Plan a route**
   - Start with `kb.hybrid(collection="<slug>")` or `kb.rerank(collection="<slug>")`.
   - For short, keyword-heavy prompts call `kb.sparse(collection="<slug>")` or `kb.sparse_splade(collection="<slug>") (requires a configured SPLADE expander)` when sparse expansion is enabled.
   - For long-form or multi-fact questions, try `kb.colbert(collection="<slug>") (requires an external ColBERT service)` (requires `COLBERT_URL`).
   - Optionally call `kb.hyde` or `kb.generate_hyde` to generate a hypothetical answer when a text-generation model is configured.

2. **Inspect evidence**
   - Call `kb.open(collection="<slug>", chunk_id=...)` for the top chunk(s) to ensure the text matches expectations.
   - Fetch nearby context with `kb.neighbors(collection="<slug>", chunk_id=...)` when the snippet feels too narrow.
   - Use `kb.entities(collection="<slug>", ...)` → `kb.linkouts(entity_id=...)` for graph-aware exploration (e.g., “show me every chunk mentioning the saturator”).

3. **Run a quality gate**
   - `kb.quality(collection="<slug>", query="<user question>", rules={"min_score": 0.4, "require_metadata": true, "require_plan_hash": true})`.
   - Review `analysis.score_summary`, `analysis.coverage_summary`, and any `analysis.warnings` (`low_query_coverage`, `duplicate_doc_hits`, etc.).
   - For table answers, verify with `kb.table(collection="<slug>", ...)` and ensure `type="table_row"` exists.

4. **Refine if needed**
   - If quality checks fail, branch:
     - Retry with a different route (`kb.sparse`, `kb.dense`, `kb.hybrid`) and the same `collection` value.
     - Increase breadth via `kb.batch(collection="<slug>", queries=[...])` (multiple phrasings at once).
     - Expand aliases with `kb.hint` and re-run sparse search.

5. **Promote or demote sources**
   - When a document is clearly authoritative, call `kb.promote(doc_id=...)` to bias future results; demote noisy sources with `kb.demote(doc_id=...)`.

## 2. Structured Answering for Tables

1. Locate the relevant table rows:
   ```json
   {"tool": "kb.table", "args": {"collection": "daf_kb", "query": "design MLSS", "limit": 5}}
   ```
2. Resolve the snippet for each row via `kb.open(collection="daf_kb", chunk_id=...)` (cite element IDs in the answer).
3. Run `kb.quality(collection="daf_kb", query="<user question>", rules={"require_metadata": true, "require_plan_hash": true, "require_table_hit": true})` so the critic confirms at least one table row was retrieved.
4. Summarise or compare rows; include page numbers/element IDs in the final response.

## 3. Ingestion QA Loop

1. `ingest.analyze_document` → review per-page routing decisions.
2. `ingest.extract_with_strategy` (reuse the stored plan hash, or override routes).
3. `ingest.validate_extraction(artifact_ref=@blocks.json, rules={"min_blocks": 20, "expect_tables": true})` – bail out or upgrade to Docling when heuristics fail.
4. `ingest.chunk_with_guidance` with explicit `profile` (`heading_based`, `procedure_block`, `table_row`).
5. `ingest.generate_metadata` (respects byte/call limits) and optionally `ingest.generate_summary` to store a semantic summary for key sections.
6. `ingest.assess_quality` to review chunk stats, metadata status, and canary query outcomes.
7. If warnings appear, call `ingest.enhance` (`add_synonyms`, `link_crossrefs`, `fix_table_pages`) or rerun earlier steps; identical inputs always produce the same `plan_hash`.
8. `ingest.upsert` (single doc) or `ingest.upsert_batch` / `ingest.corpus_upsert` once the plan is approved.

## 4. Answer Critic Pattern

To stay production-safe without training a separate critic:

1. Perform an initial retrieval (e.g., `kb.hybrid(collection="<slug>")`).
2. Call `kb.quality(collection="<slug>", query="<user question>")` to inspect score vectors (`bm25`, `dense`, `rrf`, `rerank`, `prior`, `decay`) and coverage diagnostics.
3. Use a meta-prompt such as:
   > “Given the evidence and score breakdown, is there enough support to answer? If not, suggest which retrieval tool should run next.”
4. Based on the LLM’s decision, branch to `kb.hyde`, `kb.sparse`, `kb.batch`, or abstain.
5. Log the decision with the final answer for auditing.

## 5. Multi-Hop Reasoning

1. Identify entities via `kb.entities(collection="<slug>", types=[...])`.
2. For a chosen entity, call `kb.linkouts(entity_id=...)` to fetch the supporting chunks. If you need numeric values, include `"measurement"` in the type filter to surface `parameter → has_measurement` edges.
3. Combine with `kb.summary(collection="<slug>", topic=...)` for high-level context when summaries have been generated.
4. If the trail leads to a new concept, repeat steps 1–3; use `kb.graph(node_id=...)` when you need raw neighbor inspection.

## Quick Reference (⚙️ = requires extra configuration)

| Goal | Recommended Tools |
|------|-------------------|
| High-confidence Q/A | `kb.hybrid` → `kb.quality` → `kb.open` | (falls back to `kb.rerank` if hybrid unavailable)
| Table lookup | `kb.table` → `kb.open` → `kb.quality` |
| Alias expansion | `kb.hint` → `kb.sparse` |
| Graph exploration | `kb.entities` → `kb.linkouts` → `kb.graph` | (no semantic relation inference yet)
| Metadata validation | `ingest.generate_metadata` → `ingest.assess_quality` | (add canary queries for enforcement)
| Document routing tweak | `ingest.analyze_document` → `ingest.extract_with_strategy` |

Use these playbooks as prompts or call sequences inside your MCP client. Because every tool is deterministic, you can safely chain them, log decisions, and replay workflows across environments.
