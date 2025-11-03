# MCP Agent Playbooks

These playbooks show how an MCP-aware client (Claude Desktop/Code, Codex CLI, etc.) can orchestrate ingestion and retrieval by composing the server’s atomic tools. The goal is to keep the agent in control—planning, critiquing, and refining—while the server guarantees determinism, provenance, and safety.

## 1. Retrieve → Assess → Refine

1. **Plan a route**
   - Start with `kb.hybrid_{slug}` or `kb.rerank_{slug}`.
   - For short, keyword-heavy prompts call `kb.sparse_{slug}`.
   - Use `kb.hyde_{slug}` to generate a hypothetical answer when confidence is low.

2. **Inspect evidence**
   - Call `kb.open_{slug}` for the top chunk(s) to ensure the text matches expectations.
   - Fetch nearby context with `kb.neighbors_{slug}` when the snippet feels too narrow.
   - Use `kb.entities_{slug}` → `kb.linkouts_{slug}` for graph-aware exploration (e.g., “show me every chunk mentioning the saturator”).

3. **Run a quality gate**
   - `kb.quality_{slug}` with rules such as `{"min_score": 0.4, "require_metadata": true, "require_plan_hash": true}`.
   - For table answers, verify with `kb.table_{slug}` and ensure `type="table_row"` exists.

4. **Refine if needed**
   - If quality checks fail, branch:
     - Retry with a different route (`kb.sparse`, `kb.dense`, `kb.hybrid`).
     - Increase breadth via `kb.batch_{slug}` (multiple phrasings at once).
     - Expand aliases with `kb.hint_{slug}` and re-run sparse search.

5. **Promote or demote sources**
   - When a document is clearly authoritative, call `kb.promote_{slug}` to bias future results; demote noisy sources with `kb.demote_{slug}`.

## 2. Structured Answering for Tables

1. Locate the relevant table rows:
   ```json
   {"tool": "table_daf_kb", "args": {"query": "design MLSS", "limit": 5}}
   ```
2. Resolve the snippet for each row via `kb.open_{slug}` (cite element IDs in the answer).
3. Run `kb.quality_{slug}` with `{"require_metadata": true, "require_plan_hash": true}`.
4. Summarise or compare rows; include page numbers/element IDs in the final response.

## 3. Ingestion QA Loop

1. `ingest.analyze_document` → review per-page routing decisions.
2. `ingest.extract_with_strategy` (reuse the stored plan hash, or override routes).
3. `ingest.chunk_with_guidance` with explicit `profile` (`heading_based`, `procedure_block`, `table_row`).
4. `ingest.generate_metadata` (respects byte/call limits).
5. `ingest.assess_quality` to ensure chunk counts, table coverage, and metadata status look healthy.
6. If adjustments are needed, re-run the sequence; identical inputs always produce the same `plan_hash`.

## 4. Answer Critic Pattern

To stay production-safe without training a separate critic:

1. Perform an initial retrieval (e.g., `kb.hybrid`).
2. Call `kb.quality_{slug}` and inspect score vectors (`bm25`, `dense`, `rrf`, `rerank`, `prior`, `decay`).
3. Use a meta-prompt such as:
   > “Given the evidence and score breakdown, is there enough support to answer? If not, suggest which retrieval tool should run next.”
4. Based on the LLM’s decision, branch to `kb.hyde`, `kb.sparse`, `kb.batch`, or abstain.
5. Log the decision with the final answer for auditing.

## 5. Multi-Hop Reasoning

1. Identify entities via `kb.entities_{slug}` (filter by type: `"equipment"`, `"parameter"`, `"chemical"`, etc.).
2. For a chosen entity, call `kb.linkouts_{slug}` to fetch the supporting chunks. If you need numeric values, include `"measurement"` in the type filter to surface `parameter → has_measurement` edges.
3. Combine with `kb.summary_{slug}` for high-level context.
4. If the trail leads to a new concept, repeat steps 1–3; use `kb.graph_{slug}` when you need raw neighbor inspection.

## Quick Reference

| Goal | Recommended Tools |
|------|-------------------|
| High-confidence Q/A | `kb.hybrid` → `kb.quality` → `kb.open` |
| Table lookup | `kb.table` → `kb.open` → `kb.quality` |
| Alias expansion | `kb.hint` → `kb.sparse` |
| Graph exploration | `kb.entities` → `kb.linkouts` → `kb.graph` |
| Metadata validation | `ingest.generate_metadata` → `ingest.assess_quality` |
| Document routing tweak | `ingest.analyze_document` → `ingest.extract_with_strategy` |

Use these playbooks as prompts or call sequences inside your MCP client. Because every tool is deterministic, you can safely chain them, log decisions, and replay workflows across environments.
