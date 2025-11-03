# Sparse Expansion & ColBERT Integration Guide

This note captures the steps required to add higher-recall sparse expansions (SPLADE/uniCOIL) and late-interaction reranking (ColBERT). The wiring described below is **not yet active in the default configuration**—it documents the plan for enabling these capabilities. The ingestion pipeline now accepts `--sparse-expander splade` (or `basic`) and exposes `kb.sparse_splade(collection=...)`; the ColBERT hook is wired via `COLBERT_URL` + `kb.colbert(collection=...)`, with the remaining work focused on deploying a production-grade ColBERT service and tuning fusion.

## 1. SPLADE / uniCOIL Sparse Expansion

1. **Model selection**
   - Recommended: `naver/splade-cocondenser-ensembledistil` (PyTorch) or `castorini/uniCOIL-msmarco-passage`.
   - Run inference offline during ingestion; cache expanded token weights alongside existing BM25 payloads.

2. **Ingestion changes**
   - Extend `ingest.py` to optionally call `--sparse-expander splade` (default: disabled).
   - For each chunk, persist the sparse term-weight dictionary in Qdrant payload under `sparse_terms`.
   - Update `lexical_index.py` to add a virtual table `fts_chunks_sparse(term TEXT PRIMARY KEY, weight REAL, chunk_id TEXT)` for fast sparse-only queries when SQLite FTS is insufficient.

3. **Query-time flow**
   - Add `_run_splade_sparse` in `server.py` that:
     1. Generates the SPLADE vector for the query (local batch model).
     2. Computes top-k via dot product against stored term-weight dictionaries.
     3. Returns rows shaped like other sparse queries so `kb.sparse(collection=...)` and `kb.batch(collection=...)` can route to it.
   - Update `plan_route` heuristics to favour SPLADE for acronym-heavy or very short queries.

4. **MCP exposure**
   - Add `kb.sparse_splade(collection=...)` MCP support; keep the existing `kb.sparse(collection=...)` as an alias that chooses between BM25 and SPLADE based on query length or a client-provided flag.
   - Document the call pattern in `MCP_PLAYBOOKS.md` so agents can explicitly switch tracks when needed.

## 2. ColBERT Late Interaction

1. **Model selection**
   - Recommended: `colbert-ir/colbertv2.0` (requires faiss for ANN search).
   - Build the ColBERT index offline: store document embeddings under `data/colbert/{collection}/`.

2. **Runtime service**
   - Start a ColBERT inference container or local service that can:
     - Embed queries on demand.
     - Execute ANN search against the pre-built index.
   - Record endpoint URL via `COLBERT_URL` env var.

3. **Server integration** *(planned)*
   - `_run_colbert` posts to `{COLBERT_URL}/query` and hydrates chunk payloads via the FTS index.
   - Planner will auto-select `colbert` when questions contain `?` or longer phrasings; agents can call `kb.colbert(collection=...)` directly.
   - Scores appear under `scores.dense/colbert_score`; quality gates can reason about them alongside other components.

4. **MCP tooling** *(planned)*
   - `kb.colbert(collection=...)` is available when `COLBERT_URL` is set, and `kb.batch(collection=..., routes=[...])` accepts `route="colbert"`.
   - `kb.quality(collection=...)` already consumes `scores.final`, so ColBERT confidence flows through existing checks.

## 3. Orchestrating with MCP Clients

The current tool surface already supports agentic control. Once SPLADE/ColBERT are wired (future work):

```json
{"tool": "kb.batch", "args": {
  "collection": "daf_kb",
  "queries": ["DAF MLSS target", "design MLSS Table"],
  "routes": ["hybrid", "colbert"]
}}
```

Agents will be able to:

- Fall back to SPLADE when BM25 misses domain synonyms (`kb.sparse_splade(collection=...)`).
- Run ColBERT alongside hybrid retrieval and compare score vectors via `kb.quality(collection=...)`.
- Promote/demote documents based on relative ColBERT scores without modifying the server.

## 4. Next Steps (Open Work)

1. Prototype SPLADE inference inside the ingestion pipeline (ensure it respects `plan_hash` determinism).
2. Stand up a ColBERT service and build the first collection index (e.g., `daf_kb`).
3. Add planner heuristics + MCP documentation updates.
4. Collect evaluation deltas using `eval.py` and new gold-set slices focusing on acronym-heavy or multi-hop queries.

With these building blocks in place, the MCP client will be able to choose the best retrieval modality per query while the server remains deterministic and auditable.
