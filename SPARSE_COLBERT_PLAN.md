# Sparse Expansion & ColBERT Integration Guide

This note captures the steps required to add higher-recall sparse expansions (SPLADE/uniCOIL) and late-interaction reranking (ColBERT) while keeping the MCP client in control of routing decisions. The ingestion pipeline now accepts `--sparse-expander splade` (or `basic`) and exposes `kb.sparse_splade_*`; the ColBERT hook is wired via `COLBERT_URL` + `kb.colbert_*`, with the remaining work focused on deploying a production-grade ColBERT service and tuning fusion.

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
     3. Returns rows shaped like other sparse queries so `kb.sparse_{slug}` and `kb.batch_{slug}` can route to it.
   - Update `plan_route` heuristics to favour SPLADE for acronym-heavy or very short queries.

4. **MCP exposure**
   - Add `kb.sparse_splade_{slug}` MCP tool; keep the existing `kb.sparse_{slug}` as an alias that chooses between BM25 and SPLADE based on query length or a client-provided flag.
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

3. **Server integration** *(implemented)*
   - `_run_colbert` posts to `{COLBERT_URL}/query` and hydrates chunk payloads via the FTS index.
   - Planner will auto-select `colbert` when questions contain `?` or longer phrasings; agents can call `kb.colbert_*` directly.
   - Scores appear under `scores.dense/colbert_score`; quality gates can reason about them alongside other components.

4. **MCP tooling** *(implemented)*
   - `kb.colbert_{slug}` is available when `COLBERT_URL` is set, and `kb.batch_{slug}` accepts `route="colbert"`.
   - `kb.quality_{slug}` already consumes `scores.final`, so ColBERT confidence flows through existing checks.

## 3. Orchestrating with MCP Clients

The current tool surface already supports agentic control. Once SPLADE/ColBERT are wired:

```json
{"tool": "kb.batch_daf_kb", "args": {
  "queries": ["DAF MLSS target", "design MLSS Table"],
  "routes": ["hybrid", "colbert"]
}}
```

Agents can:

- Fall back to SPLADE when BM25 misses domain synonyms (`kb.sparse_splade_{slug}`).
- Run ColBERT alongside hybrid retrieval and compare score vectors via `kb.quality_{slug}`.
- Promote/demote documents based on relative ColBERT scores without modifying the server.

## 4. Next Steps

1. Prototype SPLADE inference inside the ingestion pipeline (ensure it respects `plan_hash` determinism).
2. Stand up a ColBERT service and build the first collection index (e.g., `daf_kb`).
3. Add planner heuristics + MCP documentation updates.
4. Collect evaluation deltas using `eval.py` and new gold-set slices focusing on acronym-heavy or multi-hop queries.

With these building blocks, the MCP client can choose the best retrieval modality per query while the server remains deterministic and auditable.
