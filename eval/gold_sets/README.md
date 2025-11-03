# Gold Sets

This folder stores evaluation gold files used by `eval.py`.

## Format

Each line is a JSON object:

```
{
  "query": "...",
  "collection": "daf_kb",
  "relevance": [
    {"doc_id": "...", "gain": 1.0},
    {"path_contains": "...", "gain": 0.5}
  ]
}
```

Optional fields:
- `slug`: resolves collection from `NOMIC_KB_SCOPES`
- `subjects`: list of subjects to impersonate during eval
- `table_id`, `table_row_index` for table-specific assertions.
- `notes`: free-form comments (ignored by the evaluator)

> **HyDE / hypothesis retries:** the evaluator no longer calls any LLMs. If you want HyDE-style reruns during evaluation, precompute the hypothetical passages in your driver script (e.g., Claude orchestration) and pass them through `kb.dense` before invoking `eval.py`. When a query falls below the answerability threshold the script records an abstain without retrying automatically.

## Building a Larger Gold Set

1. Run ingestion so chunk metadata (plan_hash, table headers) is populated.
2. Use the MCP tools (`open_*`, `table_*`, `outline_*`) to find authoritative spans.
3. Record each query with at least one ground-truth relevance judgment; use `gain` to weight critical evidence.
4. Maintain ~150–300 queries per major collection to balance coverage. Run `scripts/generate_goldset.py` to bootstrap excerpt-matching queries straight from the FTS database (example below). Auto-generated seeds are already checked in under `eval/gold_sets/*_auto.jsonl`; curate and enrich them over time (multi-hop, table lookups, failure cases) before promoting them to a vetted set. The generator produces deterministic output given the same FTS snapshot and seed, so you can regenerate locally without leaking source content:

```bash
python3 scripts/generate_goldset.py \
  --fts data/daf_kb_fts.db \
  --collection daf_kb \
  --output eval/gold_sets/daf_kb_auto.jsonl \
  --limit 160
```

A starter template is provided in `daf_kb_template.jsonl` if you need to seed a new collection manually. Replace placeholders with actual doc IDs/path fragments as you curate the dataset.
