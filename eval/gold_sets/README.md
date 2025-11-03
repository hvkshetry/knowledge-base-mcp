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

## Building a Larger Gold Set

1. Run ingestion so chunk metadata (plan_hash, table headers) is populated.
2. Use the MCP tools (`open_*`, `table_*`, `outline_*`) to find authoritative spans.
3. Record each query with at least one ground-truth relevance judgment; use `gain` to weight critical evidence.
4. Maintain ~150–300 queries per major collection to balance coverage. Each `*.jsonl` file in this folder now contains 160 excerpt-matching queries (e.g. `daf_kb.jsonl`, `clarifier_kb.jsonl`, `aerobic_treatment_kb.jsonl`, etc.) which can be used as-is or extended.

A starter template is provided in `daf_kb_template.jsonl` if you need to seed a new collection. Replace placeholders with actual doc IDs/path fragments as you curate the dataset.
