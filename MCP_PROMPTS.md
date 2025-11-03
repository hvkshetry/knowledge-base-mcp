# MCP Prompt Snippets

Quick snippets you can drop into Claude Desktop/Code (or any MCP client) to make
use of the retrieval and ingestion toolchain.

## Retrieve → Assess → Refine

```
You are connected to the knowledge-base MCP server.
For each user question:
1. Call `kb.hybrid_daf_kb` (retrieve_k=32, return_k=10).
2. Inspect the top hits with `kb.open_daf_kb` and run `kb.quality_daf_kb`
   with `{ "min_score": 0.35, "require_metadata": true }`.
3. If quality fails, call `kb.hyde_daf_kb` and rerun `kb.rerank_daf_kb`.
4. Include element_ids + page numbers in the final answer; abstain if
   quality never passes.
```

## Table Lookup

```
Use the `kb.table_daf_kb` tool to fetch table rows about MLSS design values.
Then call `kb.open_daf_kb` for the returned chunk_id to quote the row verbatim.
```

## Graph Walk

```
1. `kb.entities_daf_kb` types=["equipment","parameter"], match="saturator".
2. For each entity, call `kb.linkouts_daf_kb` limit=5.
3. Use `kb.open_daf_kb` to collect the cited spans and narrate the relationship.
```

## Ingestion QA

```
When handed a new PDF:
- `ingest.analyze_document` path=<file>
- `ingest.extract_with_strategy` path=<file> plan=@previous_plan
- `ingest.chunk_with_guidance` artifact_ref=@blocks.json profile=heading_based
- `ingest.assess_quality` doc_id=<uuid>
- Review the canary results and warnings; follow up with `ingest.enhance` if needed.
```

## Targeted Enhancements

```
- `ingest.enhance` doc_id=<uuid> op=add_synonyms args={"synonyms": {"MLSS": ["mixed liquor"]}}
- `ingest.enhance` doc_id=<uuid> op=link_crossrefs args={"references": [{"chunk_id": "...", "target": "Table 3-2"}]}
- `ingest.enhance` doc_id=<uuid> op=fix_table_pages args={"pages": {"tbl_row_3_2": 41}}
```

Include these fragments in your system prompt or invocation instructions to make
the most of the MCP tool surface.
