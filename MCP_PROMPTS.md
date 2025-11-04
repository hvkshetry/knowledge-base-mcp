# MCP Prompt Snippets

Quick snippets you can drop into Claude Desktop/Code (or any MCP client) to make
use of the retrieval and ingestion toolchain.

## Retrieve → Assess → Refine

```
You are connected to the knowledge-base MCP server. Call `kb.collections` if you need to confirm available slugs before issuing retrieval calls.
For each user question:
1. Call `kb.hybrid(collection="daf_kb", retrieve_k=32, return_k=10)`.
2. Inspect the top hits with `kb.open(collection="daf_kb", chunk_id=...)` and run `kb.quality(collection="daf_kb", query="<user question>", rules={ "min_score": 0.35, "require_metadata": true })`.
3. If quality fails, compose your own HyDE hypothesis and rerun with `kb.dense` / `kb.rerank` (there is no server-side HyDE tool).
4. Include element_ids + page numbers in the final answer; abstain if
   quality never passes.
```

## Table Lookup

```
Use `kb.table(collection="daf_kb", query="design MLSS")` to fetch table rows about MLSS design values.
Then call `kb.open(collection="daf_kb", chunk_id=<id>)` to quote the row verbatim.
```

## Graph Walk

```
1. `kb.entities(collection="daf_kb", types=["equipment","parameter"], match="saturator")`.
2. For each entity, call `kb.linkouts(entity_id=<id>, limit=5)`.
3. Use `kb.open(collection="daf_kb", chunk_id=<id>)` to collect the cited spans and narrate the relationship.
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

## Upsert & Summaries

```
- `ingest.generate_summary` doc_id=<uuid> collection="daf_kb" section_path=["Chapter 2","Equipment"] summary_text="Dissolved air flotation relies on recycled flow..." summary_metadata={"model":"claude-sonnet-4","prompt_sha":"sha256:...","temperature":0.0}
- `ingest.upsert` doc_id=<uuid> collection="daf_kb" chunks_artifact=@chunks.json thin_payload=true update_graph=true update_summary=true
- `ingest.upsert_batch` upserts=[{"doc_id": <uuid>, "chunks_artifact": @chunks.json}] parallel=4
- `ingest.corpus_upsert` root_path="./daf_kb" collection="daf_kb" dry_run=true extractor="auto" chunk_profile="auto"
```

## HyDE Generation (Client-Side)

```
results = await kb.hybrid(collection="daf_kb", query=user_query, retrieve_k=32, return_k=12)
if max((row.get("score") or 0.0) for row in results if isinstance(row, dict)) < 0.35:
    hypothesis = """
    [Generate a grounded hypothetical answer summarising the expected content.]
    """
    hyde_results = await kb.dense(collection="daf_kb", query=hypothesis, retrieve_k=24, return_k=12)
```

Include these fragments in your system prompt or invocation instructions to make
the most of the MCP tool surface.
