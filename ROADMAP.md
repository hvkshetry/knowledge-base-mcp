# Roadmap

This roadmap captures the remaining improvements we plan to tackle after the current release. The tasks are grouped by theme and ordered roughly by priority.

## Retrieval & Ranking
- **SPLADE / uniCOIL expansion** – Train or integrate a sparse-expansion model so the sparse track goes beyond BM25 aliasing.
- **Late-interaction route (ColBERT)** – Build a ColBERT index for large technical collections and plug it into the planner for hard queries.
- **Learning-to-route** – Replace heuristic routing with a lightweight classifier or distilled LLM that predicts the best retrieval recipe per query.

## Knowledge Graph & Reasoning
- **Entity extraction upgrade** – Move from keyword-based entities to a proper NER/linking pipeline (equipment, streams, chemistry) and persist richer edge types.
- **Relation enrichment** – Add edge semantics (e.g., `feeds`, `located_in`, `controls`) so graph queries can answer multi-hop “why/how” questions.
- **Graph-aware search mode** – Experiment with blending graph walks into retrieval for explicitly multi-hop questions.

## Evaluation & Observability
- **Gold set expansion** – Grow each collection’s gold file to 150–300 queries with curated relevance judgments.
- **CI enforcement** – Add dedicated CI jobs that run `eval.py` with collection-specific thresholds and publish dashboards for trend analysis.
- **Latency budgets** – Profile each stage (embed, Qdrant, FTS, rerank, Docling) and set target P95 budgets for production deployments.

## Governance & Ops
- **Audit log exporter** – Stream the search audit payloads (hashed subject IDs, doc/element IDs) into a central logging or SIEM system.
- **Docling fleet tuning** – Investigate GPU-backed Docling workers or batched processing for large-scale ingests.
- **Cache lifecycle tooling** – Add utilities to inspect/compact `.ingest_cache/`, `graph.db`, and `summary.db` on long-running installations.

## Client Experience
- **UI/agent affordances** – Surface graph/summary tooling in downstream MCP clients (e.g., quick actions to jump to table rows or entity neighborhoods).
- **Self-critique refinement** – Extend the current sparse retry with a lightweight answer critic (LLM or rules) that can trigger targeted follow-up queries when citations look weak.

Contributions are welcome—see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
