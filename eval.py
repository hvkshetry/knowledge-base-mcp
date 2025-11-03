import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def load_gold(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _match(row: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    doc_id = expected.get("doc_id")
    if doc_id and row.get("doc_id") == doc_id:
        return True
    path_contains = expected.get("path_contains")
    if path_contains and path_contains.lower() in str(row.get("path", "")).lower():
        return True
    table_id = expected.get("table_id")
    if table_id and row.get("table_id") == table_id:
        row_index = expected.get("table_row_index")
        if row_index is None or str(row.get("table_row_index")) == str(row_index):
            return True
    return False


async def run_search(
    record: Dict[str, Any],
    mode: str,
    top_k: int,
    retrieve_k: int,
    return_k: int,
    server,
) -> Tuple[List[Dict[str, Any]], str, bool, float, Dict[str, float], bool]:
    query = record["query"]
    collection = record.get("collection")
    if not collection:
        slug = record.get("slug")
        scope = server.SCOPES.get(slug or "") if slug else None
        if scope:
            collection = scope.get("collection")
    if not collection:
        raise ValueError("Collection must be provided in record or scope")

    vec = await server.embed_query(query, normalize=True)
    subjects = record.get("subjects") or ["user:eval", "*"]
    route = mode
    route_retrieve = retrieve_k
    if mode == "auto":
        try:
            planned = await server.plan_route(query)
        except Exception:
            planned = {}
        route = str(planned.get("route") or "rerank")
        planned_k = planned.get("k")
        if isinstance(planned_k, int) and 1 <= planned_k <= 256:
            route_retrieve = planned_k
    route = route if route in {"semantic", "rerank", "hybrid"} else "rerank"
    route_retrieve = max(return_k, min(route_retrieve, 256))
    start = time.perf_counter()
    timings: Dict[str, float] = {}
    timings: Dict[str, float] = {}
    rows = await server._execute_search(
        route=route,
        collection=collection,
        query=query,
        query_vec=vec,
        retrieve_k=route_retrieve,
        return_k=return_k,
        top_k=top_k,
        subjects=subjects,
        timings=timings,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    hyde_used = False
    abstained = any(isinstance(r, dict) and r.get("abstain") for r in rows)
    best = server._best_score(rows)
    if server.ANSWERABILITY_THRESHOLD > 0.0 and best < server.ANSWERABILITY_THRESHOLD:
        hypo = await server.hyde(query)
        if hypo:
            try:
                hypo_vec = await server.embed_query(hypo, normalize=True)
            except Exception:
                hypo_vec = None
            if hypo_vec is not None:
                hyde_rows = await server._execute_search(
                    route="semantic",
                    collection=collection,
                    query=hypo,
                    query_vec=hypo_vec,
                    retrieve_k=min(route_retrieve, 16),
                    return_k=return_k,
                    top_k=top_k,
                    subjects=subjects,
                    timings=timings,
                )
                hyde_best = server._best_score(hyde_rows)
                if hyde_best >= server.ANSWERABILITY_THRESHOLD:
                    hyde_rows.insert(0, {"note": "HyDE retry satisfied threshold", "base_route": route})
                    rows = hyde_rows
                    hyde_used = True
                    abstained = any(isinstance(r, dict) and r.get("abstain") for r in rows)
    return rows, route, hyde_used, duration_ms, timings, abstained


def evaluate(record: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    expectations = record.get("relevance") or []
    clean_rows = [r for r in rows if isinstance(r, dict) and not r.get("note") and not r.get("abstain")]
    gains = [exp.get("gain", 1.0) for exp in expectations]
    idcg = 0.0
    for rank, gain in enumerate(sorted(gains, reverse=True), start=1):
        idcg += gain / math.log2(rank + 1)
    seen: set[int] = set()
    dcg = 0.0
    mrr = 0.0
    first_match_rank: Optional[int] = None
    for rank, row in enumerate(clean_rows, start=1):
        matched_idx = None
        matched_gain = 0.0
        for idx, exp in enumerate(expectations):
            if idx in seen:
                continue
            if _match(row, exp):
                matched_idx = idx
                matched_gain = exp.get("gain", 1.0)
                break
        if matched_idx is None:
            continue
        seen.add(matched_idx)
        if first_match_rank is None:
            first_match_rank = rank
        dcg += matched_gain / math.log2(rank + 1)
    if first_match_rank is not None:
        mrr = 1.0 / first_match_rank
    recall = (len(seen) / len(expectations)) if expectations else 0.0
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return {
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr,
        "matched": len(seen),
        "expected": len(expectations),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Path to JSONL gold set")
    parser.add_argument("--mode", default="auto", choices=["auto", "semantic", "rerank", "hybrid", "sparse"], help="Retrieval mode to evaluate")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--retrieve-k", type=int, default=24)
    parser.add_argument("--return-k", type=int, default=8)
    parser.add_argument("--fts-db", default=None, help="Override FTS database path before importing server")
    parser.add_argument("--min-ndcg", type=float, default=0.0, help="Fail if avg nDCG falls below this value")
    parser.add_argument("--min-recall", type=float, default=0.0, help="Fail if avg recall falls below this value")
    parser.add_argument("--max-latency", type=float, default=0.0, help="Fail if avg duration exceeds this value (ms)")
    parser.add_argument("--max-abstain-ratio", type=float, default=1.0, help="Fail if abstain ratio exceeds this value (0-1)")
    parser.add_argument("--output-json", help="Write aggregate metrics to this JSON file")
    parser.add_argument("--output-csv", help="Write per-query metrics to this CSV file")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query stdout logging")
    args = parser.parse_args()

    if args.fts_db:
        os.environ["FTS_DB_PATH"] = args.fts_db

    import importlib

    server = importlib.import_module("server")

    records = load_gold(Path(args.gold))
    totals = {
        "queries": 0,
        "recall": 0.0,
        "ndcg": 0.0,
        "mrr": 0.0,
        "duration_ms": 0.0,
        "hyde_used": 0,
        "abstain": 0,
    }
    stage_totals: Dict[str, float] = {}
    stage_counts: Dict[str, int] = {}
    per_query: List[Dict[str, Any]] = []
    for record in records:
        rows, route, hyde_used, duration_ms, timings, abstained = await run_search(
            record,
            args.mode,
            args.top_k,
            args.retrieve_k,
            args.return_k,
            server,
        )
        totals["queries"] += 1
        totals["duration_ms"] += duration_ms
        if abstained:
            totals["abstain"] += 1
        if hyde_used:
            totals["hyde_used"] += 1
        metrics = evaluate(record, rows)
        totals["recall"] += metrics["recall"]
        totals["ndcg"] += metrics["ndcg"]
        totals["mrr"] += metrics["mrr"]
        for stage, value in timings.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + value
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        per_query.append({
            "query": record["query"],
            "route": route,
            "hyde_used": hyde_used,
            "abstained": abstained,
            "duration_ms": duration_ms,
            **{f"timing_{k}": v for k, v in timings.items()},
            "recall": metrics["recall"],
            "ndcg": metrics["ndcg"],
            "mrr": metrics["mrr"],
        })
        if not args.quiet:
            print(f"Query: {record['query']}\n  Mode={args.mode} Route={route} HydeUsed={hyde_used} Duration={duration_ms:.1f}ms Abstained={abstained}")
            print(f"  Recall={metrics['recall']:.2f} nDCG={metrics['ndcg']:.2f} MRR={metrics['mrr']:.2f}")
            top_row = next((r for r in rows if isinstance(r, dict) and not r.get("note") and not r.get("abstain")), None)
            if top_row:
                snippet = (top_row.get("text") or "").replace("\n", " ")
                print(f"  Top result: {top_row.get('path')} [{top_row.get('chunk_start')},{top_row.get('chunk_end')}] -> {snippet[:160]}")
            else:
                print("  No results returned.")
            print()

    if totals["queries"]:
        q = totals["queries"]
        avg_recall = totals["recall"] / q
        avg_ndcg = totals["ndcg"] / q
        avg_mrr = totals["mrr"] / q
        avg_duration = totals["duration_ms"] / q
        abstain_ratio = totals["abstain"] / q
        print("=== Aggregate ===")
        print(f"Queries: {q}")
        print(f"Avg Recall: {avg_recall:.3f}")
        print(f"Avg nDCG@{args.return_k}: {avg_ndcg:.3f}")
        print(f"Avg MRR: {avg_mrr:.3f}")
        print(f"Avg Duration: {avg_duration:.1f} ms")
        stage_avg = {k: stage_totals[k] / stage_counts.get(k, q) for k in sorted(stage_totals)}
        for k, v in stage_avg.items():
            print(f"Avg {k}: {v:.1f} ms")
        print(f"HyDE Used: {totals['hyde_used']} / {q}")
        print(f"Abstains: {totals['abstain']} / {q} ({abstain_ratio:.2%})")
        failure = False
        if args.min_ndcg and avg_ndcg < args.min_ndcg:
            print(f"FAIL: avg nDCG {avg_ndcg:.3f} < threshold {args.min_ndcg}", file=sys.stderr)
            failure = True
        if args.min_recall and avg_recall < args.min_recall:
            print(f"FAIL: avg recall {avg_recall:.3f} < threshold {args.min_recall}", file=sys.stderr)
            failure = True
        if args.max_latency and avg_duration > args.max_latency:
            print(f"FAIL: avg duration {avg_duration:.1f} ms > threshold {args.max_latency}", file=sys.stderr)
            failure = True
        if args.max_abstain_ratio < 1.0 and abstain_ratio > args.max_abstain_ratio:
            print(f"FAIL: abstain ratio {abstain_ratio:.3f} > threshold {args.max_abstain_ratio}", file=sys.stderr)
            failure = True

        summary = {
            "queries": q,
            "avg_recall": avg_recall,
            "avg_ndcg": avg_ndcg,
            "avg_mrr": avg_mrr,
            "avg_duration_ms": avg_duration,
            "hydefreq": totals["hyde_used"],
            "abstain_count": totals["abstain"],
            "abstain_ratio": abstain_ratio,
            "stage_avg_ms": stage_avg,
        }

        if args.output_json:
            payload = {
                "summary": summary,
                "details": per_query,
            }
            with open(args.output_json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)

        if args.output_csv:
            stage_headers = sorted({k for entry in per_query for k in entry if k.startswith("timing_")})
            fieldnames = ["query", "route", "hyde_used", "abstained", "duration_ms", "recall", "ndcg", "mrr"] + stage_headers
            with open(args.output_csv, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for entry in per_query:
                    writer.writerow({k: entry.get(k, "") for k in fieldnames})

        if failure:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
