import json
import os
import logging
import asyncio
import time
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
import sqlite3

from document_store import DocumentStore, get_subjects_from_context
from graph_builder import neighbors as graph_neighbors
from summary_index import query_summaries


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

# ---- Env & config -----------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "snowflake-arctic-embed:xs")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", os.getenv("OLLAMA_MODEL_GENERATE", "llama3:8b"))
TEI_RERANK_URL = os.getenv("TEI_RERANK_URL", "http://localhost:8087")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
FTS_DB_PATH = os.getenv("FTS_DB_PATH", "data/fts.db")
GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "data/graph.db")
SUMMARY_DB_PATH = os.getenv("SUMMARY_DB_PATH", "data/summary.db")

# JSON like: {"kb":{"collection":"snowflake_kb","title":"Company KB"}}
# Backcompat: allow STELLA_SCOPES if NOMIC_KB_SCOPES not set
SCOPES_ENV = os.getenv("NOMIC_KB_SCOPES") or os.getenv("STELLA_SCOPES") or '{"kb":{"collection":"snowflake_kb","title":"Company KB"}}'
SCOPES: Dict[str, Dict[str, Any]] = json.loads(SCOPES_ENV)

# ---- Utilities --------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("kb-mcp")

qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
mcp = FastMCP(name="knowledge-base", version="1.0.0", instructions="Vector search with Qdrant + Ollama embeddings and TEI reranker")
doc_store = DocumentStore(FTS_DB_PATH)


def _lookup_chunk_by_element(element_id: str) -> Optional[str]:
    if not element_id:
        return None
    try:
        conn = sqlite3.connect(FTS_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id FROM fts_chunks WHERE element_ids LIKE ? LIMIT 1",
            (f'%"{element_id}"%',),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Rerank constraints to avoid payload-too-large
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "700"))
RERANK_MAX_ITEMS = int(os.getenv("RERANK_MAX_ITEMS", "16"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
# Neighbor packaging and scoring controls
NEIGHBOR_CHUNKS = int(os.getenv("NEIGHBOR_CHUNKS", "1"))
ANSWERABILITY_THRESHOLD = float(os.getenv("ANSWERABILITY_THRESHOLD", "0.0"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "0"))
DECAY_STRENGTH = float(os.getenv("DECAY_STRENGTH", "0.0"))
MIX_W_BM25 = _env_float("MIX_W_BM25", 0.2)
MIX_W_DENSE = _env_float("MIX_W_DENSE", 0.3)
MIX_W_RERANK = _env_float("MIX_W_RERANK", 0.5)


def l2norm(vec: List[float]) -> List[float]:
    import math
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


async def embed_query(text: str, normalize: bool = True) -> List[float]:
    try:
        r = await asyncio.to_thread(
            requests.post,
            f"{OLLAMA_URL}/api/embed",
            json={"model": OLLAMA_MODEL, "input": [text]},
            timeout=60,
        )
        if r.status_code == 404:
            r2 = await asyncio.to_thread(
                requests.post,
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=60,
            )
            r2.raise_for_status()
            v = r2.json().get("embedding")
        else:
            r.raise_for_status()
            v = r.json().get("embeddings", [[ ]])[0]
        return l2norm(v) if normalize else v
    except Exception:
        logger.exception("embed_query failed")
        raise


# ---- Local lexical helpers --------------------------------------------------
def _fts_search(query: str, limit: int) -> List[Dict[str, Any]]:
    if not os.path.exists(FTS_DB_PATH):
        return []
    try:
        from lexical_index import search as fts_search
        try:
            return fts_search(FTS_DB_PATH, query, limit)
        except Exception as exc:
            if "syntax error" in str(exc).lower():
                safe = re.sub(r"[\^`~!@#$%&*()+={}[\]|\\:;'<>,.?/]", " ", query).strip()
                if safe:
                    return fts_search(FTS_DB_PATH, safe, limit)
            raise
    except Exception as e:
        logger.warning("FTS search failed: %s", e)
        return []


def _fts_neighbors(doc_id: str, chunk_start: int, n: int) -> List[Dict[str, Any]]:
    if n <= 0 or not os.path.exists(FTS_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(FTS_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM fts_chunks WHERE doc_id = ? AND chunk_start < ? ORDER BY chunk_start DESC LIMIT ?",
            (doc_id, int(chunk_start), int(n)),
        )
        prev_rows = [dict(r) for r in cur.fetchall()]
        cur.execute(
            "SELECT * FROM fts_chunks WHERE doc_id = ? AND chunk_start >= ? ORDER BY chunk_start ASC LIMIT ?",
            (doc_id, int(chunk_start), int(n + 1)),
        )
        next_rows = [dict(r) for r in cur.fetchall()]
        rows = list(reversed(prev_rows)) + next_rows
        # Drop dup
        seen, out = set(), []
        for r in rows:
            key = (r.get("chunk_id"), r.get("chunk_start"))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out
    except Exception as e:
        logger.debug("fts_neighbors failed: %s", e)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _rrf_fuse(lists: List[List[Dict[str, Any]]], id_getter) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for li in lists:
        for rank, x in enumerate(li, start=1):
            xid = id_getter(x)
            if xid is None:
                continue
            if xid not in items:
                items[xid] = dict(x)
            else:
                for k, v in x.items():
                    if k not in items[xid]:
                        items[xid][k] = v
            scores[xid] = scores.get(xid, 0.0) + 1.0 / (HYBRID_RRF_K + rank)
    fused = list(items.values())
    for it in fused:
        try:
            key = it.get("chunk_id")
        except Exception:
            key = None
        it["_rrf_score"] = scores.get(str(key), 0.0)
    fused.sort(key=lambda d: d.get("_rrf_score", 0.0), reverse=True)
    return fused
def _decay_factor(mtime: Optional[int]) -> float:
    if not mtime or DECAY_HALF_LIFE_DAYS <= 0:
        return 1.0
    age_days = max(0.0, (time.time() - float(mtime)) / 86400.0)
    try:
        import math
        factor = math.pow(2.0, -age_days / DECAY_HALF_LIFE_DAYS)
    except Exception:
        factor = 1.0
    if DECAY_STRENGTH <= 0:
        return 1.0
    if DECAY_STRENGTH >= 1:
        return factor
    return (1.0 - DECAY_STRENGTH) + DECAY_STRENGTH * factor


def _parse_page_numbers(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value if isinstance(v, (int, float))]
    if isinstance(value, str):
        nums = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                nums.append(int(part))
            except ValueError:
                continue
        return nums
    if isinstance(value, (int, float)):
        return [int(value)]
    return []


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


async def _fetch_chunks_by_ids(chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    try:
        return await asyncio.to_thread(doc_store.get_records_bulk, chunk_ids)
    except Exception as exc:
        logger.debug("fetch_chunk_records failed: %s", exc)
        return {}


async def _hydrate_qdrant_hits(hits) -> None:
    missing: List[str] = []
    for h in hits or []:
        payload = getattr(h, "payload", None) or {}
        if payload.get("text"):
            continue
        chunk_id = str(getattr(h, "id", None) or payload.get("chunk_id") or "")
        if chunk_id:
            missing.append(chunk_id)
    if not missing:
        return
    fetched = await _fetch_chunks_by_ids(missing)
    for h in hits or []:
        payload = getattr(h, "payload", None) or {}
        chunk_id = str(getattr(h, "id", None) or payload.get("chunk_id") or "")
        info = fetched.get(chunk_id)
        if not info:
            continue
        payload.setdefault("text", info.get("text"))
        payload.setdefault("chunk_start", info.get("chunk_start"))
        payload.setdefault("chunk_end", info.get("chunk_end"))
        payload.setdefault("doc_id", info.get("doc_id"))
        payload.setdefault("path", info.get("path"))
        payload.setdefault("filename", info.get("filename"))
        if "page_numbers" not in payload and info.get("page_numbers") is not None:
            payload["page_numbers"] = _parse_page_numbers(info.get("page_numbers"))
        if "mtime" not in payload and info.get("mtime") is not None:
            payload["mtime"] = info.get("mtime")
        for key in ("pages", "section_path", "element_ids", "bboxes", "types", "source_tools", "table_headers", "table_units"):
            if payload.get(key) in (None, "", []):
                value = info.get(key)
                if isinstance(value, list):
                    payload[key] = value
                elif value not in (None, ""):
                    payload[key] = _coerce_json(value)
        setattr(h, "payload", payload)


async def _ensure_row_texts(
    rows: List[Dict[str, Any]],
    collection: Optional[str],
    subjects: List[str],
) -> None:
    missing: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("id") or row.get("chunk_id") or "")
        if chunk_id:
            missing.append(chunk_id)
    fetched = await _fetch_chunks_by_ids(missing)
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("id") or row.get("chunk_id") or "")
        info = fetched.get(chunk_id)
        existing_text = row.pop("text", None)
        doc_id = row.get("doc_id") or (info.get("doc_id") if info else None)
        allowed = doc_store.is_allowed(str(doc_id) if doc_id else None, collection, subjects)
        doc_store.build_row(row, info, allowed, include_text=False)
        page_numbers = row.get("page_numbers")
        if isinstance(page_numbers, str):
            row["page_numbers"] = _parse_page_numbers(page_numbers)
        if allowed:
            row.pop("forbidden", None)
            row.pop("reason", None)
            if info:
                text = info.get("text")
                if text is not None:
                    row.setdefault("text", text)
            if existing_text and "text" not in row:
                row["text"] = existing_text
        else:
            row.pop("text", None)
            row.setdefault("reason", "access_denied")
        for key in ("pages", "section_path", "element_ids", "bboxes", "types", "source_tools"):
            value = row.get(key)
            if isinstance(value, str) and value:
                try:
                    row[key] = json.loads(value)
                except Exception:
                    pass
        th = row.get("table_headers")
        if isinstance(th, str) and th:
            try:
                row["table_headers"] = json.loads(th)
            except Exception:
                pass
        tu = row.get("table_units")
        if isinstance(tu, str) and tu:
            try:
                row["table_units"] = json.loads(tu)
            except Exception:
                pass


def _extract_score(row: Dict[str, Any]) -> float:
    for key in ("final_score", "score", "rerank_score", "rrf_score", "dense_score"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _best_score(rows: List[Dict[str, Any]]) -> float:
    best = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "error" in row:
            continue
        if row.get("note"):
            continue
        if row.get("forbidden"):
            continue
        best = max(best, _extract_score(row))
    return best


def _normalize_score(value: float, stats: Optional[Tuple[float, float]]) -> float:
    if not stats:
        return 0.0
    min_val, max_val = stats
    if max_val - min_val < 1e-9:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def _combined_score(
    bm25: Optional[float],
    dense: Optional[float],
    rerank: Optional[float],
    bm_stats: Optional[Tuple[float, float]],
    dense_stats: Optional[Tuple[float, float]],
    rerank_stats: Optional[Tuple[float, float]],
    decay: float,
) -> float:
    weight_sum = max(MIX_W_BM25 + MIX_W_DENSE + MIX_W_RERANK, 1e-6)
    bm_norm = _normalize_score(bm25 or 0.0, bm_stats) if bm25 is not None else 0.0
    dense_norm = _normalize_score(dense or 0.0, dense_stats) if dense is not None else 0.0
    rerank_norm = _normalize_score(rerank or 0.0, rerank_stats) if rerank is not None else 0.0
    combined = (
        MIX_W_BM25 * bm_norm
        + MIX_W_DENSE * dense_norm
        + MIX_W_RERANK * rerank_norm
    ) / weight_sum
    return combined * decay


async def plan_route(query: str) -> Dict[str, Any]:
    """Choose a retrieval route based on cheap heuristics."""
    tokens = query.split()
    n_tokens = len(tokens)
    has_caps = any(any(c.isupper() for c in tok) for tok in tokens)
    if n_tokens <= 3 and "?" not in query:
        return {"route": "sparse", "k": 24}
    if n_tokens <= 4 and has_caps:
        return {"route": "hybrid", "k": 32}
    if "?" in query or n_tokens > 10:
        return {"route": "rerank", "k": 24}
    return {"route": "hybrid", "k": 24}


def _needs_sparse_retry(query: str, rows: List[Dict[str, Any]]) -> bool:
    if not rows:
        return True
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9]{4,}", query)]
    if not tokens:
        return False
    top = next((row for row in rows if row.get("text")), None)
    if not top:
        return True
    text = (top.get("text") or "").lower()
    hits = sum(1 for tok in tokens if tok in text)
    return (hits / len(tokens)) < 0.3


async def hyde(query: str) -> str:
    """Generate a short hypothetical answer paragraph for HyDE retrieval."""
    prompt = (
        "Write a concise factual paragraph (5-7 sentences) that could answer the question:\n"
        f"{query}\n\n"
        "Stay grounded in typical engineering knowledge; avoid speculation."
    )
    try:
        def _do_request():
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_LLM, "prompt": prompt, "options": {"temperature": 0.2}},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")

        response = await asyncio.to_thread(_do_request)
        return (response or "").strip()[:900]
    except Exception as exc:
        logger.debug("HyDE generation failed: %s", exc)
        return ""


async def _run_semantic(
    collection: str,
    query: str,
    query_vec: List[float],
    top_k: int,
    return_k: int,
    subjects: List[str],
    timings: Dict[str, float],
) -> List[Dict[str, Any]]:
    limit = max(top_k, return_k)
    try:
        start = time.perf_counter()
        hits = await asyncio.to_thread(
            qdr.search,
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        timings["qdrant_ms"] = timings.get("qdrant_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return [{"error": "qdrant_search_failed", "detail": str(exc)}]

    await _hydrate_qdrant_hits(hits)
    rows: List[Dict[str, Any]] = []
    for h in hits[:return_k]:
        pl = h.payload or {}
        row = {
            "score": getattr(h, "score", 0.0),
            "dense_score": getattr(h, "score", 0.0),
            "id": str(getattr(h, "id", "")),
            "chunk_id": str(getattr(h, "id", "")),
            "doc_id": pl.get("doc_id"),
            "path": pl.get("path"),
            "chunk_start": pl.get("chunk_start"),
            "chunk_end": pl.get("chunk_end"),
            "text": pl.get("text"),
            "pages": pl.get("page_numbers") or pl.get("pages"),
            "section_path": pl.get("section_path"),
            "element_ids": pl.get("element_ids"),
            "bboxes": pl.get("bboxes"),
            "types": pl.get("types"),
            "source_tools": pl.get("source_tools"),
            "table_headers": pl.get("table_headers"),
            "table_units": pl.get("table_units"),
        }
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    return rows


async def _run_sparse(
    collection: str,
    query: str,
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
) -> List[Dict[str, Any]]:
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    start = time.perf_counter()
    lexical_hits = await asyncio.to_thread(_fts_search, query, limit)
    timings["fts_ms"] = timings.get("fts_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    bm_values = [row.get("bm25") for row in lexical_hits if row.get("bm25") is not None]
    bm_stats = (min(bm_values), max(bm_values)) if bm_values else None
    rows: List[Dict[str, Any]] = []
    for item in lexical_hits[:return_k]:
        bm = item.get("bm25")
        score = _combined_score(
            bm25=bm,
            dense=None,
            rerank=None,
            bm_stats=bm_stats,
            dense_stats=None,
            rerank_stats=None,
            decay=_decay_factor(item.get("mtime")),
        )
        rows.append({
            "score": score,
            "final_score": score,
            "bm25_score": bm,
            "id": item.get("chunk_id"),
            "chunk_id": item.get("chunk_id"),
            "doc_id": item.get("doc_id"),
            "path": item.get("path"),
            "chunk_start": item.get("chunk_start"),
            "chunk_end": item.get("chunk_end"),
            "text": item.get("text"),
            "pages": item.get("pages"),
            "section_path": item.get("section_path"),
            "element_ids": item.get("element_ids"),
            "bboxes": item.get("bboxes"),
            "types": item.get("types"),
            "source_tools": item.get("source_tools"),
            "table_headers": item.get("table_headers"),
            "table_units": item.get("table_units"),
        })
    await _ensure_row_texts(rows, collection, subjects)
    return rows


async def _run_rerank(
    collection: str,
    query: str,
    query_vec: List[float],
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
) -> List[Dict[str, Any]]:
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    try:
        start = time.perf_counter()
        hits = await asyncio.to_thread(
            qdr.search,
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        timings["qdrant_ms"] = timings.get("qdrant_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return [{"error": "qdrant_search_failed", "detail": str(exc)}]

    await _hydrate_qdrant_hits(hits)
    texts = [str((h.payload or {}).get("text") or "")[:RERANK_MAX_CHARS] for h in hits]
    try:
        start = time.perf_counter()
        rr = await asyncio.to_thread(
            requests.post,
            f"{TEI_RERANK_URL}/rerank",
            json={"query": query, "texts": texts, "raw_scores": False},
            timeout=90,
        )
        rr.raise_for_status()
        timings["rerank_ms"] = timings.get("rerank_ms", 0.0) + (time.perf_counter() - start) * 1000.0
        data = rr.json()
        if isinstance(data, list):
            order = sorted(data, key=lambda r: r.get("score", 0.0), reverse=True)
        else:
            results = data.get("results", [])
            order = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        rerank_scores = {int(o.get("index", 0)): float(o.get("score", 0.0) or 0.0) for o in order}
    except Exception as exc:
        logger.warning("Rerank failed, falling back to semantic results: %s", exc)
        fallback_rows: List[Dict[str, Any]] = []
        for h in hits[: top_k]:
            pl = h.payload or {}
            fallback_rows.append({
                "score": getattr(h, "score", 0.0),
                "dense_score": getattr(h, "score", 0.0),
                "id": str(getattr(h, "id", "")),
                "chunk_id": str(getattr(h, "id", "")),
                "doc_id": pl.get("doc_id"),
                "path": pl.get("path"),
                "chunk_start": pl.get("chunk_start"),
                "chunk_end": pl.get("chunk_end"),
                "text": pl.get("text"),
                "pages": pl.get("page_numbers") or pl.get("pages"),
                "section_path": pl.get("section_path"),
                "element_ids": pl.get("element_ids"),
                "bboxes": pl.get("bboxes"),
                "types": pl.get("types"),
                "source_tools": pl.get("source_tools"),
                "table_headers": pl.get("table_headers"),
                "table_units": pl.get("table_units"),
            })
        await _ensure_row_texts(fallback_rows, collection, subjects)
        return fallback_rows

    dense_values = [getattr(h, "score", 0.0) for h in hits]
    dense_stats = (min(dense_values), max(dense_values)) if dense_values else None
    rerank_values = list(rerank_scores.values())
    rerank_stats = (min(rerank_values), max(rerank_values)) if rerank_values else None
    scored = []
    for o in order:
        idx = o.get("index", 0)
        base = float(o.get("score", 0.0) or 0.0)
        payload = hits[idx].payload or {}
        final = _combined_score(
            bm25=None,
            dense=getattr(hits[idx], "score", 0.0),
            rerank=base,
            bm_stats=None,
            dense_stats=dense_stats,
            rerank_stats=rerank_stats,
            decay=_decay_factor(payload.get("mtime")),
        )
        scored.append((final, idx))
    scored.sort(key=lambda t: t[0], reverse=True)
    final_scores = {idx: score for score, idx in scored}

    rows: List[Dict[str, Any]] = []
    for _, idx in scored[:return_k]:
        hit = hits[idx]
        payload = hit.payload or {}
        row = {
            "score": final_scores.get(idx, 0.0),
            "final_score": final_scores.get(idx, 0.0),
            "rerank_score": rerank_scores.get(idx, 0.0),
            "dense_score": getattr(hit, "score", 0.0),
            "id": str(getattr(hit, "id", "")),
            "chunk_id": str(getattr(hit, "id", "")),
            "doc_id": payload.get("doc_id"),
            "path": payload.get("path"),
            "chunk_start": payload.get("chunk_start"),
            "chunk_end": payload.get("chunk_end"),
            "text": payload.get("text"),
            "pages": payload.get("page_numbers") or payload.get("pages"),
            "section_path": payload.get("section_path"),
            "element_ids": payload.get("element_ids"),
            "bboxes": payload.get("bboxes"),
            "types": payload.get("types"),
            "source_tools": payload.get("source_tools"),
            "table_headers": payload.get("table_headers"),
            "table_units": payload.get("table_units"),
        }
        if NEIGHBOR_CHUNKS > 0 and row["doc_id"] and row["chunk_start"] is not None:
            neigh = _fts_neighbors(row["doc_id"], int(row["chunk_start"]), NEIGHBOR_CHUNKS)
            if neigh:
                ordered = sorted(neigh, key=lambda r: int(r.get("chunk_start", 0) or 0))
                txt = "\n".join([str(r.get("text") or "") for r in ordered]).strip()
                if txt:
                    row["text"] = txt
                row["chunk_start"] = int(ordered[0].get("chunk_start", row["chunk_start"]))
                row["chunk_end"] = int(ordered[-1].get("chunk_end", row["chunk_end"]))
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    return rows


async def _run_hybrid(
    collection: str,
    query: str,
    query_vec: List[float],
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
) -> List[Dict[str, Any]]:
    limit = min(retrieve_k, RERANK_MAX_ITEMS)
    try:
        start = time.perf_counter()
        dense_hits = await asyncio.to_thread(
            qdr.search,
            collection_name=collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        timings["qdrant_ms"] = timings.get("qdrant_ms", 0.0) + (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        return [{"error": "qdrant_search_failed", "detail": str(exc)}]

    await _hydrate_qdrant_hits(dense_hits)
    start = time.perf_counter()
    lexical_hits = await asyncio.to_thread(_fts_search, query, limit)
    timings["fts_ms"] = timings.get("fts_ms", 0.0) + (time.perf_counter() - start) * 1000.0

    dense_list: List[Dict[str, Any]] = []
    for h in dense_hits:
        payload = h.payload or {}
        dense_list.append({
            "chunk_id": str(getattr(h, "id", "")),
            "doc_id": payload.get("doc_id"),
            "path": payload.get("path"),
            "filename": payload.get("filename"),
            "chunk_start": payload.get("chunk_start"),
            "chunk_end": payload.get("chunk_end"),
            "mtime": payload.get("mtime"),
            "text": payload.get("text"),
            "dense_score": getattr(h, "score", None),
            "pages": payload.get("page_numbers") or payload.get("pages"),
            "section_path": payload.get("section_path"),
            "element_ids": payload.get("element_ids"),
            "bboxes": payload.get("bboxes"),
            "types": payload.get("types"),
            "source_tools": payload.get("source_tools"),
            "table_headers": payload.get("table_headers"),
            "table_units": payload.get("table_units"),
        })

    fts_list: List[Dict[str, Any]] = list(lexical_hits)
    fused = _rrf_fuse([dense_list, fts_list], id_getter=lambda x: x.get("chunk_id"))
    fused = fused[:limit]

    texts = [str(x.get("text") or "")[:RERANK_MAX_CHARS] for x in fused]
    try:
        start = time.perf_counter()
        rr = await asyncio.to_thread(
            requests.post,
            f"{TEI_RERANK_URL}/rerank",
            json={"query": query, "texts": texts, "raw_scores": False},
            timeout=90,
        )
        rr.raise_for_status()
        timings["rerank_ms"] = timings.get("rerank_ms", 0.0) + (time.perf_counter() - start) * 1000.0
        data = rr.json()
        if isinstance(data, list):
            order = sorted(data, key=lambda r: r.get("score", 0.0), reverse=True)
        else:
            results = data.get("results", [])
            order = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        rerank_scores = {int(o.get("index", 0)): float(o.get("score", 0.0) or 0.0) for o in order}
    except Exception as exc:
        logger.warning("Rerank failed, falling back to fused order: %s", exc)
        fallback_rows: List[Dict[str, Any]] = []
        for x in fused[: top_k]:
            fallback_rows.append({
                "score": x.get("_rrf_score"),
                "rrf_score": x.get("_rrf_score"),
                "bm25_score": x.get("bm25"),
                "dense_score": x.get("dense_score"),
                "id": x.get("chunk_id"),
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "path": x.get("path"),
                "chunk_start": x.get("chunk_start"),
                "chunk_end": x.get("chunk_end"),
                "text": x.get("text"),
                "pages": x.get("pages"),
                "section_path": x.get("section_path"),
                "element_ids": x.get("element_ids"),
                "bboxes": x.get("bboxes"),
                "types": x.get("types"),
                "source_tools": x.get("source_tools"),
                "table_headers": x.get("table_headers"),
                "table_units": x.get("table_units"),
            })
        await _ensure_row_texts(fallback_rows, collection, subjects)
        return fallback_rows

    bm_values = [x.get("bm25") for x in fused if x.get("bm25") is not None]
    bm_stats = (min(bm_values), max(bm_values)) if bm_values else None
    dense_values = [x.get("dense_score") for x in fused if x.get("dense_score") is not None]
    dense_stats = (min(dense_values), max(dense_values)) if dense_values else None
    rerank_values = list(rerank_scores.values())
    rerank_stats = (min(rerank_values), max(rerank_values)) if rerank_values else None
    scored = []
    for o in order:
        idx = o.get("index", 0)
        base = float(o.get("score", 0.0) or 0.0)
        x = fused[idx]
        final = _combined_score(
            bm25=x.get("bm25"),
            dense=x.get("dense_score"),
            rerank=base,
            bm_stats=bm_stats,
            dense_stats=dense_stats,
            rerank_stats=rerank_stats,
            decay=_decay_factor(x.get("mtime")),
        )
        scored.append((final, idx))
    scored.sort(key=lambda t: t[0], reverse=True)
    final_scores = {idx: score for score, idx in scored}

    rows: List[Dict[str, Any]] = []
    for _, idx in scored[:return_k]:
        x = fused[idx]
        row = {
            "score": final_scores.get(idx, 0.0),
            "final_score": final_scores.get(idx, 0.0),
            "rerank_score": rerank_scores.get(idx, 0.0),
            "rrf_score": x.get("_rrf_score"),
            "bm25_score": x.get("bm25"),
            "dense_score": x.get("dense_score"),
            "id": x.get("chunk_id"),
            "chunk_id": x.get("chunk_id"),
            "doc_id": x.get("doc_id"),
            "path": x.get("path"),
            "chunk_start": x.get("chunk_start"),
            "chunk_end": x.get("chunk_end"),
            "text": x.get("text"),
            "pages": x.get("pages"),
            "section_path": x.get("section_path"),
            "element_ids": x.get("element_ids"),
            "bboxes": x.get("bboxes"),
            "types": x.get("types"),
            "source_tools": x.get("source_tools"),
            "table_headers": x.get("table_headers"),
            "table_units": x.get("table_units"),
        }
        if NEIGHBOR_CHUNKS > 0 and row["doc_id"] and row["chunk_start"] is not None:
            neigh = _fts_neighbors(row["doc_id"], int(row["chunk_start"]), NEIGHBOR_CHUNKS)
            if neigh:
                ordered = sorted(neigh, key=lambda r: int(r.get("chunk_start", 0) or 0))
                txt = "\n".join([str(r.get("text") or "") for r in ordered]).strip()
                if txt:
                    row["text"] = txt
                row["chunk_start"] = int(ordered[0].get("chunk_start", row["chunk_start"]))
                row["chunk_end"] = int(ordered[-1].get("chunk_end", row["chunk_end"]))
        rows.append(row)
    await _ensure_row_texts(rows, collection, subjects)
    return rows


async def _execute_search(
    route: str,
    collection: str,
    query: str,
    query_vec: List[float],
    retrieve_k: int,
    return_k: int,
    top_k: int,
    subjects: List[str],
    timings: Dict[str, float],
) -> List[Dict[str, Any]]:
    if route == "sparse":
        return await _run_sparse(collection, query, retrieve_k, return_k, top_k, subjects, timings)
    if route == "semantic":
        return await _run_semantic(collection, query, query_vec, top_k, return_k, subjects, timings)
    if route == "rerank":
        return await _run_rerank(collection, query, query_vec, retrieve_k, return_k, top_k, subjects, timings)
    if route == "hybrid":
        return await _run_hybrid(collection, query, query_vec, retrieve_k, return_k, top_k, subjects, timings)
    return [{"error": f"invalid route '{route}'"}]


# ---- Register search tools per scope ---------------------------------------
def _register_scope(slug: str, collection: str, title: Optional[str] = None) -> None:
    tool_name = f"search_{slug}"
    tool_title = f"{title or slug}: Search"

    @mcp.tool(name=tool_name, title=tool_title)
    async def search(
        ctx: Context,
        query: str,
        mode: str = "rerank",  # "semantic", "rerank", "hybrid", or "auto"
        top_k: int = 8,
        retrieve_k: int = 24,
        return_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Vector search (semantic), vector+rerank, hybrid, or auto-planned retrieval."""
        start_time = time.perf_counter()
        timings: Dict[str, float] = {}
        route_retrieve_val = retrieve_k
        def finalize(rows: List[Dict[str, Any]], route_name: str, hyde_used: bool = False) -> List[Dict[str, Any]]:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            clean_rows = [r for r in rows if isinstance(r, dict) and not r.get("note") and not r.get("abstain")]
            payload = {
                "slug": slug,
                "mode": mode,
                "route": route_name,
                "retrieve_k": route_retrieve_val,
                "return_k": return_k,
                "top_k": top_k,
                "result_count": len(clean_rows),
                "best_score": _best_score(rows),
                "hyde_used": hyde_used,
                "duration_ms": round(duration_ms, 3),
                "abstain": any(isinstance(r, dict) and r.get("abstain") for r in rows),
                "timings": {k: round(v, 3) for k, v in timings.items()},
                "subjects": [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in subjects if s],
                "results": [
                    {
                        "doc_id": row.get("doc_id"),
                        "element_ids": row.get("element_ids"),
                        "chunk_id": row.get("chunk_id"),
                        "score": row.get("score"),
                    }
                    for row in clean_rows[:return_k]
                ],
            }
            try:
                logger.info("search_metrics %s", json.dumps(payload))
            except Exception:
                logger.debug("failed to log search metrics", exc_info=True)
            return rows

        valid_modes = {"semantic", "rerank", "hybrid", "sparse", "auto"}
        if mode not in valid_modes:
            return finalize([{"error": f"invalid mode '{mode}', expected one of {sorted(valid_modes)}"}], mode)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            return finalize([
                {"error": "invalid top_k", "detail": "top_k must be int between 1 and 100"}
            ], mode)
        if not isinstance(retrieve_k, int) or retrieve_k < 1 or retrieve_k > 256:
            return finalize([
                {"error": "invalid retrieve_k", "detail": "retrieve_k must be int between 1 and 256"}
            ], mode)
        if not isinstance(return_k, int) or return_k < 1 or return_k > retrieve_k:
            return finalize([
                {"error": "invalid return_k", "detail": "return_k must be int between 1 and retrieve_k"}
            ], mode)
        try:
            embed_start = time.perf_counter()
            vec = await embed_query(query, normalize=True)
            timings["embed_ms"] = (time.perf_counter() - embed_start) * 1000.0
        except Exception as e:
            return finalize([
                {"error": "embedding_failed", "detail": str(e)}
            ], mode)

        subjects = get_subjects_from_context(ctx)
        planned_route: Optional[Dict[str, Any]] = None
        route = mode
        route_retrieve = retrieve_k
        if mode == "auto":
            try:
                planned_route = await plan_route(query)
            except Exception as exc:
                logger.debug("plan_route failed, falling back to rerank: %s", exc)
                planned_route = {}
            route = str((planned_route or {}).get("route") or "rerank")
            if route not in {"semantic", "rerank", "hybrid", "sparse"}:
                route = "rerank"
            planned_k = (planned_route or {}).get("k")
            if isinstance(planned_k, int) and 1 <= planned_k <= 256:
                route_retrieve = planned_k
        route_retrieve = max(return_k, min(route_retrieve, 256))
        route_retrieve_val = route_retrieve

        rows = await _execute_search(
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

        if rows and isinstance(rows[0], dict) and rows[0].get("error"):
            return finalize(rows, route)

        for row in rows:
            if isinstance(row, dict) and "route" not in row:
                row["route"] = route

        best = _best_score(rows)
        hyde_used = False
        if ANSWERABILITY_THRESHOLD > 0.0 and best < ANSWERABILITY_THRESHOLD:
            hypo = await hyde(query)
            if hypo:
                hyde_start = time.perf_counter()
                try:
                    hypo_vec = await embed_query(hypo, normalize=True)
                except Exception as exc:
                    logger.debug("HyDE embed failed: %s", exc)
                    hypo_vec = None
                if hypo_vec is not None:
                    hyde_rows = await _execute_search(
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
                    timings["hyde_ms"] = timings.get("hyde_ms", 0.0) + (time.perf_counter() - hyde_start) * 1000.0
                    if hyde_rows and isinstance(hyde_rows[0], dict) and hyde_rows[0].get("error"):
                        return finalize(hyde_rows, "semantic")
                    for row in hyde_rows:
                        if isinstance(row, dict) and "route" not in row:
                            row["route"] = "semantic"
                    hyde_best = _best_score(hyde_rows)
                    if hyde_best >= ANSWERABILITY_THRESHOLD:
                        hyde_rows.insert(0, {"note": "HyDE retry satisfied threshold", "base_route": route})
                        return finalize(hyde_rows, "semantic", hyde_used=True)
            return finalize(
                [{"abstain": True, "reason": "low_answerability", "top_score": best, "threshold": ANSWERABILITY_THRESHOLD}],
                route,
            )

        if route != "sparse" and _needs_sparse_retry(query, rows):
            sparse_rows = await _execute_search(
                route="sparse",
                collection=collection,
                query=query,
                query_vec=vec,
                retrieve_k=max(route_retrieve, 32),
                return_k=return_k,
                top_k=top_k,
                subjects=subjects,
                timings=timings,
            )
            if sparse_rows and not sparse_rows[0].get("error"):
                for row in sparse_rows:
                    if isinstance(row, dict) and "route" not in row:
                        row["route"] = "sparse"
                rows = sparse_rows
                route = "sparse"

        return finalize(rows, route, hyde_used)


    @mcp.tool(name=f"open_{slug}", title=f"{title or slug}: Open Chunk")
    async def open_chunk(
        ctx: Context,
        chunk_id: Optional[str] = None,
        element_id: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        target = chunk_id or _lookup_chunk_by_element(element_id or "")
        if not target:
            return {"error": "missing_target", "detail": "Provide chunk_id or element_id"}
        row = {"id": target}
        await _ensure_row_texts([row], collection, get_subjects_from_context(ctx))
        if not row.get("text"):
            return {"error": "not_found"}
        text = row.get("text", "")
        if start is not None or end is not None:
            s = max(0, int(start or 0))
            e = int(end) if end is not None else len(text)
            row["text"] = text[s:e]
            row["slice"] = {"start": s, "end": e}
        row["chunk_id"] = target
        return row


    @mcp.tool(name=f"neighbors_{slug}", title=f"{title or slug}: Neighbor Chunks")
    async def neighbor_chunks(
        ctx: Context,
        chunk_id: str,
        n: int = 1,
    ) -> List[Dict[str, Any]]:
        if not chunk_id:
            return [{"error": "missing_chunk_id"}]
        seed = {"id": chunk_id}
        subjects_local = get_subjects_from_context(ctx)
        await _ensure_row_texts([seed], collection, subjects_local)
        doc_id = seed.get("doc_id")
        chunk_start = seed.get("chunk_start")
        if doc_id is None or chunk_start is None:
            return [{"error": "not_found"}]
        neighbor_rows_raw = _fts_neighbors(str(doc_id), int(chunk_start), max(1, int(n)))
        rows = []
        for raw in neighbor_rows_raw:
            rows.append({
                "id": raw.get("chunk_id"),
                "chunk_id": raw.get("chunk_id"),
                "doc_id": raw.get("doc_id"),
                "path": raw.get("path"),
                "chunk_start": raw.get("chunk_start"),
                "chunk_end": raw.get("chunk_end"),
                "text": raw.get("text"),
                "pages": raw.get("pages"),
                "section_path": raw.get("section_path"),
                "element_ids": raw.get("element_ids"),
            })
        await _ensure_row_texts(rows, collection, subjects_local)
        return rows


    @mcp.tool(name=f"summary_{slug}", title=f"{title or slug}: Section Summary")
    async def summary_tool(
        ctx: Context,
        topic: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        topic = (topic or "").strip()
        if not topic:
            return [{"error": "missing_topic"}]
        results = await asyncio.to_thread(
            query_summaries,
            collection,
            topic,
            max(1, int(limit)),
            SUMMARY_DB_PATH,
        )
        return results or [{"info": "no_matches"}]


    @mcp.tool(name=f"graph_{slug}", title=f"{title or slug}: Graph Neighbors")
    async def graph_tool(
        ctx: Context,
        node_id: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        node_id = (node_id or "").strip()
        if not node_id:
            return {"error": "missing_node_id"}
        data = await asyncio.to_thread(graph_neighbors, node_id, max(1, int(limit)), GRAPH_DB_PATH)
        if not data.get("node"):
            return {"error": "not_found"}
        return data


for slug, cfg in SCOPES.items():
    _register_scope(slug, cfg["collection"], cfg.get("title"))


if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        logger.exception("MCP server crashed")
        raise
