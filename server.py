import json
import os
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional

import requests
from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
import sqlite3


# ---- Env & config -----------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "snowflake-arctic-embed:xs")
TEI_RERANK_URL = os.getenv("TEI_RERANK_URL", "http://localhost:8087")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
FTS_DB_PATH = os.getenv("FTS_DB_PATH", "data/fts.db")

# JSON like: {"kb":{"collection":"snowflake_kb","title":"Company KB"}}
# Backcompat: allow STELLA_SCOPES if NOMIC_KB_SCOPES not set
SCOPES_ENV = os.getenv("NOMIC_KB_SCOPES") or os.getenv("STELLA_SCOPES") or '{"kb":{"collection":"snowflake_kb","title":"Company KB"}}'
SCOPES: Dict[str, Dict[str, Any]] = json.loads(SCOPES_ENV)

# ---- Utilities --------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("kb-mcp")

qdr = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
mcp = FastMCP(name="knowledge-base", version="1.0.0", instructions="Vector search with Qdrant + Ollama embeddings and TEI reranker")

# Rerank constraints to avoid payload-too-large
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "700"))
RERANK_MAX_ITEMS = int(os.getenv("RERANK_MAX_ITEMS", "16"))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
# Neighbor packaging and scoring controls
NEIGHBOR_CHUNKS = int(os.getenv("NEIGHBOR_CHUNKS", "1"))
ANSWERABILITY_THRESHOLD = float(os.getenv("ANSWERABILITY_THRESHOLD", "0.0"))
DECAY_HALF_LIFE_DAYS = float(os.getenv("DECAY_HALF_LIFE_DAYS", "0"))
DECAY_STRENGTH = float(os.getenv("DECAY_STRENGTH", "0.0"))


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
        return fts_search(FTS_DB_PATH, query, limit)
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


# ---- Register search tools per scope ---------------------------------------
def _register_scope(slug: str, collection: str, title: Optional[str] = None) -> None:
    tool_name = f"search_{slug}"
    tool_title = f"{title or slug}: Search"

    @mcp.tool(name=tool_name, title=tool_title)
    async def search(
        ctx: Context,
        query: str,
        mode: str = "rerank",  # "semantic", "rerank", or "hybrid"
        top_k: int = 8,
        retrieve_k: int = 24,
        return_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Vector search (semantic), vector+rerank, or hybrid (lexical + dense via RRF). Returns path and offsets."""
        if mode not in {"semantic", "rerank", "hybrid"}:
            return [{"error": f"invalid mode '{mode}', expected 'semantic', 'rerank', or 'hybrid'"}]
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            return [{"error": "invalid top_k", "detail": "top_k must be int between 1 and 100"}]
        if not isinstance(retrieve_k, int) or retrieve_k < 1 or retrieve_k > 256:
            return [{"error": "invalid retrieve_k", "detail": "retrieve_k must be int between 1 and 256"}]
        if not isinstance(return_k, int) or return_k < 1 or return_k > retrieve_k:
            return [{"error": "invalid return_k", "detail": "return_k must be int between 1 and retrieve_k"}]
        try:
            vec = await embed_query(query, normalize=True)
        except Exception as e:
            return [{"error": "embedding_failed", "detail": str(e)}]

        if mode == "semantic":
            try:
                hits = await asyncio.to_thread(
                    qdr.search,
                    collection_name=collection,
                    query_vector=vec,
                    limit=top_k,
                    with_payload=True,
                )
            except Exception as e:
                return [{"error": "qdrant_search_failed", "detail": str(e)}]
            rows: List[Dict[str, Any]] = []
            for h in hits:
                pl = h.payload or {}
                rows.append({
                    "score": h.score,
                    "id": h.id,
                    "doc_id": pl.get("doc_id"),
                    "path": pl.get("path"),
                    "chunk_start": pl.get("chunk_start"),
                    "chunk_end": pl.get("chunk_end"),
                    "text": pl.get("text"),
                })
            return rows

        if mode == "rerank":
            try:
                hits = await asyncio.to_thread(
                    qdr.search,
                    collection_name=collection,
                    query_vector=vec,
                    limit=min(retrieve_k, RERANK_MAX_ITEMS),
                    with_payload=True,
                )
            except Exception as e:
                return [{"error": "qdrant_search_failed", "detail": str(e)}]

            texts = [str((h.payload or {}).get("text") or "")[:RERANK_MAX_CHARS] for h in hits]
            try:
                rr = await asyncio.to_thread(
                    requests.post,
                    f"{TEI_RERANK_URL}/rerank",
                    json={"query": query, "texts": texts, "raw_scores": False},
                    timeout=90,
                )
                rr.raise_for_status()
                data = rr.json()
                if isinstance(data, list):
                    order = sorted(data, key=lambda r: r.get("score", 0.0), reverse=True)
                else:
                    results = data.get("results", [])
                    order = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
                rerank_scores = {int(o.get("index", 0)): float(o.get("score", 0.0) or 0.0) for o in order}
            except Exception as e:
                logger.warning("Rerank failed, falling back to semantic results: %s", e)
                rows: List[Dict[str, Any]] = []
                for h in hits[:top_k]:
                    pl = h.payload or {}
                    rows.append({
                        "score": h.score,
                        "id": h.id,
                        "doc_id": pl.get("doc_id"),
                        "path": pl.get("path"),
                        "chunk_start": pl.get("chunk_start"),
                        "chunk_end": pl.get("chunk_end"),
                        "text": pl.get("text"),
                    })
                return rows

            scored = []
            for o in order:
                i = o.get("index", 0)
                base = float(o.get("score", 0.0) or 0.0)
                pl = (hits[i].payload or {})
                final = base * _decay_factor(pl.get("mtime"))
                scored.append((final, i))
            scored.sort(key=lambda t: t[0], reverse=True)
            final_scores = {i: s for (s, i) in scored}

            if ANSWERABILITY_THRESHOLD > 0.0 and (scored[0][0] if scored else 0.0) < ANSWERABILITY_THRESHOLD:
                return [{"abstain": True, "reason": "low_answerability", "top_score": scored[0][0] if scored else 0.0, "threshold": ANSWERABILITY_THRESHOLD}]

            idxs = [i for _s, i in scored][:return_k]
            rows: List[Dict[str, Any]] = []
            for i in idxs:
                h = hits[i]
                pl = h.payload or {}
                base_row = {
                    "score": final_scores.get(i, 0.0),
                    "final_score": final_scores.get(i, 0.0),
                    "rerank_score": rerank_scores.get(i, 0.0),
                    "dense_score": h.score,
                    "id": h.id,
                    "doc_id": pl.get("doc_id"),
                    "path": pl.get("path"),
                    "chunk_start": pl.get("chunk_start"),
                    "chunk_end": pl.get("chunk_end"),
                    "text": pl.get("text"),
                }
                if NEIGHBOR_CHUNKS > 0 and base_row["doc_id"] and base_row["chunk_start"] is not None:
                    neigh = _fts_neighbors(base_row["doc_id"], int(base_row["chunk_start"]), NEIGHBOR_CHUNKS)
                    if neigh:
                        ordered = sorted(neigh, key=lambda r: int(r.get("chunk_start", 0) or 0))
                        txt = "\n".join([str(r.get("text") or "") for r in ordered]).strip()
                        base_row["text"] = txt or base_row["text"]
                        base_row["chunk_start"] = int(ordered[0].get("chunk_start", base_row["chunk_start"]))
                        base_row["chunk_end"] = int(ordered[-1].get("chunk_end", base_row["chunk_end"]))
                rows.append(base_row)
            return rows

        # hybrid mode
        try:
            dense_hits = await asyncio.to_thread(
                qdr.search,
                collection_name=collection,
                query_vector=vec,
                limit=min(retrieve_k, RERANK_MAX_ITEMS),
                with_payload=True,
            )
        except Exception as e:
            return [{"error": "qdrant_search_failed", "detail": str(e)}]

        lexical_hits = await asyncio.to_thread(_fts_search, query, min(retrieve_k, RERANK_MAX_ITEMS))

        dense_list: List[Dict[str, Any]] = []
        for h in dense_hits:
            pl = h.payload or {}
            dense_list.append({
                "chunk_id": str(h.id),
                "doc_id": pl.get("doc_id"),
                "path": pl.get("path"),
                "filename": pl.get("filename"),
                "chunk_start": pl.get("chunk_start"),
                "chunk_end": pl.get("chunk_end"),
                "mtime": pl.get("mtime"),
                "text": pl.get("text"),
                "dense_score": getattr(h, 'score', None),
            })

        fts_list: List[Dict[str, Any]] = []
        for r in lexical_hits:
            fts_list.append(r)

        fused = _rrf_fuse([dense_list, fts_list], id_getter=lambda x: x.get("chunk_id"))
        fused = fused[: min(retrieve_k, RERANK_MAX_ITEMS)]

        texts = [str(x.get("text") or "")[:RERANK_MAX_CHARS] for x in fused]
        try:
            rr = await asyncio.to_thread(
                requests.post,
                f"{TEI_RERANK_URL}/rerank",
                json={"query": query, "texts": texts, "raw_scores": False},
                timeout=90,
            )
            rr.raise_for_status()
            data = rr.json()
            if isinstance(data, list):
                order = sorted(data, key=lambda r: r.get("score", 0.0), reverse=True)
            else:
                results = data.get("results", [])
                order = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
            rerank_scores = {int(o.get("index", 0)): float(o.get("score", 0.0) or 0.0) for o in order}
        except Exception as e:
            logger.warning("Rerank failed, falling back to fused order: %s", e)
            rows: List[Dict[str, Any]] = []
            for x in fused[:top_k]:
                rows.append({
                    "score": x.get("_rrf_score"),
                    "rrf_score": x.get("_rrf_score"),
                    "bm25_score": x.get("bm25"),
                    "dense_score": x.get("dense_score"),
                    "id": x.get("chunk_id"),
                    "doc_id": x.get("doc_id"),
                    "path": x.get("path"),
                    "chunk_start": x.get("chunk_start"),
                    "chunk_end": x.get("chunk_end"),
                    "text": x.get("text"),
                })
            return rows

        scored = []
        for o in order:
            i = o.get("index", 0)
            base = float(o.get("score", 0.0) or 0.0)
            x = fused[i]
            final = base * _decay_factor(x.get("mtime"))
            scored.append((final, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        final_scores = {i: s for (s, i) in scored}

        if ANSWERABILITY_THRESHOLD > 0.0 and (scored[0][0] if scored else 0.0) < ANSWERABILITY_THRESHOLD:
            return [{"abstain": True, "reason": "low_answerability", "top_score": scored[0][0] if scored else 0.0, "threshold": ANSWERABILITY_THRESHOLD}]

        idxs = [i for _s, i in scored][:return_k]
        rows: List[Dict[str, Any]] = []
        for i in idxs:
            x = fused[i]
            row = {
                "score": final_scores.get(i, 0.0),
                "final_score": final_scores.get(i, 0.0),
                "rerank_score": rerank_scores.get(i, 0.0),
                "rrf_score": x.get("_rrf_score"),
                "bm25_score": x.get("bm25"),
                "dense_score": x.get("dense_score"),
                "id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "path": x.get("path"),
                "chunk_start": x.get("chunk_start"),
                "chunk_end": x.get("chunk_end"),
                "text": x.get("text"),
            }
            if NEIGHBOR_CHUNKS > 0 and row["doc_id"] and row["chunk_start"] is not None:
                neigh = _fts_neighbors(row["doc_id"], int(row["chunk_start"]), NEIGHBOR_CHUNKS)
                if neigh:
                    ordered = sorted(neigh, key=lambda r: int(r.get("chunk_start", 0) or 0))
                    txt = "\n".join([str(r.get("text") or "") for r in ordered]).strip()
                    row["text"] = txt or row["text"]
                    row["chunk_start"] = int(ordered[0].get("chunk_start", row["chunk_start"]))
                    row["chunk_end"] = int(ordered[-1].get("chunk_end", row["chunk_end"]))
            rows.append(row)
        return rows


for slug, cfg in SCOPES.items():
    _register_scope(slug, cfg["collection"], cfg.get("title"))


if __name__ == "__main__":
    try:
        mcp.run()
    except Exception:
        logger.exception("MCP server crashed")
        raise
