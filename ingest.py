import argparse
import os
import pathlib
import uuid
import time
import fnmatch
import hashlib
from typing import Any, Dict, List, Tuple
import gc
import re
import json

import requests
import numpy as np
from tqdm import tqdm

from ingest_blocks import extract_document_blocks, chunk_blocks
from graph_builder import update_graph
from summary_index import upsert_summaries


# Default skip patterns to avoid noisy/system files
DEFAULT_SKIP_PATTERNS = [
    "*/.*",  # hidden files/dirs (dotfiles)
    "*/~$*",  # Office lock/temp files
    "*/Thumbs.db", "*/thumbs.db",
    "*/Desktop.ini", "*/desktop.ini",
    "*.tmp", "*.temp", "*.crdownload", "*.partial", "*.swp", "*.swo",
    "*/$RECYCLE.BIN/*",
    "*/System Volume Information/*",
    "*.thmx",
    "*.mcdx",
    "*/Markitdown/*",
]


def file_uri(p: pathlib.Path) -> str:
    return p.resolve().as_uri()


# -------------------- Extractors --------------------
def extract_markitdown(p: pathlib.Path) -> Tuple[str, Dict[str, Any]]:
    from markitdown import MarkItDown
    md = MarkItDown()
    res = md.convert(str(p))
    return res.text_content or "", {}


def extract_docling(p: pathlib.Path) -> Tuple[str, Dict[str, Any]]:
    from docling.document_converter import DocumentConverter
    doc = DocumentConverter().convert(str(p)).document
    return doc.export_to_markdown() or "", {}


def extract_pdf_pymupdf(p: pathlib.Path) -> Tuple[str, Dict[str, Any]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return "", {}
    try:
        texts = []
        spans: List[Tuple[int, int, int]] = []
        with fitz.open(str(p)) as doc:
            offset = 0
            total_pages = len(doc)
            for idx, page in enumerate(doc):
                page_text = page.get_text("text") or ""
                texts.append(page_text)
                end_offset = offset + len(page_text)
                spans.append((offset, end_offset, idx + 1))
                # Account for the newline inserted by join (except after last page)
                offset = end_offset
                if idx < total_pages - 1:
                    offset += 1
        raw_text = "\n".join(texts)
        if not raw_text:
            return "", {}
        leading_trim = len(raw_text) - len(raw_text.lstrip())
        text = raw_text.strip()
        adjusted_spans = []
        text_len = len(text)
        for start, end, page_num in spans:
            adj_start = max(0, start - leading_trim)
            adj_end = max(adj_start, min(text_len, end - leading_trim))
            if adj_end <= adj_start:
                continue
            if adj_start >= text_len:
                continue
            if adj_end <= 0:
                continue
            adjusted_spans.append(
                {
                    "page": page_num,
                    "start": adj_start,
                    "end": adj_end,
                }
            )
        return text, {"page_spans": adjusted_spans}
    except Exception:
        return "", {}


def choose_extractor(extractor: str, p: pathlib.Path):
    # Prefer MarkItDown for speed, even for PDFs
    if extractor == "docling":
        return extract_docling
    return extract_markitdown


def extract_with_fallback(extractor: str, p: pathlib.Path) -> Tuple[str, Dict[str, Any]]:
    """Try preferred extractor; for PDFs, fall back to Docling on failure.

    Docling fallback can be disabled by setting env NO_DOCLING_FALLBACK to
    one of: '1', 'true', 'yes'.
    """
    text = ""
    meta: Dict[str, Any] = {}
    ext = p.suffix.lower()
    no_docling = os.getenv("NO_DOCLING_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
    # For PDFs, try fast PyMuPDF first
    if ext == ".pdf":
        text, meta = extract_pdf_pymupdf(p)
        if text.strip():
            return text, meta
    # Try preferred extractor (MarkItDown by default)
    try:
        text, meta = choose_extractor(extractor, p)(p)
    except Exception:
        text = ""
        meta = {}
    # If PDF and no text, try Docling as fallback (CPU-only to avoid CUDA issues)
    if not text.strip() and ext == ".pdf" and extractor != "docling" and not no_docling:
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            text, meta = extract_docling(p)
        except Exception:
            text, meta = "", {}
    return text or "", meta


# -------------------- Chunking --------------------
def fallback_chunk(text: str, max_chars: int = 1800, overlap: int = 150) -> List[Dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap >= max_chars:
        overlap = max(0, max_chars // 4)
    out: List[Dict[str, Any]] = []
    i, n = 0, len(text)
    step = max_chars - overlap
    while i < n:
        j = min(i + max_chars, n)
        segment = text[i:j]
        if segment.strip():
            out.append(
                {
                    "start": i,
                    "end": j,
                    "text": segment,
                    "meta": {"chunk_type": "paragraph", "heading_path": []},
                }
            )
        if j == n:
            break
        i += step
    return out


# -------------------- Embeddings (Ollama) --------------------
def l2_normalize(mat: List[List[float]]) -> List[List[float]]:
    arr = np.array(mat, dtype="float32")
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (arr / n).tolist()

def embed_texts(ollama_url: str, model: str, texts: List[str], batch_size: int = 32, timeout: int = 120,
                normalize: bool = True, parallel: int = 1, num_threads: int = 8, keep_alive: str = "1h",
                force_per_item: bool = False) -> List[List[float]]:
    """Embed texts via Ollama.

    By default, prefers batch endpoint /api/embed. If force_per_item is True,
    uses per-item /api/embeddings instead (optionally with parallel workers).
    """
    embeddings: List[List[float]] = []
    headers = {"content-type": "application/json"}
    def embed_via_per_item(batch: List[str]) -> List[List[float]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        per_item: List[List[float]] = [None] * len(batch)
        def do_one(idx_text):
            idx, t = idx_text
            r2 = requests.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": t, "keep_alive": keep_alive, "options": {"num_thread": num_threads}},
                timeout=timeout,
                headers=headers,
            )
            r2.raise_for_status()
            vec = r2.json().get("embedding")
            if vec is None:
                raise RuntimeError("Missing 'embedding' in /api/embeddings response")
            return idx, vec
        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(do_one, (j, t)) for j, t in enumerate(batch)]
                for fut in as_completed(futures):
                    j, vec = fut.result()
                    per_item[j] = vec
        else:
            for j, t in enumerate(batch):
                _, vec = do_one((j, t))
                per_item[j] = vec
        return per_item

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                if not force_per_item:
                    # Try batch endpoint first
                    r = requests.post(
                        f"{ollama_url}/api/embed",
                        json={"model": model, "input": batch, "keep_alive": keep_alive, "options": {"num_thread": num_threads}},
                        timeout=timeout,
                        headers=headers,
                    )
                    if r.status_code == 404:
                        raise requests.HTTPError("/api/embed not found", response=r)
                    r.raise_for_status()
                    data = r.json()
                    batch_emb = data.get("embeddings") or []
                    if normalize:
                        batch_emb = l2_normalize(batch_emb)
                    embeddings.extend(batch_emb)
                    break
                # Force per-item endpoint
                per_item = embed_via_per_item(batch)
                if normalize:
                    per_item = l2_normalize(per_item)
                embeddings.extend(per_item)
                break
            except Exception as e:
                # Fallback to per-item /api/embeddings if batch endpoint missing
                if not force_per_item and isinstance(e, requests.HTTPError) and getattr(e, 'response', None) is not None and e.response.status_code == 404:
                    try:
                        per_item = embed_via_per_item(batch)
                        if normalize:
                            per_item = l2_normalize(per_item)
                        embeddings.extend(per_item)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(1.5 * (attempt + 1))
                else:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))
    return embeddings


def embed_texts_robust(
    ollama_url: str,
    model: str,
    texts: List[str],
    timeout: int = 120,
    normalize: bool = True,
    num_threads: int = 8,
    keep_alive: str = "1h",
    max_retries: int = 2,
) -> List[List[float]]:
    """Embed texts one-by-one with per-item retries; returns list equal in length to texts with None for failures."""
    headers = {"content-type": "application/json"}
    out: List[List[float]] = [None] * len(texts)
    for i, t in enumerate(texts):
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": model, "prompt": t, "keep_alive": keep_alive, "options": {"num_thread": num_threads}},
                    timeout=timeout,
                    headers=headers,
                )
                r.raise_for_status()
                vec = r.json().get("embedding")
                if vec is None:
                    raise RuntimeError("Missing 'embedding' in /api/embeddings response")
                if normalize:
                    vec = l2_normalize([vec])[0]
                out[i] = vec
                break
            except Exception:
                if attempt == max_retries:
                    out[i] = None
                else:
                    time.sleep(1.0 * (attempt + 1))
                continue
    return out


def get_embedding_dim(ollama_url: str, model: str) -> int:
    try:
        vec = embed_texts(ollama_url, model, ["sanity"], batch_size=1, normalize=False)[0]
        return len(vec)
    except Exception as ex:
        raise SystemExit(f"Failed to probe Ollama at {ollama_url} for model {model}: {ex}")


# -------------------- Qdrant --------------------
_QDRANT_CLIENT = None
_QDRANT_TIMEOUT = None


def qdrant_distance(metric: str):
    from qdrant_client import models
    return {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }[metric]


def ensure_qdrant_collection(url: str, api_key: str, name: str, size: int, metric: str):
    from qdrant_client import QdrantClient, models
    global _QDRANT_CLIENT, _QDRANT_TIMEOUT
    if _QDRANT_CLIENT is None:
        client_kwargs = {"url": url, "api_key": api_key}
        if _QDRANT_TIMEOUT is not None:
            client_kwargs["timeout"] = _QDRANT_TIMEOUT
        _QDRANT_CLIENT = QdrantClient(**client_kwargs)
    dist = qdrant_distance(metric)
    if not _QDRANT_CLIENT.collection_exists(collection_name=name):
        _QDRANT_CLIENT.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=dist),
        )
    else:
        info = _QDRANT_CLIENT.get_collection(collection_name=name)
        cfg = info.config.params.vectors
        current_size = getattr(cfg, "size", None)
        current_dist = getattr(cfg, "distance", None)
        if current_size != size or current_dist != dist:
            raise SystemExit(
                f"Existing collection '{name}' has size={current_size}, distance={current_dist}; expected size={size}, distance={dist}."
                " Delete or recreate the collection manually before re-running ingest."
            )


def upsert_qdrant(collection: str, vectors: List[List[float]], payloads: List[dict], ids: List[str]):
    from qdrant_client import models
    points = [models.PointStruct(id=i, vector=v, payload=p)
              for i, v, p in zip(ids, vectors, payloads)]
    _QDRANT_CLIENT.upsert(collection_name=collection, points=points)


def qdrant_any_by_filter(collection: str, must: List[dict]) -> bool:
    from qdrant_client import models
    conds = []
    for c in must:
        key, val = c["key"], c["value"]
        conds.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
    flt = models.Filter(must=conds)
    points, _ = _QDRANT_CLIENT.scroll(collection_name=collection, scroll_filter=flt, limit=1, with_payload=False, with_vectors=False)
    return bool(points)


def qdrant_delete_by_doc_id(collection: str, doc_id: str):
    from qdrant_client import models
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    _QDRANT_CLIENT.delete(collection_name=collection, points_selector=models.FilterSelector(filter=flt))


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory with source documents")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="snowflake-arctic-embed:xs")
    ap.add_argument("--extractor", choices=["markitdown", "docling", "auto"], default="markitdown")

    # Metric/normalization
    ap.add_argument("--metric", choices=["cosine", "dot", "euclid"], default="cosine")

    # Chunking
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--overlap", type=int, default=150)

    # Embedding batch
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ollama-threads", type=int, default=8)
    ap.add_argument("--ollama-keepalive", default="1h")
    ap.add_argument("--ollama-timeout", type=int, default=120, help="HTTP timeout (seconds) for Ollama embedding requests")
    ap.add_argument("--ollama-per-item", action="store_true", help="Use per-item /api/embeddings endpoint instead of batch /api/embed")
    ap.add_argument("--embed-robust", action="store_true", help="Skip per-chunk embedding failures and continue (best for stubborn documents)")
    ap.add_argument("--embed-window-size", type=int, default=64, help="Embed this many chunks at a time in robust mode")

    # File-queue batching
    ap.add_argument("--max-docs-per-run", type=int, default=0, help="Process at most this many documents in a single run (0 = unlimited)")

    # Content filters
    ap.add_argument("--min-words", type=int, default=0, help="Skip documents with fewer than this many alphabetic words after extraction (0 = disable)")

    # File filtering
    ap.add_argument("--ext", default="", help="comma-separated list of extensions to include (e.g., .pdf,.docx). Empty = all")
    ap.add_argument("--skip", default="", help="comma-separated glob patterns to exclude (e.g., '*/node_modules/*,*.tmp')")
    ap.add_argument("--include", default="", help="comma-separated glob patterns to include; if set, only matching files are processed")
    ap.add_argument("--max-file-mb", type=int, default=64, help="Skip files larger than this size (MB)")
    ap.add_argument("--max-walk-depth", type=int, default=-1, help="Limit directory traversal depth relative to root (-1 = unlimited, 0 = only root files)")

    # Qdrant
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-collection", default="snowflake_kb")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--qdrant-timeout", type=int, default=300, help="Timeout in seconds for Qdrant HTTP requests")

    # Lexical FTS
    ap.add_argument("--fts-db", default=os.getenv("FTS_DB_PATH", "data/fts.db"), help="Path to SQLite FTS index DB")
    ap.add_argument("--no-fts", action="store_true", help="Do not write to lexical FTS index")
    ap.add_argument("--fts-only", action="store_true", help="Only update lexical FTS; skip embeddings and Qdrant")
    ap.add_argument("--fts-rebuild", action="store_true", help="Drop and recreate FTS table before ingest")
    ap.add_argument("--thin-payload", action="store_true", help="Store minimal payload in vector index (omit text)")

    # Incremental ingest
    ap.add_argument("--skip-existing", action="store_true", help="Skip files whose doc_id already exists in store")
    ap.add_argument("--changed-only", action="store_true", help="Only (re)ingest when content_hash differs; requires Qdrant store")
    ap.add_argument("--delete-changed", action="store_true", help="Delete existing chunks for a doc_id before reingesting when changed")

    args = ap.parse_args()

    if not args.thin_payload:
        thin_env = os.getenv("THIN_VECTOR_PAYLOAD", "").strip().lower()
        args.thin_payload = thin_env in {"1", "true", "yes"}

    root = pathlib.Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")

    # Normalize FTS DB path to absolute to avoid issues if CWD changes in libraries
    try:
        args.fts_db = os.path.abspath(args.fts_db)
    except Exception:
        pass

    global _QDRANT_TIMEOUT
    _QDRANT_TIMEOUT = args.qdrant_timeout

    include_exts = {e.strip().lower() for e in args.ext.split(',') if e.strip()} if args.ext else None
    skip_patterns = list(DEFAULT_SKIP_PATTERNS)
    if args.skip:
        skip_patterns += [s.strip() for s in args.skip.split(',') if s.strip()]
    include_patterns = [s.strip() for s in args.include.split(',') if s.strip()] if args.include else []

    # Initialize Qdrant collection if we are doing vector ingest
    if not args.fts_only:
        embedding_dim = get_embedding_dim(args.ollama_url, args.ollama_model)
        ensure_qdrant_collection(args.qdrant_url, args.qdrant_api_key, args.qdrant_collection, embedding_dim, args.metric)

    files = []
    root_parts = root.resolve().parts
    for path, dirnames, filenames in os.walk(root):
        # Prune traversal depth
        if args.max_walk_depth >= 0:
            try:
                rel_parts = pathlib.Path(path).resolve().parts[len(root_parts):]
                depth = len(rel_parts)
            except Exception:
                depth = 0
            if depth > args.max_walk_depth:
                continue
            if depth == args.max_walk_depth:
                dirnames[:] = []
        for fn in filenames:
            p = pathlib.Path(path) / fn
            if include_exts and p.suffix.lower() not in include_exts:
                continue
            if skip_patterns and any(fnmatch.fnmatch(str(p).lower(), pat.lower()) for pat in skip_patterns):
                continue
            if include_patterns and not any(fnmatch.fnmatch(str(p).lower(), pat.lower()) for pat in include_patterns):
                continue
            try:
                if p.stat().st_size > args.max_file_mb * 1024 * 1024:
                    continue
            except Exception:
                pass
            files.append(p)

    if not files:
        print("No files found to ingest.")
        return

    processed_docs = 0
    processed_chunks = 0
    errors = 0
    def alpha_word_count(text: str) -> int:
        # Count tokens with at least 2 alphabetic characters (A-Z). Fast heuristic for English text.
        tokens = re.findall(r"[A-Za-z]{2,}", text)
        return len(tokens)

    # Prepare FTS writer if needed
    fts_writer = None
    if not args.no_fts:
        try:
            # If doing a clean rebuild, remove old DB and WAL/SHM sidecars first
            if args.fts_rebuild:
                try:
                    base = args.fts_db
                    for side in ("", "-wal", "-shm"):
                        p = base + side
                        if os.path.exists(p):
                            os.remove(p)
                except Exception:
                    pass
            from lexical_index import FTSWriter
            fts_writer = FTSWriter(args.fts_db, recreate=args.fts_rebuild)
        except Exception as fex:
            print(f"WARN: Failed to initialize FTS writer: {fex}")
            fts_writer = None

    try:
        for p in tqdm(files, desc="Ingesting"):
            try:
                # Use stable doc_id derived from file URI; store POSIX path for MCP tools
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_uri(p)))
                path_str = p.resolve().as_posix()

                # Fast skip: if skipping existing docs, check before extraction
                if args.skip_existing:
                    if qdrant_any_by_filter(args.qdrant_collection, [{"key": "doc_id", "value": doc_id}]):
                        continue

                # Extract early and apply content filters before heavy work
                text, extraction_meta = extract_with_fallback(args.extractor, p)
                if not text.strip():
                    continue
                if args.min_words and alpha_word_count(text) < args.min_words:
                    continue
                triage_blocks, _ = extract_document_blocks(p, doc_id)
                chunks, raw_text = chunk_blocks(
                    triage_blocks,
                    args.max_chars,
                    overlap_sentences=max(1, args.overlap // 80 if args.overlap else 1),
                )
                if not raw_text.strip():
                    raw_text = text
                content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                mtime = int(p.stat().st_mtime)

                try:
                    update_graph(args.qdrant_collection, doc_id, path_str, chunks)
                except Exception as gex:
                    print(f"WARN: graph update failed for {p}: {gex}")
                try:
                    upsert_summaries(args.qdrant_collection, doc_id, chunks)
                except Exception as sex:
                    print(f"WARN: summary update failed for {p}: {sex}")

                # Incremental changed-only check before doing embeddings
                if args.changed_only and not args.fts_only:
                    same_hash = qdrant_any_by_filter(
                        args.qdrant_collection,
                        [{"key": "doc_id", "value": doc_id}, {"key": "content_hash", "value": content_hash}],
                    )
                    if same_hash:
                        continue
                    if args.delete_changed:
                        qdrant_delete_by_doc_id(args.qdrant_collection, doc_id)

                ids: List[str] = []
                payloads: List[Dict[str, Any]] = []
                for chunk in chunks:
                    s = int(chunk.get("chunk_start", 0) or 0)
                    e = int(chunk.get("chunk_end", 0) or 0)
                    chunk_uuid = uuid.uuid5(uuid.UUID(doc_id), f"{s}-{e}")
                    ids.append(str(chunk_uuid))
                    payload = {
                        "doc_id": doc_id,
                        "path": path_str,
                        "chunk_start": s,
                        "chunk_end": e,
                        "filename": p.name,
                        "mtime": mtime,
                        "content_hash": content_hash,
                        "page_numbers": chunk.get("pages", []),
                        "section_path": chunk.get("section_path", []),
                        "element_ids": chunk.get("element_ids", []),
                        "bboxes": chunk.get("bboxes", []),
                        "types": chunk.get("types", []),
                        "source_tools": chunk.get("source_tools", []),
                        "table_headers": chunk.get("headers", []),
                        "table_units": chunk.get("units", []),
                        "text": chunk.get("text", ""),
                    }
                    if args.thin_payload:
                        payload["thin_payload"] = True
                        payload.pop("text", None)
                    payloads.append(payload)

                # Vector upsert path
                upserted_vecs = 0
                if not args.fts_only:
                    if args.embed_robust:
                        n = len(chunks)
                        win = max(1, int(args.embed_window_size))
                        for start in range(0, n, win):
                            end = min(start + win, n)
                            subset = chunks[start:end]
                            texts_w = [c.get("text", "") for c in subset]
                            vecs_w = embed_texts_robust(
                                args.ollama_url,
                                args.ollama_model,
                                texts_w,
                                timeout=args.ollama_timeout,
                                normalize=(args.metric == "cosine"),
                                num_threads=args.ollama_threads,
                                keep_alive=args.ollama_keepalive,
                                max_retries=2,
                            )
                            # Filter successful ones
                            ok_indices = [i for i, v in enumerate(vecs_w) if v is not None]
                            if ok_indices:
                                sel_ids = [ids[start + i] for i in ok_indices]
                                sel_payloads = [payloads[start + i] for i in ok_indices]
                                sel_vecs = [vecs_w[i] for i in ok_indices]
                                upsert_qdrant(args.qdrant_collection, sel_vecs, sel_payloads, sel_ids)
                                upserted_vecs += len(sel_ids)
                    else:
                        vecs = embed_texts(
                            args.ollama_url,
                            args.ollama_model,
                            [c.get("text", "") for c in chunks],
                            batch_size=args.batch_size,
                            timeout=args.ollama_timeout,
                            normalize=(args.metric == "cosine"),
                            parallel=args.parallel,
                            num_threads=args.ollama_threads,
                            keep_alive=args.ollama_keepalive,
                            force_per_item=args.ollama_per_item,
                        )
                        upsert_qdrant(args.qdrant_collection, vecs, payloads, ids)
                        upserted_vecs = len(ids)

                # Also upsert into local FTS index (lexical)
                if not args.no_fts and fts_writer is not None:
                    try:
                        fts_writer.delete_doc(doc_id)
                        rows = []
                        for i, chunk in enumerate(chunks):
                            s = int(chunk.get("chunk_start", 0) or 0)
                            e = int(chunk.get("chunk_end", 0) or 0)
                            t = chunk.get("text", "")
                            pages = chunk.get("pages", [])
                            section_path = chunk.get("section_path", [])
                            element_ids = chunk.get("element_ids", [])
                            bboxes = chunk.get("bboxes", [])
                            types = chunk.get("types", [])
                            source_tools = chunk.get("source_tools", [])
                            table_headers = chunk.get("headers", [])
                            table_units = chunk.get("units", [])
                            rows.append({
                                "text": t,
                                "chunk_id": ids[i],
                                "doc_id": doc_id,
                                "path": path_str,
                                "filename": p.name,
                                "chunk_start": s,
                                "chunk_end": e,
                                "mtime": mtime,
                                "page_numbers": ",".join(str(n) for n in pages) if pages else "",
                                "pages": json.dumps(pages, ensure_ascii=False),
                                "section_path": json.dumps(section_path, ensure_ascii=False) if section_path else "",
                                "element_ids": json.dumps(element_ids, ensure_ascii=False) if element_ids else "",
                                "bboxes": json.dumps(bboxes, ensure_ascii=False) if bboxes else "",
                                "types": json.dumps(types, ensure_ascii=False) if types else "",
                                "source_tools": json.dumps(source_tools, ensure_ascii=False) if source_tools else "",
                                "table_headers": json.dumps(table_headers, ensure_ascii=False) if table_headers else "",
                                "table_units": json.dumps(table_units, ensure_ascii=False) if table_units else "",
                            })
                        fts_writer.upsert_many(rows)
                    except Exception as fex:
                        print(f"WARN: FTS upsert failed for {p}: {fex}")

                processed_docs += 1
                processed_chunks += (len(ids) if args.fts_only else upserted_vecs)

                # Opportunistic GC to limit memory growth on very long runs
                if processed_docs % 10 == 0:
                    gc.collect()

                # Respect file-queue limit
                if args.max_docs_per_run and processed_docs >= args.max_docs_per_run:
                    break

            except Exception as ex:
                print(f"ERR {p}: {ex}")
                errors += 1
                # Continue trying other files
                continue

    finally:
        if fts_writer is not None:
            try:
                fts_writer.close()
            except Exception:
                pass

    # Print concise summary that scripts can parse
    print(f"SUMMARY processed_docs={processed_docs} processed_chunks={processed_chunks} errors={errors}")


if __name__ == "__main__":
    main()
