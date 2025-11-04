import argparse
import os
import pathlib
import uuid
import time
import fnmatch
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import gc
import re
import json

import requests
import numpy as np
from tqdm import tqdm

from ingest_blocks import extract_document_blocks, chunk_blocks
from graph_builder import update_graph
from summary_index import upsert_summaries
from metadata_schema import generate_metadata
from sparse_expansion import SparseExpander
from ingest_core import (
    DEFAULT_FTS_DB,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_QDRANT_API_KEY,
    DEFAULT_QDRANT_METRIC,
    DEFAULT_QDRANT_URL,
    INGEST_MODEL_VERSION,
    INGEST_PROMPT_SHA,
    MAX_METADATA_BYTES,
    embed_texts,
    embed_texts_robust,
    ensure_qdrant_collection,
    get_qdrant_client,
    qdrant_any_by_filter,
    qdrant_delete_by_doc_id,
    upsert_qdrant,
)


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

PLAN_DIR = pathlib.Path(os.getenv("INGEST_PLAN_DIR", "data/ingest_plans"))


def file_uri(p: pathlib.Path) -> str:
    return p.resolve().as_uri()


def load_ingest_plan(doc_id: str) -> Optional[Dict[str, Any]]:
    try:
        path = PLAN_DIR / f"{doc_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def save_ingest_plan(doc_id: str, data: Dict[str, Any]) -> None:
    try:
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        path = PLAN_DIR / f"{doc_id}.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        print(f"WARN: failed to persist ingest plan for {doc_id}: {exc}")


def sanitize_triage_for_plan(triage: Dict[str, Any]) -> Dict[str, Any]:
    pages_out: List[Dict[str, Any]] = []
    for entry in triage.get("pages", []):
        if not isinstance(entry, dict):
            continue
        page_num = entry.get("page")
        try:
            page_int = int(page_num)
        except Exception:
            page_int = None
        pages_out.append(
            {
                "page": page_int,
                "route": entry.get("route"),
                "original_route": entry.get("original_route"),
                "text_chars": entry.get("text_chars"),
                "images": entry.get("images"),
                "vector_lines": entry.get("vector_lines"),
                "multicolumn_score": entry.get("multicolumn_score"),
                "has_table_token": entry.get("has_table_token"),
                "has_figure_token": entry.get("has_figure_token"),
                "table_tokens": entry.get("table_tokens"),
                "text_density": entry.get("text_density"),
                "confidence": entry.get("confidence"),
                "sample_text": entry.get("sample_text"),
            }
        )
    result = {"pages": pages_out}
    stats = triage.get("confidence_stats")
    if isinstance(stats, dict):
        result["confidence_stats"] = stats
    return result


def compute_plan_hash(payload: Dict[str, Any]) -> str:
    material = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def get_embedding_dim(ollama_url: str, model: str) -> int:
    try:
        vecs = embed_texts(ollama_url, model, ["probe"], batch_size=1, normalize=False)
        if not vecs:
            raise RuntimeError("empty embedding response")
        return len(vecs[0])
    except Exception as exc:
        raise SystemExit(f"Failed to probe Ollama at {ollama_url} for model {model}: {exc}")


def select_chunk_profile(blocks: List[Any]) -> str:
    if not blocks:
        return "fixed_window"
    table_rows = sum(1 for b in blocks if getattr(b, "type", "") == "table_row")
    if table_rows:
        return "table_row"
    step_like = 0
    for b in blocks:
        btype = getattr(b, "type", "")
        text = getattr(b, "text", "") or ""
        text = text.strip()
        if btype == "list":
            step_like += 1
        elif re.match(r"^\d+(\.\d+)*\s", text):
            step_like += 1
    if step_like and step_like >= max(3, len(blocks) // 4):
        return "procedure_block"
    return "heading_based"


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
    if extractor == "pymupdf":
        return extract_pdf_pymupdf
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory with source documents")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="snowflake-arctic-embed:xs")
    # NOTE: Extractor removed - ingest_blocks.py now uses Docling-only for all documents
    ap.add_argument("--sparse-expander", choices=["none", "basic", "splade"], default=None, help="Optional sparse expander for SPLADE/uniCOIL-style term weights")

    # Metric/normalization
    ap.add_argument("--metric", choices=["cosine", "dot", "euclid"], default="cosine")

    # Chunking
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--overlap", type=int, default=150)

    # Embedding batch (increased default to 128 for better throughput)
    ap.add_argument("--batch-size", type=int, default=128)
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
        raw_thin = os.getenv("THIN_PAYLOAD")
        if raw_thin is None:
            raw_thin = os.getenv("THIN_VECTOR_PAYLOAD")
        if raw_thin:
            args.thin_payload = raw_thin.strip().lower() in {"1", "true", "yes"}
    if args.thin_payload:
        print("Thin payload mode enabled: vector payloads will omit raw text.")

    root = pathlib.Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root does not exist or is not a directory: {root}")

    # Normalize FTS DB path to absolute to avoid issues if CWD changes in libraries
    try:
        args.fts_db = os.path.abspath(args.fts_db)
    except Exception:
        pass

    include_exts = {e.strip().lower() for e in args.ext.split(',') if e.strip()} if args.ext else None
    skip_patterns = list(DEFAULT_SKIP_PATTERNS)
    if args.skip:
        skip_patterns += [s.strip() for s in args.skip.split(',') if s.strip()]
    include_patterns = [s.strip() for s in args.include.split(',') if s.strip()] if args.include else []

    # Initialize Qdrant collection if we are doing vector ingest
    qdrant_client = get_qdrant_client(args.qdrant_url, args.qdrant_api_key, timeout=args.qdrant_timeout)
    if not args.fts_only:
        embedding_dim = get_embedding_dim(args.ollama_url, args.ollama_model)
        ensure_qdrant_collection(qdrant_client, args.qdrant_collection, embedding_dim, args.metric)

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

    # Configure sparse expander shared with ingestion + query
    sparse_method = args.sparse_expander
    if sparse_method is None:
        sparse_method = os.getenv("SPARSE_EXPANDER", "none")
    sparse_method = (sparse_method or "none").strip().lower()
    sparse_expander = SparseExpander(sparse_method)
    if sparse_expander.enabled:
        print(f"Sparse expander enabled: {sparse_expander.method}")
    else:
        sparse_expander = None

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
                    if qdrant_any_by_filter(qdrant_client, args.qdrant_collection, [{"key": "doc_id", "value": doc_id}]):
                        continue

                # Extract with Docling (content filtering moved to post-extraction)
                existing_plan = load_ingest_plan(doc_id)
                plan_override = existing_plan.get("triage") if isinstance(existing_plan, dict) else None
                triage_blocks, triage_info = extract_document_blocks(p, doc_id, plan_override=plan_override)

                # Apply content filters after extraction
                if not triage_blocks:
                    continue
                text = "\n\n".join(b.text for b in triage_blocks if b.text)
                if not text.strip():
                    continue
                if args.min_words and alpha_word_count(text) < args.min_words:
                    continue
                overlap_sentences = max(1, args.overlap // 80 if args.overlap else 1)
                if plan_override and isinstance(plan_override, dict):
                    # Reuse overlap settings from existing plan if provided
                    params = existing_plan.get("chunk_params") if isinstance(existing_plan, dict) else None
                    if isinstance(params, dict):
                        overlap_sentences = int(params.get("overlap_sentences", overlap_sentences) or overlap_sentences)
                chunk_profile = (
                    existing_plan.get("chunk_profile")
                    if isinstance(existing_plan, dict) and existing_plan.get("chunk_profile")
                    else select_chunk_profile(triage_blocks)
                )
                chunks, raw_text = chunk_blocks(
                    triage_blocks,
                    args.max_chars,
                    overlap_sentences=overlap_sentences,
                    profile=chunk_profile,
                )
                if not raw_text.strip():
                    raw_text = text
                content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
                mtime = int(p.stat().st_mtime)

                plan_payload = {
                    "doc_id": doc_id,
                    "path": path_str,
                    "collection": args.qdrant_collection,
                    "content_hash": content_hash,
                    "chunk_profile": chunk_profile,
                    "chunk_params": {
                        "max_chars": args.max_chars,
                        "overlap_sentences": overlap_sentences,
                    },
                    "triage": sanitize_triage_for_plan(triage_info),
                    "model_version": INGEST_MODEL_VERSION,
                    "prompt_sha": INGEST_PROMPT_SHA,
                }
                if existing_plan and isinstance(existing_plan.get("client_orchestration"), dict):
                    plan_payload["client_orchestration"] = existing_plan["client_orchestration"]
                doc_metadata, metadata_rejects = generate_metadata(raw_text, chunks)
                metadata_bytes = len(json.dumps(doc_metadata, ensure_ascii=False).encode("utf-8")) if doc_metadata else 0
                if doc_metadata and metadata_bytes > MAX_METADATA_BYTES:
                    metadata_rejects.append(f"metadata_bytes_exceeded:{metadata_bytes}>{MAX_METADATA_BYTES}")
                    print(f"WARN: metadata exceeds MAX_METADATA_BYTES for {p} ({metadata_bytes} bytes)")
                    doc_metadata = {}
                if doc_metadata:
                    plan_payload["doc_metadata"] = doc_metadata
                if metadata_rejects:
                    plan_payload["metadata_rejects"] = metadata_rejects
                    print(f"WARN: metadata validation issues for {p}: {metadata_rejects}")
                plan_payload["metadata_calls"] = 1 if doc_metadata else 0
                plan_hash = compute_plan_hash(plan_payload)
                plan_payload["plan_hash"] = plan_hash
                if not existing_plan or existing_plan.get("plan_hash") != plan_hash:
                    save_ingest_plan(doc_id, plan_payload)
                for chunk in chunks:
                    chunk["plan_hash"] = plan_hash
                    if doc_metadata:
                        chunk["doc_metadata"] = doc_metadata
                if sparse_expander:
                    for chunk in chunks:
                        terms = sparse_expander.encode(chunk.get("text", ""))
                        if terms:
                            chunk["sparse_terms"] = [{"term": term, "weight": float(weight)} for term, weight in terms]
                        else:
                            chunk["sparse_terms"] = []

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
                        qdrant_client,
                        args.qdrant_collection,
                        [{"key": "doc_id", "value": doc_id}, {"key": "content_hash", "value": content_hash}],
                    )
                    if same_hash:
                        continue
                    if args.delete_changed:
                        qdrant_delete_by_doc_id(qdrant_client, args.qdrant_collection, doc_id)

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
                        "chunk_profile": chunk.get("profile"),
                        "plan_hash": plan_hash,
                        "model_version": plan_payload.get("model_version"),
                        "prompt_sha": plan_payload.get("prompt_sha"),
                        "doc_metadata": doc_metadata,
                        "text": chunk.get("text", ""),
                    }
                    if sparse_expander and chunk.get("sparse_terms"):
                        payload["sparse_terms"] = chunk.get("sparse_terms")
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
                                upsert_qdrant(qdrant_client, args.qdrant_collection, sel_vecs, sel_payloads, sel_ids)
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
                        upsert_qdrant(qdrant_client, args.qdrant_collection, vecs, payloads, ids)
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
                            profile_tag = chunk.get("profile") or ""
                            doc_metadata_payload = chunk.get("doc_metadata") or {}
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
                                "chunk_profile": profile_tag,
                                "plan_hash": plan_hash,
                                "model_version": plan_payload.get("model_version", ""),
                                "prompt_sha": plan_payload.get("prompt_sha", ""),
                                "doc_metadata": json.dumps(doc_metadata_payload, ensure_ascii=False) if doc_metadata_payload else "",
                                "sparse_terms": chunk.get("sparse_terms", []),
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
