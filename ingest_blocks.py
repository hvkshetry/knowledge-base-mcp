import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import warnings

warnings.filterwarnings("ignore", message=r".*SwigPy.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*swigvarlink.*", category=DeprecationWarning)

os.environ.setdefault("HF_HOME", str(Path(".cache") / "hf"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

from structured_chunk import detect_blocks, _expand_table_block


# Docling-only extraction - no routing or triage needed


@dataclass
class Block:
    doc_id: str
    path: str
    page: int
    type: str
    text: str
    section_path: List[str]
    element_id: str
    bbox: Optional[List[float]]
    headers: Optional[List[str]]
    units: Optional[Any]
    span: Optional[List[int]]
    source_tool: str


def _init_counters() -> Dict[str, int]:
    return defaultdict(int)


DOC_CONVERTER: Optional[Any] = None  # Type: DocumentConverter when loaded


def _get_docling_converter():
    """Lazy-load Docling DocumentConverter to avoid slow startup."""
    global DOC_CONVERTER
    if DOC_CONVERTER is None:
        from docling.document_converter import DocumentConverter
        DOC_CONVERTER = DocumentConverter()
    return DOC_CONVERTER


def _next_element_id(counters: Dict[str, int], prefix: str, page: int) -> str:
    counters[prefix] += 1
    return f"{prefix}_{page}_{counters[prefix]}"


def _update_heading_path(
    heading_path: List[str],
    text: str,
    level: int,
) -> List[str]:
    level = max(1, min(level, 6))
    while len(heading_path) >= level:
        heading_path.pop()
    heading_path.append(text.strip())
    return heading_path


def _infer_heading_level(text: str) -> int:
    match = re.match(r"^(?P<num>(?:\d+\.)*\d+)\s+", text.strip())
    if match:
        return min(6, match.group("num").count(".") + 1)
    if text.isupper():
        return 2
    return 1


def _docling_full_document_to_blocks(
    path: Path,
    doc_id: str,
) -> Tuple[List[Block], Dict[str, Any]]:
    """
    Process entire PDF with Docling in a single call.
    Returns all blocks and metadata for the document.
    """
    converter = _get_docling_converter()

    print(f"Converting full document with Docling: {path}")
    try:
        result = converter.convert(str(path))
        doc_dict = result.document.export_to_dict()
    except Exception as exc:
        print(f"ERROR: Docling failed for {path}: {exc}")
        return [], {"error": str(exc), "total_pages": 0}

    if not doc_dict:
        return [], {"error": "Empty doc_dict", "total_pages": 0}

    # Group all content by page for organized processing
    page_items: Dict[int, Dict[str, List]] = defaultdict(lambda: {"tables": [], "texts": [], "pictures": [], "captions": []})

    # Organize tables by page
    for table in doc_dict.get("tables", []):
        prov = table.get("prov") or []
        if prov:
            page_no = prov[0].get("page_no")
            if page_no:
                page_items[page_no]["tables"].append(table)
                page_items[page_no]["captions"].extend(table.get("captions", []))

    # Organize text items by page
    for item in doc_dict.get("texts", []):
        prov = item.get("prov") or []
        if prov:
            page_no = prov[0].get("page_no")
            if page_no:
                page_items[page_no]["texts"].append(item)

    # Organize pictures by page
    for picture in doc_dict.get("pictures", []):
        prov = picture.get("prov") or []
        if prov:
            page_no = prov[0].get("page_no")
            if page_no:
                page_items[page_no]["pictures"].append(picture)

    # Now process all pages in order
    blocks: List[Block] = []
    heading_state: List[str] = []
    counters = _init_counters()
    total_pages = max(page_items.keys()) if page_items else 0

    for page_num in sorted(page_items.keys()):
        page_data = page_items[page_num]

        # Process tables first
        for table in page_data["tables"]:
            table_id = _next_element_id(counters, "table", page_num)
            cells = table.get("data", {}).get("table_cells", [])
            rows_map: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
            headers_map: Dict[int, str] = {}

            for cell in cells:
                start_row = cell.get("start_row_offset_idx")
                start_col = cell.get("start_col_offset_idx")
                text = (cell.get("text") or "").strip()
                if text:
                    rows_map[start_row][start_col] = cell
                if start_row == 0:
                    headers_map[start_col] = text

            headers = [headers_map[idx] for idx in sorted(headers_map)]

            for row_idx, col_map in sorted(rows_map.items()):
                if row_idx == 0:
                    continue
                parts = []
                units_map = {}
                for col_idx, cell in sorted(col_map.items()):
                    header = headers_map.get(col_idx, f"col_{col_idx}")
                    value = cell.get("text", "").strip()
                    if not value:
                        continue
                    parts.append(f"{header}: {value}")
                    parts_header = re.match(r"^(?P<name>.+?)\s*\((?P<unit>[^)]+)\)$", header)
                    if parts_header:
                        units_map[parts_header.group("name")] = parts_header.group("unit")

                if not parts:
                    continue

                bbox = None
                bboxes = [cell.get("bbox") for cell in col_map.values() if cell.get("bbox")]
                if bboxes:
                    left = min(b["l"] for b in bboxes)
                    top = min(b["t"] for b in bboxes)
                    right = max(b["r"] for b in bboxes)
                    bottom = max(b["b"] for b in bboxes)
                    bbox = [left, top, right, bottom]

                element_id = _next_element_id(counters, "table_row", page_num)
                blocks.append(
                    Block(
                        doc_id=doc_id,
                        path=str(path),
                        page=page_num,
                        type="table_row",
                        text="; ".join(parts),
                        section_path=heading_state.copy(),
                        element_id=element_id,
                        bbox=bbox,
                        headers=headers,
                        units=units_map or None,
                        span=None,
                        source_tool="docling",
                    )
                )

        # Process text items for this page
        for item in page_data["texts"]:
            prov = item.get("prov") or []
            if not prov:
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            bbox_dict = prov[0].get("bbox") or {}
            bbox = [bbox_dict.get(k) for k in ("l", "t", "r", "b")] if bbox_dict else None
            span = prov[0].get("charspan")
            label = item.get("label", "text")

            if label in {"heading", "title"}:
                level = _infer_heading_level(text)
                heading_state = _update_heading_path(heading_state, text, level)
                element_id = _next_element_id(counters, "heading", page_num)
                blocks.append(
                    Block(
                        doc_id=doc_id,
                        path=str(path),
                        page=page_num,
                        type="heading",
                        text=text,
                        section_path=heading_state.copy(),
                        element_id=element_id,
                        bbox=bbox,
                        headers=None,
                        units=None,
                        span=span,
                        source_tool="docling",
                    )
                )
                continue

            block_type = "para"
            if "list" in label:
                block_type = "list"
            element_prefix = block_type if block_type != "para" else "para"
            element_id = _next_element_id(counters, element_prefix, page_num)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page_num,
                    type=block_type,
                    text=text,
                    section_path=heading_state.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=None,
                    units=None,
                    span=span,
                    source_tool="docling",
                )
            )

        # Process captions for this page
        for caption in page_data["captions"]:
            cap_text = (caption.get("text") or "").strip()
            if not cap_text:
                continue
            prov = caption.get("prov") or []
            bbox = None
            if prov:
                b = prov[0].get("bbox") or {}
                bbox = [b.get("l"), b.get("t"), b.get("r"), b.get("b")] if b else None
            element_id = _next_element_id(counters, "caption", page_num)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page_num,
                    type="caption",
                    text=cap_text,
                    section_path=heading_state.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=None,
                    units=None,
                    span=None,
                    source_tool="docling",
                )
            )

        # Process pictures for this page
        for picture in page_data["pictures"]:
            prov = picture.get("prov") or []
            if not prov:
                continue
            bbox_dict = prov[0].get("bbox") or {}
            bbox = [bbox_dict.get("l"), bbox_dict.get("t"), bbox_dict.get("r"), bbox_dict.get("b")] if bbox_dict else None
            element_id = _next_element_id(counters, "figure", page_num)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page_num,
                    type="figure",
                    text="",
                    section_path=heading_state.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=None,
                    units=None,
                    span=None,
                    source_tool="docling",
                )
            )

    metadata = {
        "total_pages": total_pages,
        "total_blocks": len(blocks),
        "extractor": "docling",
    }

    return blocks, metadata


def _serialize_blocks(blocks: Iterable[Block]) -> List[Dict[str, Any]]:
    serialised = []
    for b in blocks:
        serialised.append(
            {
                "doc_id": b.doc_id,
                "path": b.path,
                "page": b.page,
                "type": b.type,
                "text": b.text,
                "section_path": b.section_path,
                "element_id": b.element_id,
                "bbox": b.bbox,
                "headers": b.headers,
                "units": b.units,
                "span": b.span,
                "source_tool": b.source_tool,
            }
        )
    return serialised


def _deserialize_blocks(items: Iterable[Dict[str, Any]]) -> List[Block]:
    blocks: List[Block] = []
    for it in items:
        blocks.append(
            Block(
                doc_id=it["doc_id"],
                path=it["path"],
                page=it["page"],
                type=it["type"],
                text=it["text"],
                section_path=it.get("section_path") or [],
                element_id=it.get("element_id", ""),
                bbox=it.get("bbox"),
                headers=it.get("headers"),
                units=it.get("units"),
                span=it.get("span"),
                source_tool=it.get("source_tool", "unknown"),
            )
        )
    return blocks


def extract_document_blocks(
    path: Path,
    doc_id: str,
    plan_override: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Block], Dict[str, Any]]:
    """
    Extract blocks from PDF using Docling only - no routing, no triage.
    Breaking change: always uses Docling for full document processing.
    """
    return _docling_full_document_to_blocks(path, doc_id)


def chunk_blocks(
    blocks: List[Block],
    max_chars: int,
    overlap_sentences: int = 1,
    profile: str = "heading_based",
) -> Tuple[List[Dict[str, Any]], str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    profile = (profile or "heading_based").lower()
    respect_headings = profile != "fixed_window"
    sentence_overlap = overlap_sentences if respect_headings else 0

    chunks: List[Dict[str, Any]] = []
    buffer: List[Block] = []
    chunk_cursor = 0
    raw_text_parts: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer, chunk_cursor
        if not buffer:
            return
        text = "\n\n".join(b.text for b in buffer if b.text)
        if not text.strip():
            buffer = []
            return
        pages = sorted({b.page for b in buffer})
        section_path = buffer[-1].section_path if buffer else []
        element_ids = [b.element_id for b in buffer if b.element_id]
        bboxes = [b.bbox for b in buffer]
        types = [b.type for b in buffer]
        source_tools = list({b.source_tool for b in buffer})
        headers: List[str] = []
        units: List[str] = []
        for b in buffer:
            if b.headers:
                headers.extend([str(h).strip() for h in b.headers if h])
            if b.units:
                if isinstance(b.units, dict):
                    units.extend(str(v).strip() for v in b.units.values() if v)
                elif isinstance(b.units, list):
                    units.extend(str(u).strip() for u in b.units if u)
                else:
                    units.append(str(b.units).strip())
        chunk = {
            "text": text,
            "pages": pages,
            "section_path": section_path,
            "element_ids": element_ids,
            "bboxes": bboxes,
            "types": types,
            "source_tools": source_tools,
            "headers": headers,
            "table_headers": headers,
            "units": units,
            "table_units": units,
            "chunk_start": chunk_cursor,
            "chunk_end": chunk_cursor + len(text),
            "doc_id": buffer[0].doc_id,
            "path": buffer[0].path,
            "profile": profile,
        }
        chunks.append(chunk)
        chunk_cursor += len(text)
        if sentence_overlap > 0:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            tail = " ".join(sentences[-sentence_overlap:]).strip()
            buffer = [
                Block(
                    doc_id=buffer[-1].doc_id,
                    path=buffer[-1].path,
                    page=buffer[-1].page,
                    type="para",
                    text=tail,
                    section_path=section_path,
                    element_id="",
                    bbox=None,
                    headers=None,
                    units=None,
                    span=None,
                    source_tool=buffer[-1].source_tool,
                )
            ] if tail else []
        else:
            buffer = []

    for block in blocks:
        raw_text_parts.append(block.text)
        if profile == "table_row" and block.type == "table_row":
            flush_buffer()
            text = block.text or ""
            if not text.strip():
                continue
            row_headers = [str(h).strip() for h in (block.headers or []) if h]
            row_units: List[str] = []
            if block.units:
                if isinstance(block.units, dict):
                    row_units.extend(str(v).strip() for v in block.units.values() if v)
                elif isinstance(block.units, list):
                    row_units.extend(str(u).strip() for u in block.units if u)
                else:
                    row_units.append(str(block.units).strip())
            chunk = {
                "text": text,
                "pages": [block.page],
                "section_path": block.section_path,
                "element_ids": [block.element_id] if block.element_id else [],
                "bboxes": [block.bbox] if block.bbox else [],
                "types": [block.type],
                "source_tools": [block.source_tool] if block.source_tool else [],
                "headers": row_headers,
                "table_headers": row_headers,
                "units": row_units,
                "table_units": row_units,
                "chunk_start": chunk_cursor,
                "chunk_end": chunk_cursor + len(text),
                "doc_id": block.doc_id,
                "path": block.path,
                "profile": profile,
            }
            chunks.append(chunk)
            chunk_cursor += len(text)
            continue

        if respect_headings and block.type == "heading":
            flush_buffer()
            buffer = [block]
            flush_buffer()
            buffer = []
            continue

        buffer.append(block)
        total_chars = sum(len(b.text) for b in buffer)
        if total_chars >= max_chars:
            flush_buffer()

    flush_buffer()
    raw_text = "\n".join(raw_text_parts)
    return chunks, raw_text
