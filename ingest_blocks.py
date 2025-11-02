import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz

os.environ.setdefault("HF_HOME", str(Path(".cache") / "hf"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

from docling.document_converter import DocumentConverter

from structured_chunk import detect_blocks, _expand_table_block


TABLE_TOKEN = re.compile(r"\bTable\s+\d+(\-\d+)?\b", re.IGNORECASE)
FIGURE_TOKEN = re.compile(r"\b(Fig\.?|Figure)\s+\d+(\-\d+)?\b", re.IGNORECASE)


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


def _cache_key(path: Path, page: int, route: str) -> Path:
    root = Path(os.getenv("CACHE_DIR", ".ingest_cache"))
    key = hashlib.sha1(f"{path.resolve()}::{page}::{route}".encode("utf-8")).hexdigest()
    return root / f"{key}.json"


def triage_pdf(path: Path) -> Dict[str, Any]:
    doc = fitz.open(path)
    pages = []
    for page_index, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", flags=fitz.TEXTFLAGS_TEXT)
        text_chars = sum(len(b[4]) for b in blocks if isinstance(b[4], str))
        images = len(page.get_images(full=True))
        drawings = page.get_drawings()
        vector_lines = sum(
            1
            for drawing in drawings
            for item in drawing.get("items", [])
            if item and item[0] == "l"
        )

        centers = [
            (b[0] + b[2]) / 2
            for b in blocks
            if isinstance(b[4], str) and b[4].strip()
        ]
        multicolumn_score = 0.0
        if len(centers) >= 8:
            centers.sort()
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            if gaps:
                mean_gap = sum(gaps) / len(gaps)
                if mean_gap:
                    multicolumn_score = max(gaps) / (mean_gap + 1e-6)

        sample_text = " ".join(
            (b[4] for b in blocks if isinstance(b[4], str))
        )[:4000]
        has_table_token = bool(TABLE_TOKEN.search(sample_text))
        has_figure_token = bool(FIGURE_TOKEN.search(sample_text))
        is_scan = text_chars < 200 and images > 0
        is_tabley = vector_lines > 120 or has_table_token
        is_multicol = multicolumn_score > float(os.getenv("MULTICOL_GAP_FACTOR", "4.0"))

        route = "markitdown"
        if (
            is_scan
            or is_tabley
            or is_multicol
            or has_figure_token
        ):
            route = "docling"

        page_text = page.get_text("text") or ""
        pages.append(
            {
                "page": page_index,
                "route": route,
                "text_chars": text_chars,
                "images": images,
                "vector_lines": vector_lines,
                "multicolumn_score": multicolumn_score,
                "has_table_token": has_table_token,
                "has_figure_token": has_figure_token,
                "page_text": page_text,
            }
        )
    doc.close()

    heavy_pages = sum(1 for p in pages if p["route"] == "docling")
    fraction = heavy_pages / max(len(pages), 1)
    heavy_threshold = float(os.getenv("ROUTE_HEAVY_FRACTION", "0.30"))
    small_doc_threshold = int(os.getenv("SMALL_DOC_DOCLING", "8"))
    if len(pages) <= small_doc_threshold or fraction >= heavy_threshold:
        for p in pages:
            p["route"] = "docling"

    return {"pages": pages}


def _init_counters() -> Dict[str, int]:
    return defaultdict(int)


DOC_CONVERTER: Optional[DocumentConverter] = None
DOC_CONVERTER_TIMEOUT = int(os.getenv("DOCLING_TIMEOUT", "45"))


def _get_docling_converter() -> DocumentConverter:
    global DOC_CONVERTER
    if DOC_CONVERTER is None:
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


def _blocks_from_text(
    doc_id: str,
    path: Path,
    page: int,
    text: str,
    heading_path: List[str],
    counters: Dict[str, int],
    source_tool: str,
) -> Tuple[List[Block], List[str]]:
    result: List[Block] = []
    local_heading = heading_path.copy()
    text_blocks = detect_blocks(text)
    for tb in text_blocks:
        block_text = tb.text.strip()
        if not block_text:
            continue
        span = [tb.start, tb.end]
        if tb.block_type == "heading":
            level = tb.heading_level or _infer_heading_level(block_text)
            local_heading = _update_heading_path(local_heading, block_text, level)
            element_id = _next_element_id(counters, "heading", page)
            result.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page,
                    type="heading",
                    text=block_text,
                    section_path=local_heading.copy(),
                    element_id=element_id,
                    bbox=None,
                    headers=None,
                    units=None,
                    span=span,
                    source_tool=source_tool,
                )
            )
            continue

        if tb.block_type == "table":
            table_rows = _expand_table_block(tb, local_heading.copy(), counters["table"])
            if table_rows:
                counters["table"] += 1
                for row_idx, row in enumerate(table_rows):
                    element_id = _next_element_id(counters, "table_row", page)
                    result.append(
                        Block(
                            doc_id=doc_id,
                            path=str(path),
                            page=page,
                            type="table_row",
                            text=row["text"],
                            section_path=local_heading.copy(),
                            element_id=element_id,
                            bbox=None,
                            headers=row.get("meta", {}).get("table_headers"),
                            units=row.get("meta", {}).get("table_units"),
                            span=[row.get("start"), row.get("end")],
                            source_tool=source_tool,
                        )
                    )
                continue

        element_id = _next_element_id(counters, tb.block_type, page)
        block_type = tb.block_type
        if block_type == "paragraph":
            block_type = "para"
        result.append(
            Block(
                doc_id=doc_id,
                path=str(path),
                page=page,
                type=block_type,
                text=block_text,
                section_path=local_heading.copy(),
                element_id=element_id,
                bbox=None,
                headers=None,
                units=None,
                span=span,
                source_tool=source_tool,
            )
        )
    return result, local_heading


def _docling_page_to_blocks(
    path: Path,
    doc_id: str,
    page: int,
    heading_path: List[str],
    counters: Dict[str, int],
) -> Tuple[List[Block], List[str]]:
    converter = _get_docling_converter()
    tmp_pdf = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = Path(tmp.name)
        original = fitz.open(path)
        single = fitz.open()
        single.insert_pdf(original, from_page=page - 1, to_page=page - 1)
        single.save(tmp_pdf)
        single.close()
        original.close()

        def _convert() -> Dict[str, Any]:
            result = converter.convert(str(tmp_pdf))
            return result.document.export_to_dict()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_convert)
            try:
                doc_dict = future.result(timeout=DOC_CONVERTER_TIMEOUT)
            except Exception:
                doc_dict = None
    finally:
        if tmp_pdf:
            try:
                tmp_pdf.unlink(missing_ok=True)
            except Exception:
                pass

    if not doc_dict:
        return [], heading_path

    blocks: List[Block] = []
    local_heading = heading_path.copy()
    tables = doc_dict.get("tables", [])
    pictures = doc_dict.get("pictures", [])
    captions = []

    for table in tables:
        prov = table.get("prov") or []
        if not prov or prov[0].get("page_no") != page:
            continue
        table_id = _next_element_id(counters, "table", page)
        captions.extend(table.get("captions", []))
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
            element_id = _next_element_id(counters, "table_row", page)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page,
                    type="table_row",
                    text="; ".join(parts),
                    section_path=heading_path.copy(),
                    element_id=element_id,
                    bbox=bbox,
                    headers=headers,
                    units=units_map or None,
                    span=None,
                    source_tool="docling",
                )
            )

    for item in doc_dict.get("texts", []):
        prov = item.get("prov") or []
        if not prov:
            continue
        page_no = prov[0].get("page_no")
        if page_no != page:
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
            local_heading = _update_heading_path(local_heading, text, level)
            element_id = _next_element_id(counters, "heading", page)
            blocks.append(
                Block(
                    doc_id=doc_id,
                    path=str(path),
                    page=page,
                    type="heading",
                    text=text,
                    section_path=local_heading.copy(),
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
        element_id = _next_element_id(counters, element_prefix, page)
        blocks.append(
            Block(
                doc_id=doc_id,
                path=str(path),
                page=page,
                type=block_type,
                text=text,
                section_path=local_heading.copy(),
                element_id=element_id,
                bbox=bbox,
                headers=None,
                units=None,
                span=span,
                source_tool="docling",
            )
        )

    for caption in captions:
        cap_text = (caption.get("text") or "").strip()
        if not cap_text:
            continue
        prov = caption.get("prov") or []
        bbox = None
        if prov:
            b = prov[0].get("bbox") or {}
            bbox = [b.get("l"), b.get("t"), b.get("r"), b.get("b")] if b else None
        element_id = _next_element_id(counters, "caption", page)
        blocks.append(
            Block(
                doc_id=doc_id,
                path=str(path),
                page=page,
                type="caption",
                text=cap_text,
                section_path=heading_path.copy(),
                element_id=element_id,
                bbox=bbox,
                headers=None,
                units=None,
                span=None,
                source_tool="docling",
            )
        )

    for picture in pictures:
        prov = picture.get("prov") or []
        if not prov or prov[0].get("page_no") != page:
            continue
        bbox_dict = prov[0].get("bbox") or {}
        bbox = [bbox_dict.get("l"), bbox_dict.get("t"), bbox_dict.get("r"), bbox_dict.get("b")] if bbox_dict else None
        element_id = _next_element_id(counters, "figure", page)
        blocks.append(
            Block(
                doc_id=doc_id,
                path=str(path),
                page=page,
                type="figure",
                text="",
                section_path=heading_path.copy(),
                element_id=element_id,
                bbox=bbox,
                headers=None,
                units=None,
                span=None,
                source_tool="docling",
            )
        )

    return blocks, local_heading


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


def extract_document_blocks(path: Path, doc_id: str) -> Tuple[List[Block], Dict[str, Any]]:
    triage = triage_pdf(path)
    blocks: List[Block] = []
    heading_state: List[str] = []
    counters = _init_counters()
    cache_root = Path(os.getenv("CACHE_DIR", ".ingest_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    for page_info in triage["pages"]:
        page_num = page_info["page"]
        route = page_info["route"]
        cache_file = _cache_key(path, page_num, route)
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            page_blocks = _deserialize_blocks(cached)
            blocks.extend(page_blocks)
            if page_blocks:
                heading_state = page_blocks[-1].section_path.copy()
            continue

        if route == "docling":
            page_blocks, heading_state = _docling_page_to_blocks(path, doc_id, page_num, heading_state, counters)
            if not page_blocks:
                route = "markitdown"
        if route != "docling":
            page_text = page_info.get("page_text") or ""
            page_blocks, heading_state = _blocks_from_text(
                doc_id=doc_id,
                path=path,
                page=page_num,
                text=page_text,
                heading_path=heading_state,
                counters=counters,
                source_tool="markitdown",
            )

        cache_file.write_text(
            json.dumps(_serialize_blocks(page_blocks), ensure_ascii=False),
            encoding="utf-8",
        )
        blocks.extend(page_blocks)
    return blocks, triage


def chunk_blocks(
    blocks: List[Block],
    max_chars: int,
    overlap_sentences: int = 1,
) -> Tuple[List[Dict[str, Any]], str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
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
        headers = [
            b.headers for b in buffer if b.headers
        ]
        units = [
            b.units for b in buffer if b.units
        ]
        chunk = {
            "text": text,
            "pages": pages,
            "section_path": section_path,
            "element_ids": element_ids,
            "bboxes": bboxes,
            "types": types,
            "source_tools": source_tools,
            "headers": headers,
            "units": units,
            "chunk_start": chunk_cursor,
            "chunk_end": chunk_cursor + len(text),
            "doc_id": buffer[0].doc_id,
            "path": buffer[0].path,
        }
        chunks.append(chunk)
        chunk_cursor += len(text)
        if overlap_sentences > 0:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            tail = " ".join(sentences[-overlap_sentences:]).strip()
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
        if block.type == "heading":
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
