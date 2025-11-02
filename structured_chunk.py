import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TextBlock:
    block_type: str
    text: str
    start: int
    end: int
    lines: List[Tuple[str, int, int]]
    heading_level: Optional[int] = None
    heading_text: Optional[str] = None


HEADING_MD_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+)$")
HEADING_NUM_RE = re.compile(r"^(?P<num>(?:\d+\.)*\d+)\s+(?P<title>.+)$")
LIST_RE = re.compile(r"^(\*|-|\+|\d+[.)])\s+")
TABLE_ALIGNMENT_RE = re.compile(r"^\s*[:|\- ]+\s*$")


def _split_lines(text: str) -> Iterable[Tuple[str, int, int]]:
    """Yield (line_without_newline, start_offset, end_offset)."""
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        length = len(raw_line)
        stripped = raw_line.rstrip("\n")
        start = offset
        end = offset + len(stripped)
        yield stripped, start, end
        offset += length
    if text and not text.endswith("\n"):
        yield "", len(text), len(text)


def _classify_block(lines: List[Tuple[str, int, int]]) -> TextBlock:
    start = lines[0][1]
    end = lines[-1][2]
    raw_lines = [ln for ln, _, _ in lines]
    joined = "\n".join(raw_lines).strip()
    block = TextBlock(block_type="paragraph", text=joined, start=start, end=end, lines=lines)
    first = raw_lines[0].strip()
    if not first:
        return block

    md_match = HEADING_MD_RE.match(first)
    if md_match:
        level = len(md_match.group(1))
        title = md_match.group("title").strip()
        block.block_type = "heading"
        block.heading_level = level
        block.heading_text = title
        block.text = title
        return block

    num_match = HEADING_NUM_RE.match(first)
    if num_match and len(raw_lines) == 1:
        num = num_match.group("num")
        title = num_match.group("title").strip()
        level = min(6, num.count(".") + 1)
        block.block_type = "heading"
        block.heading_level = level
        block.heading_text = title
        block.text = f"{num} {title}"
        return block

    if all(LIST_RE.match(line.strip() or "") for line in raw_lines if line.strip()):
        block.block_type = "list"
        return block

    if _looks_like_table(raw_lines):
        block.block_type = "table"
        return block

    return block


def _looks_like_table(lines: List[str]) -> bool:
    if len(lines) < 2:
        return False
    pipe_density = sum(1 for ln in lines if "|" in ln)
    tab_density = sum(1 for ln in lines if "\t" in ln)
    if pipe_density >= max(2, len(lines) // 2):
        return True
    if tab_density >= max(2, len(lines) // 2):
        return True
    return False


def _parse_markdown_table(block: TextBlock) -> Optional[Tuple[List[str], List[Tuple[List[str], int, int]]]]:
    rows: List[Tuple[List[str], int, int]] = []
    raw_rows: List[Tuple[str, int, int]] = []
    for line, start, end in block.lines:
        stripped = line.strip()
        if not stripped:
            continue
        raw_rows.append((stripped, start, end))
    if not raw_rows:
        return None
    if "|" not in raw_rows[0][0]:
        return None

    def split_row(text: str) -> List[str]:
        return [cell.strip() for cell in text.strip("|").split("|")]

    header_cells = split_row(raw_rows[0][0])
    data_index = 1
    if len(raw_rows) > 1 and TABLE_ALIGNMENT_RE.match(raw_rows[1][0]):
        data_index = 2
    if len(header_cells) < 2:
        return None
    data_rows = raw_rows[data_index:]
    if not data_rows:
        return None
    for row_text, start, end in data_rows:
        if "|" not in row_text:
            continue
        cells = split_row(row_text)
        if len(cells) != len(header_cells):
            continue
        rows.append((cells, start, end))
    if not rows:
        return None
    return header_cells, rows


def detect_blocks(text: str) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    current_lines: List[Tuple[str, int, int]] = []
    for line, start, end in _split_lines(text):
        if line.strip():
            current_lines.append((line, start, end))
            continue
        if current_lines:
            block = _classify_block(current_lines)
            blocks.append(block)
            current_lines = []
    if current_lines:
        block = _classify_block(current_lines)
        blocks.append(block)
    return blocks


def _extract_units(headers: List[str]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    unit_re = re.compile(r"^(?P<name>.+?)\s*\((?P<unit>[^)]+)\)\s*$")
    for header in headers:
        match = unit_re.match(header)
        if match:
            units[match.group("name").strip()] = match.group("unit").strip()
    return units


def _expand_table_block(
    block: TextBlock,
    heading_path: List[str],
    table_counter: int,
) -> List[Dict[str, Any]]:
    parsed = _parse_markdown_table(block)
    if not parsed:
        return []
    headers, rows = parsed
    units_map = _extract_units(headers)
    table_id = f"table-{table_counter}"
    out: List[Dict[str, Any]] = []
    for idx, (cells, start, end) in enumerate(rows):
        parts = []
        for header, value in zip(headers, cells):
            header_name = header
            if header_name in units_map:
                header_name = header_name.split("(")[0].strip()
            parts.append(f"{header_name}: {value}")
        chunk_text = "; ".join(parts).strip()
        if not chunk_text:
            continue
        meta = {
            "chunk_type": "table_row",
            "heading_path": heading_path.copy(),
            "table_id": table_id,
            "table_row_index": idx,
            "table_headers": headers,
        }
        if units_map:
            meta["table_units"] = units_map
        out.append(
            {
                "start": start,
                "end": end,
                "text": chunk_text,
                "meta": meta,
            }
        )
    return out


def generate_chunks(
    text: str,
    max_chars: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    blocks = detect_blocks(text)
    heading_path: List[str] = []
    chunks: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []
    buffer_start: Optional[int] = None
    buffer_end: Optional[int] = None
    table_counter = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_start, buffer_end
        if not buffer:
            return
        chunk_text = "\n\n".join(item["text"] for item in buffer).strip()
        if not chunk_text:
            buffer = []
            buffer_start = None
            buffer_end = None
            return
        chunk_start = buffer_start if buffer_start is not None else buffer[0]["start"]
        chunk_end = buffer_end if buffer_end is not None else buffer[-1]["end"]
        chunk_type = "list" if any(item["block_type"] == "list" for item in buffer) else "paragraph"
        meta = {
            "chunk_type": chunk_type,
            "heading_path": heading_path.copy(),
        }
        chunks.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "text": chunk_text,
                "meta": meta,
            }
        )
        if overlap > 0 and len(chunk_text) > overlap:
            overlap_text = chunk_text[-overlap:]
            buffer = [
                {
                    "text": overlap_text,
                    "start": max(chunk_end - len(overlap_text), chunk_start),
                    "end": chunk_end,
                    "block_type": chunk_type,
                }
            ]
            buffer_start = buffer[0]["start"]
            buffer_end = chunk_end
        else:
            buffer = []
            buffer_start = None
            buffer_end = None

    for block in blocks:
        if block.block_type == "heading":
            flush_buffer()
            level = block.heading_level or 1
            while len(heading_path) >= level:
                heading_path.pop()
            if block.heading_text:
                heading_path.append(block.heading_text)
            continue

        if block.block_type == "table":
            flush_buffer()
            table_counter += 1
            table_chunks = _expand_table_block(block, heading_path, table_counter)
            if table_chunks:
                chunks.extend(table_chunks)
                continue
            # fallback: treat as paragraph

        text_piece = block.text.strip()
        if not text_piece:
            continue
        if buffer_start is None:
            buffer_start = block.start
        buffer_end = block.end
        buffer.append(
            {
                "text": text_piece,
                "start": block.start,
                "end": block.end,
                "block_type": block.block_type,
            }
        )
        total_length = sum(len(item["text"]) + 2 for item in buffer) - 2
        if total_length >= max_chars:
            flush_buffer()

    flush_buffer()
    return chunks
