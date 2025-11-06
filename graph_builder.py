import hashlib
import itertools
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "data/graph.db")

# Global keyphrase extractor cache for lazy loading
_KEYPHRASE_EXTRACTOR = None

ENTITY_MIN_LEN = 4
ENTITY_TYPE_CONCEPT = "concept"
ENTITY_TYPE_UNIT = "unit"
ENTITY_TYPE_MEASUREMENT = "measurement"

UPPER_ENTITY_RE = re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z0-9]{2,})*)\b")
MEASUREMENT_RE = re.compile(
    r"(?P<parameter>[A-Za-z][A-Za-z0-9\s/%°\-\(\)]{2,}?)\s*(?:=|:)\s*(?P<value>-?\d+(?:\.\d+)?(?:\s*[x×]\s*10\^\d+)?)\s*(?P<unit>[A-Za-zμ/%°\-\^0-9]+)?"
)
RELATION_PATTERNS = [
    (re.compile(r"(?P<src>[A-Za-z][\w\s/\-\(\)]{2,}?)\s+(feeds|supplies|pumps|returns|sends)\s+(?P<dst>[A-Za-z][\w\s/\-\(\)]{2,})", re.IGNORECASE), "feeds"),
    (re.compile(r"(?P<src>[A-Za-z][\w\s/\-\(\)]{2,}?)\s+(discharges|drains|flows)\s+(to|into)\s+(?P<dst>[A-Za-z][\w\s/\-\(\)]{2,})", re.IGNORECASE), "discharges_to"),
    (re.compile(r"(?P<src>[A-Za-z][\w\s/\-\(\)]{2,}?)\s+(located|installed|situated)\s+(in|at|inside)\s+(?P<dst>[A-Za-z][\w\s/\-\(\)]{2,})", re.IGNORECASE), "located_in"),
]


def _get_keyphrase_extractor():
    """Lazy load YAKE keyphrase extractor."""
    global _KEYPHRASE_EXTRACTOR
    if _KEYPHRASE_EXTRACTOR is None:
        try:
            import yake
            _KEYPHRASE_EXTRACTOR = yake.KeywordExtractor(
                lan="en",
                n=2,  # Max ngram size (2-word phrases sufficient for 700-char chunks)
                dedupLim=0.8,  # Better deduplication
                top=8,  # Top N keyphrases per chunk (8 is sufficient for ~700 chars)
                features=None
            )
        except Exception as e:
            print(f"Warning: Could not load YAKE extractor: {e}")
            _KEYPHRASE_EXTRACTOR = False
    return _KEYPHRASE_EXTRACTOR if _KEYPHRASE_EXTRACTOR is not False else None


def _normalize_label(label: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()
    return norm


def _ensure_graph(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # Enable WAL mode and set busy timeout (same as lexical_index.py)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            label TEXT,
            type TEXT,
            collection TEXT,
            doc_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT,
            dst TEXT,
            type TEXT,
            collection TEXT,
            doc_id TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)")
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique ON edges(src, dst, type, collection, doc_id)"
        )
    except sqlite3.IntegrityError:
        # Legacy graphs may contain duplicate edges; remove them and recreate the index.
        conn.execute(
            """
            DELETE FROM edges
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM edges
                GROUP BY src, dst, type, collection, doc_id
            )
            """
        )
        conn.commit()
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique ON edges(src, dst, type, collection, doc_id)"
        )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc ON nodes(doc_id)")
    return conn


def _parse_doc_ids(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data if x]
            if data:
                return [str(data)]
        except json.JSONDecodeError:
            pass
        return [raw] if raw else []
    return []


def _format_doc_ids(doc_ids: Iterable[str]) -> Optional[str]:
    unique = sorted({str(doc_id) for doc_id in doc_ids if doc_id})
    if not unique:
        return None
    return json.dumps(unique, ensure_ascii=False)


def _upsert_node(
    cur: sqlite3.Cursor,
    node_id: str,
    label: str,
    node_type: str,
    collection: str,
    doc_ids: Iterable[str],
) -> None:
    existing_doc_ids: List[str] = []
    cur.execute("SELECT doc_id FROM nodes WHERE id = ?", (node_id,))
    row = cur.fetchone()
    if row and row[0]:
        existing_doc_ids = _parse_doc_ids(row[0])
    combined = set(existing_doc_ids)
    combined.update(str(doc_id) for doc_id in doc_ids if doc_id)
    doc_field = _format_doc_ids(combined)
    cur.execute(
        """
        INSERT INTO nodes(id, label, type, collection, doc_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            label = excluded.label,
            type = excluded.type,
            collection = excluded.collection,
            doc_id = excluded.doc_id
        """,
        (node_id, label, node_type, collection, doc_field),
    )


def _node_id_doc(doc_id: str) -> str:
    return f"doc::{doc_id}"


def _node_id_section(doc_id: str, section_path: Tuple[str, ...]) -> str:
    encoded = json.dumps(section_path, ensure_ascii=False)
    return f"section::{doc_id}::{hash(encoded)}"


def _node_id_chunk(doc_id: str, element_id: str) -> str:
    return f"chunk::{doc_id}::{element_id}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def _node_id_entity(label: str) -> str:
    return f"entity::{_slugify(label)}"


def _measurement_node_id(parameter: str, value: str, unit: Optional[str], doc_id: str, chunk_node: str) -> str:
    base = f"{parameter}|{value}|{unit or ''}|{doc_id}|{chunk_node}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"measurement::{digest}"


def _keyphrase_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract keyphrases dynamically using YAKE unsupervised extraction.
    Returns list of (label, type='entity') tuples - no hardcoded classification.
    """
    extractor = _get_keyphrase_extractor()
    if extractor is None or not text:
        return []

    # Skip tiny chunks - headings/captions already contribute via other heuristics
    text_stripped = text.strip()
    if len(text_stripped) < 120 or len(text_stripped.split()) < 25:
        return []

    entities: List[Tuple[str, str]] = []
    try:
        # Extract keyphrases with YAKE
        keywords = extractor.extract_keywords(text)

        for keyphrase, score in keywords:
            # Filter noise
            keyphrase = keyphrase.strip()
            if len(keyphrase) < ENTITY_MIN_LEN:
                continue

            # Skip if mostly numeric or dotted leaders
            if re.match(r"^[\d\s\.]+$", keyphrase) or ". ." in keyphrase:
                continue

            # Alphanumeric ratio check (at least 50% alphanumeric)
            alnum_count = sum(c.isalnum() for c in keyphrase)
            if alnum_count / len(keyphrase) < 0.5:
                continue

            # All entities are generic "entity" - let graph structure reveal relationships
            entities.append((keyphrase, "entity"))

    except Exception as e:
        # Silently fail if keyphrase extraction fails
        pass

    return entities


def _table_entities(chunk: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Extract entities from table headers - store as generic 'entity' type.
    Graph structure and co-occurrence will reveal semantic relationships.
    """
    headers = chunk.get("table_headers") or []
    out: List[Tuple[str, str]] = []
    if isinstance(headers, dict):
        headers = headers.values()
    if isinstance(headers, list):
        for header in headers:
            if isinstance(header, dict):
                header = " ".join(str(v) for v in header.values())
            if not isinstance(header, str):
                continue
            label = header.strip()
            if len(label) < ENTITY_MIN_LEN:
                continue
            # Generic entity type - no hardcoded assumptions
            out.append((label, "entity"))
    return out


def _uppercase_entities(text: str) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    for match in UPPER_ENTITY_RE.findall(text):
        label = match.strip()
        if len(label) < ENTITY_MIN_LEN:
            continue
        entities.append((label, "entity"))
    return entities


def _concept_entities(doc_metadata: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not isinstance(doc_metadata, dict):
        return []
    concepts = doc_metadata.get("key_concepts") or []
    out: List[Tuple[str, str]] = []
    for concept in concepts:
        if not isinstance(concept, str):
            continue
        name = concept.strip()
        if len(name) < ENTITY_MIN_LEN:
            continue
        out.append((name, ENTITY_TYPE_CONCEPT))
    return out


def _unit_entities(doc_metadata: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not isinstance(doc_metadata, dict):
        return []
    units = doc_metadata.get("units") or []
    out: List[Tuple[str, str]] = []
    for unit in units:
        if not isinstance(unit, str):
            continue
        label = unit.strip()
        if not label:
            continue
        out.append((label, ENTITY_TYPE_UNIT))
    return out


def _metadata_bucket_entities(doc_metadata: Optional[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    if not isinstance(doc_metadata, dict):
        return []
    entities_meta = doc_metadata.get("entities") or {}
    collected: List[Tuple[str, str, str]] = []
    if isinstance(entities_meta, dict):
        for etype, items in entities_meta.items():
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, str):
                        continue
                    label = item.strip()
                    if len(label) < ENTITY_MIN_LEN:
                        continue
                    collected.append((label, etype or "entity", "metadata"))
    return collected


def _extract_entities(chunk: Dict[str, Any], doc_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract entities from chunk using hybrid approach:
    - Dynamic keyphrase extraction (YAKE) with embedding classification
    - Table headers with context-aware classification
    - Uppercase acronym detection
    - Metadata entities (from doc-level aggregation)
    """
    text = (chunk.get("text") or "").strip()
    if not text:
        return []

    entities = []

    # Phase A: Candidate harvesting
    entities.extend(_keyphrase_entities(text))  # Dynamic keyphrases with YAKE + embedding classification
    entities.extend(_table_entities(chunk))  # Table headers with dynamic classification
    entities.extend(_uppercase_entities(text))  # Acronyms (already classified as "entity", could be improved)
    entities.extend([(label, etype) for label, etype, _ in _metadata_bucket_entities(doc_metadata)])

    # Phase B: Normalization & deduplication
    seen: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []

    for label, etype in entities:
        label = label.strip()

        # Skip dotted leaders and noise
        if ". . ." in label or re.match(r"^[\d\s\.]+$", label):
            continue

        # Require minimum alphanumeric content
        if not any(c.isalnum() for c in label):
            continue

        # Alphanumeric ratio check (at least 50%)
        alnum_count = sum(c.isalnum() for c in label)
        if len(label) > 0 and alnum_count / len(label) < 0.5:
            continue

        # Deduplication by normalized key
        key = label.lower()
        if key in seen:
            continue
        seen[key] = etype

        # Determine source (for provenance)
        source = "metadata"  # Default for metadata entities
        if any(label == h for h in (chunk.get("table_headers") or [])):
            source = "table"
        elif etype not in {"equipment", "chemical", "parameter"}:
            source = "text"

        out.append({
            "label": label,
            "type": etype or "unknown",
            "source": source,
        })

    return out


def _extract_relations(text: str, entity_lookup: Dict[str, str]) -> List[Tuple[str, str, str]]:
    relations: List[Tuple[str, str, str]] = []
    if not text or not entity_lookup:
        return relations
    for pattern, relation in RELATION_PATTERNS:
        for match in pattern.finditer(text):
            src_norm = _normalize_label(match.group("src"))
            dst_norm = _normalize_label(match.group("dst"))
            if not src_norm or not dst_norm or src_norm == dst_norm:
                continue
            src_id = entity_lookup.get(src_norm)
            dst_id = entity_lookup.get(dst_norm)
            if not src_id or not dst_id:
                continue
            relations.append((src_id, dst_id, relation))
    return relations


def _doc_metadata(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for chunk in chunks:
        meta = chunk.get("doc_metadata")
        if isinstance(meta, dict) and meta:
            return meta
    return None


def _co_occurrence_pairs(node_ids: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for left, right in itertools.combinations(sorted(node_ids), 2):
        pairs.append((left, right))
    return pairs


def update_graph(
    collection: str,
    doc_id: str,
    path: str,
    chunks: Iterable[Dict],
    graph_path: str = GRAPH_DB_PATH,
) -> Dict[str, List[str]]:
    """
    Update knowledge graph with entities and relationships.
    Returns aggregated doc-level entities grouped by type.
    """
    if not collection:
        return {}
    chunk_list = list(chunks)
    conn = _ensure_graph(Path(graph_path))
    cur = conn.cursor()

    doc_node = _node_id_doc(doc_id)
    _upsert_node(cur, doc_node, path, "doc", collection, [doc_id])

    doc_meta = _doc_metadata(chunk_list)
    seen_edges: set[Tuple[str, str, str]] = set()

    # Track all entities found in document for aggregation
    doc_entities_by_type: Dict[str, set] = {}

    for label, etype in itertools.chain(_concept_entities(doc_meta), _unit_entities(doc_meta)):
        entity_id = _node_id_entity(label)
        _upsert_node(cur, entity_id, label, etype, collection, [doc_id])
        edge_type = "has_concept" if etype == ENTITY_TYPE_CONCEPT else "uses_unit"
        key = (doc_node, entity_id, edge_type)
        if key not in seen_edges:
            cur.execute(
                "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (doc_node, entity_id, edge_type, collection, doc_id),
            )
            seen_edges.add(key)

    # Progress reporting for graph entity extraction
    for chunk in tqdm(chunk_list, desc="    Graph entities", leave=False, unit="chunk"):
        section_path = tuple(chunk.get("section_path") or [])
        if section_path:
            section_node = _node_id_section(doc_id, section_path)
            _upsert_node(cur, section_node, " > ".join(section_path), "section", collection, [doc_id])
            cur.execute(
                "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (doc_node, section_node, "has_section", collection, doc_id),
            )
        else:
            section_node = None

        element_ids = chunk.get("element_ids") or []
        types = chunk.get("types") or []
        chunk_entities = _extract_entities(chunk, doc_meta)
        parameter_id_lookup: Dict[str, str] = {}
        entity_lookup: Dict[str, str] = {}

        for idx, element_id in enumerate(element_ids):
            if not element_id:
                continue
            chunk_type = types[idx] if idx < len(types) else "chunk"
            chunk_node = _node_id_chunk(doc_id, element_id)
            _upsert_node(cur, chunk_node, element_id, chunk_type, collection, [doc_id])
            parent = section_node or doc_node
            cur.execute(
                "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (parent, chunk_node, "contains", collection, doc_id),
            )

            entity_ids_for_chunk: List[str] = []
            for entity in chunk_entities:
                label = entity["label"]
                etype = entity["type"] or "entity"
                entity_id = _node_id_entity(label)
                _upsert_node(cur, entity_id, label, etype, collection, [doc_id])
                relation = "row_has_parameter" if entity.get("source") == "table" else "mentions"
                key = (chunk_node, entity_id, relation)
                if key not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (chunk_node, entity_id, relation, collection, doc_id),
                    )
                    seen_edges.add(key)
                entity_ids_for_chunk.append(entity_id)
                norm_label = _normalize_label(label)
                if norm_label:
                    entity_lookup.setdefault(norm_label, entity_id)
                if etype == "parameter" and norm_label:
                    parameter_id_lookup[norm_label] = entity_id

                # Aggregate entities for doc-level metadata
                if etype not in doc_entities_by_type:
                    doc_entities_by_type[etype] = set()
                doc_entities_by_type[etype].add(label)

            measurements = []
            text = chunk.get("text") or ""
            for match in MEASUREMENT_RE.finditer(text):
                parameter = match.group("parameter").strip()
                value = match.group("value").strip()
                unit = match.group("unit")
                if len(parameter) < ENTITY_MIN_LEN:
                    continue
                measurements.append((parameter, value, unit))

            for parameter, value, unit in measurements:
                norm_param = _normalize_label(parameter)
                parameter_entity_id = parameter_id_lookup.get(norm_param)
                if not parameter_entity_id:
                    parameter_entity_id = _node_id_entity(parameter)
                    _upsert_node(cur, parameter_entity_id, parameter, "parameter", collection, [doc_id])
                    if norm_param:
                        parameter_id_lookup[norm_param] = parameter_entity_id
                        entity_lookup.setdefault(norm_param, parameter_entity_id)
                elif norm_param:
                    entity_lookup.setdefault(norm_param, parameter_entity_id)
                entity_ids_for_chunk.append(parameter_entity_id)
                measurement_label = f"{value} {unit or ''}".strip()
                measurement_node = _measurement_node_id(parameter, value, unit, doc_id, chunk_node)
                _upsert_node(cur, measurement_node, measurement_label, ENTITY_TYPE_MEASUREMENT, collection, [doc_id])
                key_measure = (chunk_node, measurement_node, "mentions")
                if key_measure not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (chunk_node, measurement_node, "mentions", collection, doc_id),
                    )
                    seen_edges.add(key_measure)
                key_param_link = (parameter_entity_id, measurement_node, "has_measurement")
                if key_param_link not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (parameter_entity_id, measurement_node, "has_measurement", collection, doc_id),
                    )
                    seen_edges.add(key_param_link)
                edge_param_chunk = (chunk_node, parameter_entity_id, "mentions")
                if edge_param_chunk not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (chunk_node, parameter_entity_id, "mentions", collection, doc_id),
                    )
                    seen_edges.add(edge_param_chunk)

            for src_id, dst_id, relation in _extract_relations(text, entity_lookup):
                key_rel = (src_id, dst_id, relation)
                if key_rel not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (src_id, dst_id, relation, collection, doc_id),
                    )
                    seen_edges.add(key_rel)

            for left_id, right_id in _co_occurrence_pairs(entity_ids_for_chunk):
                key = (left_id, right_id, "co_occurs")
                if key not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (left_id, right_id, "co_occurs", collection, doc_id),
                    )
                    seen_edges.add(key)

    conn.commit()
    # Checkpoint WAL to prevent .db-wal file from growing too large
    conn.execute("PRAGMA wal_checkpoint(FULL);")
    conn.close()

    # Return aggregated entities grouped by type (convert sets to sorted lists)
    return {etype: sorted(list(labels)) for etype, labels in doc_entities_by_type.items()}


def neighbors(
    node_id: str,
    limit: int = 25,
    graph_path: str = GRAPH_DB_PATH,
) -> Dict[str, List[Dict[str, str]]]:
    conn = _ensure_graph(Path(graph_path))
    cur = conn.cursor()
    result = {
        "node": None,
        "outbound": [],
        "inbound": [],
    }
    cur.execute("SELECT id, label, type, collection, doc_id FROM nodes WHERE id = ?", (node_id,))
    row = cur.fetchone()
    if row:
        doc_ids = _parse_doc_ids(row[4])
        result["node"] = {
            "id": row[0],
            "label": row[1],
            "type": row[2],
            "collection": row[3],
            "doc_id": doc_ids[0] if doc_ids else None,
            "doc_ids": doc_ids,
        }
    cur.execute(
        """
        SELECT edges.dst, nodes.label, nodes.type
        FROM edges JOIN nodes ON edges.dst = nodes.id
        WHERE edges.src = ?
        LIMIT ?
        """,
        (node_id, limit),
    )
    for dst, label, node_type in cur.fetchall():
        result["outbound"].append({"id": dst, "label": label, "type": node_type})
    cur.execute(
        """
        SELECT edges.src, nodes.label, nodes.type
        FROM edges JOIN nodes ON edges.src = nodes.id
        WHERE edges.dst = ?
        LIMIT ?
        """,
        (node_id, limit),
    )
    for src, label, node_type in cur.fetchall():
        result["inbound"].append({"id": src, "label": label, "type": node_type})
    conn.close()
    return result


def list_entities(
    collection: str,
    types: Optional[Sequence[str]] = None,
    match: Optional[str] = None,
    limit: int = 50,
    graph_path: str = GRAPH_DB_PATH,
) -> List[Dict[str, Any]]:
    conn = _ensure_graph(Path(graph_path))
    cur = conn.cursor()
    sql = [
        "SELECT id, label, type, doc_id FROM nodes WHERE collection = ?",
        "AND type NOT IN ('doc', 'section')",
    ]
    params: List[Any] = [collection]
    if types:
        placeholders = ",".join("?" for _ in types)
        sql.append(f"AND type IN ({placeholders})")
        params.extend(types)
    if match:
        sql.append("AND LOWER(label) LIKE ?")
        params.append(f"%{match.lower()}%")
    sql.append("ORDER BY label LIMIT ?")
    params.append(max(1, min(limit, 500)))
    cur.execute(" ".join(sql), params)
    records = []
    for row in cur.fetchall():
        doc_ids = _parse_doc_ids(row[3])
        records.append({
            "id": row[0],
            "label": row[1],
            "type": row[2],
            "doc_id": doc_ids[0] if doc_ids else None,
            "doc_ids": doc_ids,
        })
    conn.close()
    return records


def entity_linkouts(
    entity_id: str,
    limit: int = 50,
    graph_path: str = GRAPH_DB_PATH,
) -> Dict[str, Any]:
    conn = _ensure_graph(Path(graph_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT id, label, type, collection, doc_id FROM nodes WHERE id = ?",
        (entity_id,),
    )
    entity_row = cur.fetchone()
    if not entity_row:
        conn.close()
        return {"entity": None, "mentions": []}
    cur.execute(
        """
        SELECT nodes.id, nodes.label, nodes.type, nodes.doc_id, edges.type
        FROM edges
        JOIN nodes ON edges.src = nodes.id
        WHERE edges.dst = ?
        ORDER BY nodes.doc_id, nodes.label
        LIMIT ?
        """,
        (entity_id, max(1, min(limit, 200))),
    )
    mentions: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        doc_ids = _parse_doc_ids(row[3])
        mentions.append({
            "node_id": row[0],
            "label": row[1],
            "type": row[2],
            "doc_id": doc_ids[0] if doc_ids else None,
            "doc_ids": doc_ids,
            "relation": row[4],
        })
    conn.close()
    entity_doc_ids = _parse_doc_ids(entity_row[4]) if entity_row else []
    return {
        "entity": {
            "id": entity_row[0],
            "label": entity_row[1],
            "type": entity_row[2],
            "collection": entity_row[3],
            "doc_id": entity_doc_ids[0] if entity_doc_ids else None,
            "doc_ids": entity_doc_ids,
        },
        "mentions": mentions,
    }
