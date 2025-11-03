import itertools
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "data/graph.db")
KEYWORD_CLASSES: Dict[str, Sequence[str]] = {
    "equipment": (
        "pump",
        "compressor",
        "tank",
        "basin",
        "clarifier",
        "saturator",
        "hydroejector",
        "injector",
        "dissolver",
        "belt press",
        "centrifuge",
        "flocculator",
    ),
    "chemical": (
        "polymer",
        "polyacrylamide",
        "alum",
        "coagulant",
        "flocculant",
        "ferric",
    ),
    "parameter": (
        "air-to-solid",
        "air to solids",
        "saturation rate",
        "pressurisation rate",
        "hydraulic load",
        "capture rate",
        "solubility",
        "density",
        "velocity",
        "pressure",
        "temperature",
    ),
}
ENTITY_MIN_LEN = 4
ENTITY_TYPE_CONCEPT = "concept"
ENTITY_TYPE_UNIT = "unit"

UPPER_ENTITY_RE = re.compile(r"\b([A-Z]{3,}(?:\s+[A-Z0-9]{2,})*)\b")


def _ensure_graph(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
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
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique ON edges(src, dst, type, collection, doc_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc ON nodes(doc_id)")
    return conn


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


def _keyword_entities(text_lower: str) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    for etype, keywords in KEYWORD_CLASSES.items():
        for keyword in keywords:
            if keyword in text_lower:
                entities.append((keyword, etype))
    return entities


def _table_entities(chunk: Dict[str, Any]) -> List[Tuple[str, str]]:
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
            out.append((label, "parameter"))
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


def _collect_chunk_entities(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = (chunk.get("text") or "").strip()
    if not text:
        return []
    text_lower = text.lower()
    entities = []
    entities.extend(_keyword_entities(text_lower))
    entities.extend(_table_entities(chunk))
    entities.extend(_uppercase_entities(text))
    seen: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for label, etype in entities:
        key = label.lower()
        if key in seen:
            continue
        seen[key] = etype
        out.append(
            {
                "label": label.strip(),
                "type": etype,
                "source": "table" if etype == "parameter" else "text",
            }
        )
    return out


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
) -> None:
    if not collection:
        return
    chunk_list = list(chunks)
    conn = _ensure_graph(Path(graph_path))
    cur = conn.cursor()

    doc_node = _node_id_doc(doc_id)
    cur.execute(
        """
        INSERT OR REPLACE INTO nodes(id, label, type, collection, doc_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doc_node, path, "doc", collection, doc_id),
    )

    doc_meta = _doc_metadata(chunk_list)
    seen_edges: set[Tuple[str, str, str]] = set()

    for label, etype in itertools.chain(_concept_entities(doc_meta), _unit_entities(doc_meta)):
        entity_id = _node_id_entity(label)
        cur.execute(
            """
            INSERT OR REPLACE INTO nodes(id, label, type, collection, doc_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (entity_id, label, etype, collection, None),
        )
        edge_type = "has_concept" if etype == ENTITY_TYPE_CONCEPT else "uses_unit"
        key = (doc_node, entity_id, edge_type)
        if key not in seen_edges:
            cur.execute(
                "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (doc_node, entity_id, edge_type, collection, doc_id),
            )
            seen_edges.add(key)

    for chunk in chunk_list:
        section_path = tuple(chunk.get("section_path") or [])
        if section_path:
            section_node = _node_id_section(doc_id, section_path)
            cur.execute(
                """
                INSERT OR REPLACE INTO nodes(id, label, type, collection, doc_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (section_node, " > ".join(section_path), "section", collection, doc_id),
            )
            cur.execute(
                "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (doc_node, section_node, "has_section", collection, doc_id),
            )
        else:
            section_node = None

        element_ids = chunk.get("element_ids") or []
        types = chunk.get("types") or []
        chunk_entities = _collect_chunk_entities(chunk)

        for idx, element_id in enumerate(element_ids):
            if not element_id:
                continue
            chunk_type = types[idx] if idx < len(types) else "chunk"
            chunk_node = _node_id_chunk(doc_id, element_id)
            cur.execute(
                """
                INSERT OR REPLACE INTO nodes(id, label, type, collection, doc_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_node, element_id, chunk_type, collection, doc_id),
            )
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
                cur.execute(
                    """
                    INSERT OR REPLACE INTO nodes(id, label, type, collection, doc_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (entity_id, label, etype, collection, None),
                )
                relation = "row_has_parameter" if entity.get("source") == "table" else "mentions"
                key = (chunk_node, entity_id, relation)
                if key not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (chunk_node, entity_id, relation, collection, doc_id),
                    )
                    seen_edges.add(key)
                entity_ids_for_chunk.append(entity_id)

            for left_id, right_id in _co_occurrence_pairs(entity_ids_for_chunk):
                key = (left_id, right_id, "co_occurs")
                if key not in seen_edges:
                    cur.execute(
                        "INSERT OR IGNORE INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                        (left_id, right_id, "co_occurs", collection, doc_id),
                    )
                    seen_edges.add(key)

    conn.commit()
    conn.close()


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
        result["node"] = {
            "id": row[0],
            "label": row[1],
            "type": row[2],
            "collection": row[3],
            "doc_id": row[4],
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
    records = [
        {"id": row[0], "label": row[1], "type": row[2], "doc_id": row[3]}
        for row in cur.fetchall()
    ]
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
    mentions = [
        {
            "node_id": row[0],
            "label": row[1],
            "type": row[2],
            "doc_id": row[3],
            "relation": row[4],
        }
        for row in cur.fetchall()
    ]
    conn.close()
    return {
        "entity": {
            "id": entity_row[0],
            "label": entity_row[1],
            "type": entity_row[2],
            "collection": entity_row[3],
            "doc_id": entity_row[4],
        },
        "mentions": mentions,
    }
