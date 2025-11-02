import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "data/graph.db")
ENTITY_KEYWORDS = {
    "pump": "equipment",
    "compressor": "equipment",
    "tank": "equipment",
    "basin": "equipment",
    "dissolver": "equipment",
    "air": "parameter",
    "flow": "parameter",
    "pressure": "parameter",
    "temperature": "parameter",
    "saturation": "parameter",
    "polymer": "chemical",
    "alum": "chemical",
    "coagulant": "chemical",
}
ENTITY_MIN_LEN = 4


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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc ON nodes(doc_id)")
    return conn


def _node_id_doc(doc_id: str) -> str:
    return f"doc::{doc_id}"


def _node_id_section(doc_id: str, section_path: Tuple[str, ...]) -> str:
    encoded = json.dumps(section_path, ensure_ascii=False)
    return f"section::{doc_id}::{hash(encoded)}"


def _node_id_chunk(doc_id: str, element_id: str) -> str:
    return f"chunk::{doc_id}::{element_id}"


def _normalize_entity(token: str) -> Optional[str]:
    token = token.lower().strip()
    if len(token) < ENTITY_MIN_LEN:
        return None
    return token


def _node_id_entity(name: str) -> str:
    return f"entity::{name}"


def update_graph(
    collection: str,
    doc_id: str,
    path: str,
    chunks: Iterable[Dict],
    graph_path: str = GRAPH_DB_PATH,
) -> None:
    if not collection:
        return
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
    for chunk in chunks:
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
                "INSERT INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (doc_node, section_node, "has_section", collection, doc_id),
            )
        else:
            section_node = None

        element_ids = chunk.get("element_ids") or []
        types = chunk.get("types") or []
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
                "INSERT INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                (parent, chunk_node, "contains", collection, doc_id),
            )
            text = chunk.get("text", "") or ""
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)
            matched_entities = set()
            for tok in tokens:
                key = tok.lower()
                if key in ENTITY_KEYWORDS:
                    matched_entities.add((key, ENTITY_KEYWORDS[key]))
            for name, etype in matched_entities:
                entity_id = _node_id_entity(name)
                cur.execute(
                    "INSERT OR IGNORE INTO nodes(id, label, type, collection, doc_id) VALUES(?,?,?,?,?)",
                    (entity_id, name, etype, collection, None),
                )
                cur.execute(
                    "INSERT INTO edges(src, dst, type, collection, doc_id) VALUES(?,?,?,?,?)",
                    (chunk_node, entity_id, "mentions", collection, doc_id),
                )
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
