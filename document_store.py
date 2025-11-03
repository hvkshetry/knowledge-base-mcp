import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from lexical_index import fetch_texts_by_chunk_ids  # type: ignore
except Exception:  # pragma: no cover - lexical_index not yet built
    fetch_texts_by_chunk_ids = None  # type: ignore


DEFAULT_ACL = {
    "default": {"allow": ["*"], "deny": []},
    "collections": {},
    "documents": {},
}


def _normalise_subject(token: str) -> str:
    token = (token or "").strip()
    return token.lower()


def get_subjects_from_context(ctx) -> List[str]:
    """Extract subject tokens from FastMCP Context metadata."""
    subjects: List[str] = []
    metadata = getattr(ctx, "metadata", {}) or {}

    def add_subject(value: Optional[str], prefix: str) -> None:
        if not value:
            return
        token = f"{prefix}:{value}".strip()
        norm = _normalise_subject(token)
        if norm and norm not in subjects:
            subjects.append(norm)

    user = metadata.get("user") if isinstance(metadata, dict) else None
    if isinstance(user, dict):
        add_subject(user.get("id"), "user")
        add_subject(user.get("email"), "email")
        roles = user.get("roles")
        if isinstance(roles, (list, tuple)):
            for role in roles:
                add_subject(str(role), "role")
    roles = metadata.get("roles")
    if isinstance(roles, (list, tuple)):
        for role in roles:
            add_subject(str(role), "role")
    add_subject(metadata.get("tenant"), "tenant")

    if not subjects:
        subjects.append("user:anonymous")
    subjects.append("*")  # convenience for matching
    return subjects


class DocumentStore:
    """Centralised document span fetcher with ACL checks."""

    def __init__(self, fts_db_path: str, acl_config_path: Optional[str] = None):
        self.fts_db_path = fts_db_path
        self.acl_config_path = acl_config_path or os.getenv("ACL_CONFIG_PATH")
        self._acl = self._load_acl()

    def _load_acl(self) -> Dict[str, Any]:
        cfg_path = self.acl_config_path
        if not cfg_path:
            return DEFAULT_ACL.copy()
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("ACL config root must be an object")
            merged = DEFAULT_ACL.copy()
            merged["default"] = data.get("default", merged["default"])
            merged["collections"] = data.get("collections", {})
            merged["documents"] = data.get("documents", {})
            return merged
        except FileNotFoundError:
            return DEFAULT_ACL.copy()
        except Exception:
            return DEFAULT_ACL.copy()

    def refresh_acl(self) -> None:
        self._acl = self._load_acl()
        self.fetch_chunk_records.cache_clear()  # type: ignore

    @lru_cache(maxsize=4096)
    def fetch_chunk_records(self, chunk_ids_key: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
        if fetch_texts_by_chunk_ids is None:
            return {}
        chunk_ids = list(chunk_ids_key)
        return fetch_texts_by_chunk_ids(self.fts_db_path, chunk_ids)

    def get_records_bulk(self, chunk_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        chunk_ids = [cid for cid in chunk_ids if cid]
        if not chunk_ids:
            return {}
        key = tuple(chunk_ids)
        return self.fetch_chunk_records(key)

    def _match_entry(self, entry: Optional[Dict[str, Any]], subjects: List[str]) -> Optional[bool]:
        if not entry:
            return None
        deny = entry.get("deny") or []
        allow = entry.get("allow") or []
        for token in deny:
            token_norm = _normalise_subject(token)
            if token_norm == "*" or token_norm in subjects:
                return False
        for token in allow:
            token_norm = _normalise_subject(token)
            if token_norm == "*" or token_norm in subjects:
                return True
        if allow:
            return False
        return None

    def is_allowed(self, doc_id: Optional[str], collection: Optional[str], subjects: List[str]) -> bool:
        subjects = subjects or ["user:anonymous", "*"]
        doc_entry = None
        if doc_id:
            doc_entry = self._acl.get("documents", {}).get(doc_id)
        decision = self._match_entry(doc_entry, subjects)
        if decision is not None:
            return decision
        coll_entry = None
        if collection:
            coll_entry = self._acl.get("collections", {}).get(collection)
        decision = self._match_entry(coll_entry, subjects)
        if decision is not None:
            return decision
        default_entry = self._acl.get("default")
        decision = self._match_entry(default_entry, subjects)
        if decision is not None:
            return decision
        return True

    def build_row(
        self,
        row: Dict[str, Any],
        record: Optional[Dict[str, Any]],
        allowed: bool,
        include_text: bool = True,
    ) -> None:
        if not record:
            return
        for key in ("chunk_start", "chunk_end", "doc_id", "path", "filename", "mtime"):
            if row.get(key) is None and record.get(key) is not None:
                row[key] = record.get(key)
        if row.get("page_numbers") in (None, "", []):
            row["page_numbers"] = record.get("page_numbers")
        if row.get("pages") in (None, "", []):
            pages = record.get("pages")
            if isinstance(pages, list):
                row["pages"] = pages
            elif isinstance(pages, str) and pages:
                try:
                    row["pages"] = json.loads(pages)
                except Exception:
                    row["pages"] = pages
        if row.get("section_path") in (None, ""):
            section_path = record.get("section_path")
            if isinstance(section_path, list):
                row["section_path"] = section_path
            elif isinstance(section_path, str) and section_path:
                try:
                    row["section_path"] = json.loads(section_path)
                except Exception:
                    row["section_path"] = section_path
        if row.get("element_ids") in (None, ""):
            element_ids = record.get("element_ids")
            if isinstance(element_ids, list):
                row["element_ids"] = element_ids
            elif isinstance(element_ids, str) and element_ids:
                try:
                    row["element_ids"] = json.loads(element_ids)
                except Exception:
                    row["element_ids"] = element_ids
        if row.get("bboxes") in (None, ""):
            bboxes = record.get("bboxes")
            if isinstance(bboxes, list):
                row["bboxes"] = bboxes
            elif isinstance(bboxes, str) and bboxes:
                try:
                    row["bboxes"] = json.loads(bboxes)
                except Exception:
                    row["bboxes"] = bboxes
        if row.get("types") in (None, ""):
            types = record.get("types")
            if isinstance(types, list):
                row["types"] = types
            elif isinstance(types, str) and types:
                try:
                    row["types"] = json.loads(types)
                except Exception:
                    row["types"] = types
        if row.get("source_tools") in (None, ""):
            sources = record.get("source_tools")
            if isinstance(sources, list):
                row["source_tools"] = sources
            elif isinstance(sources, str) and sources:
                try:
                    row["source_tools"] = json.loads(sources)
                except Exception:
                    row["source_tools"] = sources
        if row.get("table_headers") in (None, ""):
            table_headers = record.get("table_headers")
            if isinstance(table_headers, list):
                row["table_headers"] = table_headers
            elif isinstance(table_headers, str) and table_headers:
                try:
                    row["table_headers"] = json.loads(table_headers)
                except Exception:
                    row["table_headers"] = table_headers
        if row.get("table_units") in (None, ""):
            table_units = record.get("table_units")
            if isinstance(table_units, dict):
                row["table_units"] = table_units
            elif isinstance(table_units, list):
                row["table_units"] = table_units
            elif isinstance(table_units, str) and table_units:
                try:
                    row["table_units"] = json.loads(table_units)
                except Exception:
                    row["table_units"] = table_units
        if row.get("chunk_profile") in (None, ""):
            profile = record.get("chunk_profile")
            if profile not in (None, ""):
                row["chunk_profile"] = profile
        if row.get("plan_hash") in (None, ""):
            ph = record.get("plan_hash")
            if ph not in (None, ""):
                row["plan_hash"] = ph
        if row.get("model_version") in (None, ""):
            mv = record.get("model_version")
            if mv not in (None, ""):
                row["model_version"] = mv
        if row.get("prompt_sha") in (None, ""):
            ps = record.get("prompt_sha")
            if ps not in (None, ""):
                row["prompt_sha"] = ps
        if row.get("doc_metadata") in (None, ""):
            meta = record.get("doc_metadata")
            if isinstance(meta, dict):
                row["doc_metadata"] = meta
            elif isinstance(meta, str) and meta:
                try:
                    row["doc_metadata"] = json.loads(meta)
                except Exception:
                    row["doc_metadata"] = meta
        if not allowed:
            if "text" in row:
                row.pop("text", None)
            row["forbidden"] = True
            row.setdefault("reason", "access_denied")
            return
        if include_text:
            text = record.get("text")
            if text is not None:
                row.setdefault("text", text)
