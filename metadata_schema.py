import re
from collections import Counter
from typing import Dict, List, Tuple, Any

from pydantic import BaseModel, Field, ValidationError, field_validator


STOPWORDS = {
    "the", "and", "for", "with", "that", "from", "this", "were", "have", "been",
    "into", "over", "each", "such", "within", "which", "their", "through", "into",
    "your", "about", "shall", "should", "could", "would", "there", "these", "those",
    "when", "where", "while", "against", "under", "above", "below", "after", "before",
    "between", "among", "using", "used", "based", "other", "than", "also", "only",
    "both", "per", "per", "case", "cases", "include", "includes", "including",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-/]{2,}")


class EntityBucket(BaseModel):
    equipment: List[str] = Field(default_factory=list, max_length=12)
    chemicals: List[str] = Field(default_factory=list, max_length=12)
    parameters: List[str] = Field(default_factory=list, max_length=12)

    @field_validator("equipment", "chemicals", "parameters", mode="before")
    def _normalize_items(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            items = [value]
        else:
            items = list(value)
        normalized: List[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                normalized.append(text[:64])
        return normalized


class Metadata(BaseModel):
    summary: str = Field(default="", max_length=320)
    key_concepts: List[str] = Field(default_factory=list, max_length=8)
    entities: EntityBucket = Field(default_factory=EntityBucket)
    units: List[str] = Field(default_factory=list, max_length=16)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("key_concepts", mode="before")
    def _normalize_concepts(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            items = [value]
        else:
            items = list(value)
        normalized: List[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                normalized.append(text[:48])
        return normalized

    @field_validator("units", mode="before")
    def _normalize_units(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            items = [value]
        else:
            items = list(value)
        normalized: List[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                normalized.append(text[:32])
        return normalized


def _top_concepts(text: str) -> List[str]:
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text) if tok]
    filtered = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2]
    if not filtered:
        return []
    ranking = Counter(filtered).most_common(16)
    out: List[str] = []
    for tok, _ in ranking:
        if tok not in out:
            out.append(tok)
        if len(out) >= 8:
            break
    return out


def _collect_units(chunks: List[Dict[str, object]]) -> List[str]:
    units: List[str] = []
    for chunk in chunks:
        chunk_units = chunk.get("units")
        if isinstance(chunk_units, dict):
            units.extend(list(chunk_units.values()))
        elif isinstance(chunk_units, list):
            for entry in chunk_units:
                if isinstance(entry, dict):
                    units.extend(str(v) for v in entry.values())
                elif isinstance(entry, str):
                    units.append(entry)
    unique: List[str] = []
    for unit in units:
        norm = str(unit).strip()
        if norm and norm not in unique:
            unique.append(norm)
        if len(unique) >= 16:
            break
    return unique


def _quality_score(summary: str, key_concepts: List[str]) -> float:
    score = 0.0
    if summary:
        score += 0.4
    if key_concepts:
        score += min(0.4, len(key_concepts) * 0.05)
    return round(min(score + 0.2, 1.0), 3)


def generate_metadata(raw_text: str, chunks: List[Dict[str, object]]) -> Tuple[Dict[str, object], List[str]]:
    """Produce deterministic metadata within the bounded schema."""
    summary = (raw_text or "").strip().replace("\n", " ")
    summary = summary[:320]
    concepts = _top_concepts(raw_text or "")
    units = _collect_units(chunks)
    entities = EntityBucket()  # placeholder for later enrichment
    candidate = {
        "summary": summary,
        "key_concepts": concepts,
        "entities": entities.model_dump(),
        "units": units,
        "quality_score": _quality_score(summary, concepts),
    }
    rejects: List[str] = []
    try:
        model = Metadata(**candidate)
        return model.model_dump(), rejects
    except ValidationError as exc:
        rejects.append(str(exc))
        return {}, rejects
