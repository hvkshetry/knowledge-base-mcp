import logging
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import re

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/]{2,}")


def _clean_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    token = token.replace("▁", " ").replace("Ġ", " ").strip()
    token = token.strip("#").strip()
    token = token.lower()
    token = token.strip("-_'\"`")
    if len(token) < 3:
        return ""
    if all(ch.isdigit() for ch in token):
        return ""
    return token


class _BasicExpander:
    def __init__(self, top_k: int = 32):
        self.top_k = top_k

    def encode(self, text: str, top_k: int = None) -> List[Tuple[str, float]]:
        top = top_k or self.top_k
        tokens = [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) >= 3]
        if not tokens:
            return []
        counts = Counter(tokens)
        total = sum(counts.values())
        items = []
        for tok, freq in counts.most_common(top * 2):
            weight = math.log1p(freq) / (total or 1.0)
            if weight <= 0:
                continue
            items.append((tok, float(weight)))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top]


class _SpladeHFExpander:
    def __init__(self, model_name: str, top_k: int = 48):
        self.model_name = model_name
        self.top_k = top_k
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"transformers/torch required for SPLADE ({exc})")
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def encode(self, text: str, top_k: int = None) -> List[Tuple[str, float]]:
        from torch import log1p, relu  # type: ignore
        import torch

        if not text.strip():
            return []
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded).logits  # (1, seq_len, vocab)
        scores = log1p(relu(logits)).squeeze(0)  # (seq_len, vocab)
        pooled, _ = torch.max(scores, dim=0)  # (vocab,)
        values, indices = torch.topk(pooled, k=min(top_k or self.top_k, pooled.numel()))
        items: List[Tuple[str, float]] = []
        for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
            if val <= 0:
                continue
            token = self.tokenizer.convert_ids_to_tokens(int(idx))
            norm = _clean_token(token)
            if not norm:
                continue
            items.append((norm, float(val)))
        unique: Dict[str, float] = {}
        for term, weight in items:
            unique[term] = max(unique.get(term, 0.0), weight)
        sorted_items = sorted(unique.items(), key=lambda x: x[1], reverse=True)
        top = top_k or self.top_k
        return sorted_items[:top]


class SparseExpander:
    """Shared sparse expansion helper for ingestion and query-time use."""

    def __init__(self, method: str = None, model_name: str = None, top_k: int = 48):
        resolved = (method or os.getenv("SPARSE_EXPANDER", "none") or "none").strip().lower()
        self.method = resolved
        self.top_k = top_k
        self._impl = None
        if resolved in {"none", "off", "false"}:
            self.method = "none"
            return
        if resolved in {"splade", "splade-hf"}:
            name = model_name or os.getenv("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
            try:
                self._impl = _SpladeHFExpander(name, top_k=top_k)
                self.method = "splade"
                logger.info("SPLADE sparse expander initialized with model %s", name)
                return
            except Exception as exc:
                logger.warning("Failed to load SPLADE model (%s); falling back to basic expander", exc)
                self.method = "basic"
                self._impl = _BasicExpander(top_k=top_k)
                return
        if resolved in {"basic", "tf"}:
            self.method = "basic"
            self._impl = _BasicExpander(top_k=top_k)
            return
        logger.warning("Unknown sparse expander '%s'; defaulting to none", resolved)
        self.method = "none"

    @property
    def enabled(self) -> bool:
        return self._impl is not None

    def encode(self, text: str, top_k: int = None) -> List[Tuple[str, float]]:
        if not self.enabled:
            return []
        return self._impl.encode(text, top_k=top_k or self.top_k)

    def encode_dict(self, text: str, top_k: int = None) -> Dict[str, float]:
        return {term: weight for term, weight in self.encode(text, top_k=top_k)}

    def encode_many(self, texts: Iterable[str], top_k: int = None) -> List[List[Tuple[str, float]]]:
        return [self.encode(t, top_k=top_k) for t in texts]
