"""Local ColBERT scoring via RAGatouille."""
from __future__ import annotations

import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

os.environ.setdefault("HF_HOME", str(Path("data") / "hf_cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path("data") / "torch_extensions"))

_LANGCHAIN_PATCHED = False
_LANGCHAIN_LOCK = threading.Lock()


def _ensure_langchain_shim() -> None:
    global _LANGCHAIN_PATCHED
    if _LANGCHAIN_PATCHED:
        return
    with _LANGCHAIN_LOCK:
        if _LANGCHAIN_PATCHED:
            return
        try:
            import langchain  # noqa: F401
        except Exception:
            return
        try:
            from langchain_community.document_compressors.base import BaseDocumentCompressor
        except Exception:
            return
        import types
        import langchain

        retrievers_mod = types.ModuleType("langchain.retrievers")
        retrievers_mod.__path__ = []  # type: ignore[attr-defined]
        doc_comp_mod = types.ModuleType("langchain.retrievers.document_compressors")
        doc_comp_mod.__path__ = []  # type: ignore[attr-defined]
        doc_comp_mod.__package__ = "langchain.retrievers"
        base_mod = types.ModuleType("langchain.retrievers.document_compressors.base")
        base_mod.__package__ = "langchain.retrievers.document_compressors"
        base_mod.BaseDocumentCompressor = BaseDocumentCompressor  # type: ignore[attr-defined]

        # Register modules so downstream imports succeed.
        sys.modules.setdefault("langchain.retrievers", retrievers_mod)
        sys.modules.setdefault("langchain.retrievers.document_compressors", doc_comp_mod)
        sys.modules.setdefault("langchain.retrievers.document_compressors.base", base_mod)
        setattr(doc_comp_mod, "base", base_mod)
        setattr(retrievers_mod, "document_compressors", doc_comp_mod)
        setattr(langchain, "retrievers", retrievers_mod)
        _LANGCHAIN_PATCHED = True


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    _ensure_langchain_shim()
    from ragatouille import RAGPretrainedModel

    model = RAGPretrainedModel.from_pretrained(model_name)
    return model


class ColbertLocalReranker:
    """Thin wrapper that uses a local ColBERT model for reranking.

    The implementation leans on RAGatouille's pretrained interface but avoids
    building a separate index – we simply score query/passage pairs on demand.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0")
        self._model = _load_model(self.model_name)

    def score(self, query: str, passages: Iterable[str]) -> List[float]:
        if not query or not passages:
            return []
        docs = list(passages)
        if not docs:
            return []
        scores = self._model.score(query=query, docs=docs)
        return list(scores)
