import argparse
import requests
from qdrant_client import QdrantClient, models


def embed_query(ollama_url: str, model: str, text: str, metric: str) -> list[float]:
    # Try batch endpoint first
    r = requests.post(f"{ollama_url}/api/embed", json={"model": model, "input": [text]}, timeout=60)
    if r.status_code == 404:
        # Fallback to single-text embeddings endpoint
        r2 = requests.post(f"{ollama_url}/api/embeddings", json={"model": model, "prompt": text}, timeout=60)
        r2.raise_for_status()
        emb = r2.json()["embedding"]
    else:
        r.raise_for_status()
        emb = r.json()["embeddings"][0]
    if metric == "cosine":
        import numpy as np
        arr = np.array(emb, dtype="float32")
        n = (arr @ arr) ** 0.5 or 1.0
        return (arr / n).tolist()
    return emb


def run_search(ollama_url: str, model: str, qdrant_url: str, collection: str, query: str, top_k: int, metric: str):
    vec = embed_query(ollama_url, model, query, metric)
    client = QdrantClient(url=qdrant_url)
    res = client.search(
        collection_name=collection,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    if not res:
        print("No results.")
        return
    for i, p in enumerate(res, 1):
        pl = p.payload or {}
        path = pl.get("path")
        s, e = pl.get("chunk_start"), pl.get("chunk_end")
        text = (pl.get("text") or "").replace("\n", " ")
        snippet = text[:160] + ("…" if len(text) > 160 else "")
        print(f"{i}. score={p.score:.4f} path={path} [{s},{e}]\n   {snippet}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="snowflake-arctic-embed:xs")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="snowflake_kb")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--metric", choices=["cosine", "dot", "euclid"], default="cosine")
    args = ap.parse_args()

    run_search(args.ollama_url, args.ollama_model, args.qdrant_url, args.collection, args.query, args.top_k, args.metric)


if __name__ == "__main__":
    main()
