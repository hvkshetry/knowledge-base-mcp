import argparse
import requests


def ensure_alias(url: str, api_key: str | None, alias: str, collection: str) -> None:
    # Use Qdrant HTTP API to upsert alias atomically via actions list
    actions = [
        {"delete_alias": {"alias_name": alias}},
        {"create_alias": {"alias_name": alias, "collection_name": collection}},
    ]
    headers = {"content-type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    # Qdrant alias actions endpoint
    r = requests.post(f"{url}/collections/aliases", json={"actions": actions}, headers=headers, timeout=30)
    if r.status_code >= 300:
        raise SystemExit(f"Failed to update alias: {r.status_code} {r.text}")


def main():
    ap = argparse.ArgumentParser(description="Point a Qdrant alias to a collection")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key", default=None)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--collection", required=True)
    args = ap.parse_args()
    ensure_alias(args.qdrant_url, args.qdrant_api_key, args.alias, args.collection)
    print(f"Alias '{args.alias}' now points to collection '{args.collection}'.")


if __name__ == "__main__":
    main()
