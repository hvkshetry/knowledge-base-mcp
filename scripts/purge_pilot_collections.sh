#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL=${QDRANT_URL:-http://localhost:6333}

echo "Deleting pilot collections from Qdrant at $QDRANT_URL ..."
for COL in snowflake_pdf_pilot snowflake_all_pilot; do
  echo "- Deleting $COL"
  curl -s -X DELETE "$QDRANT_URL/collections/$COL" -H 'content-type: application/json' || true
done
echo "Done. Current collections:"
curl -s "$QDRANT_URL/collections" | jq '.result.collections[].name'

