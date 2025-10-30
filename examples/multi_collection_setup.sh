#!/bin/bash
# Multi-collection setup example
#
# This script demonstrates how to ingest multiple separate document collections
# that will be accessible as separate search tools in your MCP client.

set -e  # Exit on error

echo "=== Multi-Collection Setup Example ==="
echo ""
echo "This will create three separate knowledge bases:"
echo "  1. Technical documentation (engineering_kb)"
echo "  2. Business documents (business_kb)"
echo "  3. Personal research (research_kb)"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Python virtual environment not detected"
    echo "Activate with: source .venv/bin/activate"
    exit 1
fi

# Define collections (customize these paths)
declare -A COLLECTIONS=(
    ["engineering_kb"]="${TECHNICAL_DOCS_DIR:-./docs/technical}"
    ["business_kb"]="${BUSINESS_DOCS_DIR:-./docs/business}"
    ["research_kb"]="${RESEARCH_DOCS_DIR:-./docs/research}"
)

# Ingest each collection
for collection in "${!COLLECTIONS[@]}"; do
    docs_dir="${COLLECTIONS[$collection]}"

    echo "---"
    echo "Processing: $collection"
    echo "Directory: $docs_dir"

    if [ ! -d "$docs_dir" ]; then
        echo "  Skipping (directory not found)"
        continue
    fi

    echo "  Starting ingestion..."
    python ingest.py \
        --root "$docs_dir" \
        --collection "$collection" \
        --ext .pdf,.docx,.txt \
        --extractor auto \
        --changed-only \
        --delete-changed \
        --fts-db "data/${collection}_fts.db" \
        || echo "  Warning: Ingestion failed for $collection"

    echo "  Complete!"
done

echo ""
echo "=== Multi-Collection Setup Complete ==="
echo ""
echo "Now update your MCP configuration (.mcp.json or Claude Desktop config):"
echo ""
cat <<'EOF'
{
  "env": {
    "NOMIC_KB_SCOPES": "{
      \"engineering\": {
        \"collection\": \"engineering_kb\",
        \"title\": \"Engineering Documentation\"
      },
      \"business\": {
        \"collection\": \"business_kb\",
        \"title\": \"Business Documents\"
      },
      \"research\": {
        \"collection\": \"research_kb\",
        \"title\": \"Personal Research\"
      }
    }"
  },
  "autoApprove": [
    "search_engineering",
    "search_business",
    "search_research"
  ]
}
EOF
echo ""
echo "After updating configuration, restart your MCP client to access:"
echo "  - search_engineering"
echo "  - search_business"
echo "  - search_research"
