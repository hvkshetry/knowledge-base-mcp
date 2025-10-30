#!/bin/bash
# Simple document ingestion example
#
# This script demonstrates basic usage of ingest.py for a single directory.
# Perfect for getting started or testing the system.

set -e  # Exit on error

# Configuration
DOCS_DIR="${DOCS_DIR:-./my_documents}"
COLLECTION_NAME="${COLLECTION_NAME:-my_kb}"
FILE_EXTENSIONS="${FILE_EXTENSIONS:-.pdf,.docx,.txt}"

echo "=== Simple Ingestion Example ==="
echo "Documents directory: $DOCS_DIR"
echo "Collection name: $COLLECTION_NAME"
echo "File extensions: $FILE_EXTENSIONS"
echo ""

# Check if documents directory exists
if [ ! -d "$DOCS_DIR" ]; then
    echo "Error: Documents directory '$DOCS_DIR' not found"
    echo "Set DOCS_DIR environment variable to your documents path:"
    echo "  export DOCS_DIR=/path/to/your/documents"
    echo "  $0"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Python virtual environment not detected"
    echo "Activate with: source .venv/bin/activate"
    echo ""
fi

# Run ingestion
echo "Starting ingestion..."
python ingest.py \
    --root "$DOCS_DIR" \
    --collection "$COLLECTION_NAME" \
    --ext "$FILE_EXTENSIONS" \
    --extractor auto \
    --fts-db "data/${COLLECTION_NAME}_fts.db"

echo ""
echo "=== Ingestion Complete ==="
echo ""
echo "Next steps:"
echo "1. Verify ingestion:"
echo "   curl http://localhost:6333/collections/$COLLECTION_NAME"
echo ""
echo "2. Test search:"
echo "   python validate_search.py --query 'test query' --collection $COLLECTION_NAME"
echo ""
echo "3. Update MCP configuration to include this collection in NOMIC_KB_SCOPES"
