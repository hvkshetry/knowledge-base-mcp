#!/bin/bash
# Incremental ingestion example
#
# This script demonstrates how to set up incremental ingestion that only
# processes new or changed documents. Perfect for regular updates.

set -e  # Exit on error

# Configuration
DOCS_DIR="${DOCS_DIR:-./my_documents}"
COLLECTION_NAME="${COLLECTION_NAME:-my_kb}"
FILE_EXTENSIONS="${FILE_EXTENSIONS:-.pdf,.docx,.txt}"

echo "=== Incremental Ingestion Example ==="
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

echo "Strategy: Only process new or changed documents"
echo "Changed documents will have their old chunks deleted and replaced"
echo ""

# Run incremental ingestion
echo "Starting incremental ingestion..."
python ingest.py \
    --root "$DOCS_DIR" \
    --collection "$COLLECTION_NAME" \
    --ext "$FILE_EXTENSIONS" \
    --extractor auto \
    --changed-only \
    --delete-changed \
    --fts-db "data/${COLLECTION_NAME}_fts.db"

echo ""
echo "=== Incremental Ingestion Complete ==="
echo ""
echo "This script can be run regularly (daily/weekly) to keep your"
echo "knowledge base up to date with minimal processing."
echo ""
echo "To schedule with cron (daily at 2 AM):"
echo "  0 2 * * * cd /path/to/knowledge-base-mcp && source .venv/bin/activate && DOCS_DIR=$DOCS_DIR COLLECTION_NAME=$COLLECTION_NAME $0 >> logs/ingest.log 2>&1"
