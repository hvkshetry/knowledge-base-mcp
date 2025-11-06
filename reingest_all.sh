#!/bin/bash
# Re-ingest all *_kb collections with Docling-only approach

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get all *_kb directories
KB_DIRS=$(ls -d *_kb 2>/dev/null | sort)

echo "Found collections to ingest:"
echo "$KB_DIRS"
echo ""

# Loop through each collection
for kb_dir in $KB_DIRS; do
    # Skip daf_kb (already ingested)
    if [ "$kb_dir" = "daf_kb" ]; then
        echo "Skipping $kb_dir (already ingested)"
        continue
    fi

    echo "============================================"
    echo "Starting ingestion: $kb_dir"
    echo "Started at: $(date)"
    echo "============================================"

    .venv/bin/python3 ingest.py --root "$kb_dir" --qdrant-collection "$kb_dir" --max-chars 700 --batch-size 128 --parallel 1 --ollama-threads 4 --fts-db "data/${kb_dir}_fts.db" --fts-rebuild

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Successfully completed: $kb_dir"
    else
        echo "✗ FAILED: $kb_dir (exit code: $exit_code)"
    fi

    echo "Completed at: $(date)"
    echo ""
done

echo "============================================"
echo "All collections processed!"
echo "Finished at: $(date)"
echo "============================================"
