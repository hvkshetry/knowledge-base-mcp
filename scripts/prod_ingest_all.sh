#!/usr/bin/env bash
#
# Production Ingestion Script
#
# This script provides automated batch ingestion for large document collections
# with memory management through batching strategies.
#
# USAGE:
#   export ROOT_PATH=/path/to/your/documents
#   export COLLECTION=your_collection_name
#   ./scripts/prod_ingest_all.sh
#
# ENVIRONMENT VARIABLES:
#   ROOT_PATH        - Root directory to scan (REQUIRED)
#   COLLECTION       - Qdrant collection name (default: main_kb)
#   QDRANT_URL       - Qdrant endpoint (default: http://localhost:6333)
#   OLLAMA_URL       - Ollama endpoint (default: http://localhost:11434)
#   OLLAMA_MODEL     - Embedding model (default: snowflake-arctic-embed:xs)
#   BATCH_FILES      - Documents per batch (default: 50, set 0 for directory splitting)
#   MAX_CHARS        - Chunk size (default: 1600)
#   OVERLAP          - Chunk overlap (default: 150)
#   BATCH_SIZE       - Embedding batch size (default: 48)
#   PARALLEL         - Parallel workers (default: 4)
#   MAX_FILE_MB      - Max file size (default: 64)
#   EXTS             - File extensions (e.g., ".pdf,.docx")
#   SKIP             - Skip patterns (e.g., "*/drafts/*,*.tmp")
#   MIN_WORDS        - Minimum word count (default: 0)
#   NO_DOCLING_FALLBACK - Disable Docling fallback (default: 1 for production)
#
# BATCHING STRATEGIES:
#   1. File-based (BATCH_FILES > 0): Process N files per iteration
#   2. Directory-based (BATCH_FILES = 0): Process directories at DEPTH level
#
# EXAMPLES:
#   # Basic usage with required path
#   export ROOT_PATH=/my/documents
#   export COLLECTION=my_kb
#   ./scripts/prod_ingest_all.sh
#
#   # Custom batch size and file types
#   export ROOT_PATH=/my/documents
#   export COLLECTION=my_kb
#   export BATCH_FILES=100
#   export EXTS=".pdf,.docx"
#   ./scripts/prod_ingest_all.sh
#
#   # Memory-constrained system
#   export ROOT_PATH=/my/documents
#   export COLLECTION=my_kb
#   export BATCH_FILES=25
#   export BATCH_SIZE=16
#   export PARALLEL=2
#   ./scripts/prod_ingest_all.sh

set -euo pipefail

# Configurable params (override via env)
ROOT_PATH=${ROOT_PATH:-""}
QDRANT_URL=${QDRANT_URL:-"http://localhost:6333"}
OLLAMA_URL=${OLLAMA_URL:-"http://localhost:11434"}
OLLAMA_MODEL=${OLLAMA_MODEL:-"snowflake-arctic-embed:xs"}
COLLECTION=${COLLECTION:-"main_kb"}
MAX_CHARS=${MAX_CHARS:-1600}
OVERLAP=${OVERLAP:-150}
BATCH_SIZE=${BATCH_SIZE:-48}
PARALLEL=${PARALLEL:-4}
THREADS=${THREADS:-8}
KEEPALIVE=${KEEPALIVE:-"1h"}
EXTS=${EXTS:-""}
SKIP=${SKIP:-""}
MAX_FILE_MB=${MAX_FILE_MB:-64}
# Disable Docling fallback by default for large runs to conserve RAM; set to 0 to enable
NO_DOCLING_FALLBACK=${NO_DOCLING_FALLBACK:-1}

# File-queue batching: process only N docs per run; repeat until done.
# Default to 50 to avoid OOM; set to 0 to use directory splitting instead.
BATCH_FILES=${BATCH_FILES:-50}

# Minimum alpha word count to index a file (0 = disabled)
MIN_WORDS=${MIN_WORDS:-0}

# Validate required parameters
if [ -z "$ROOT_PATH" ]; then
    echo "ERROR: ROOT_PATH environment variable is required"
    echo ""
    echo "Usage:"
    echo "  export ROOT_PATH=/path/to/your/documents"
    echo "  export COLLECTION=your_collection_name"
    echo "  $0"
    echo ""
    echo "See script header for full documentation."
    exit 1
fi

if [ ! -d "$ROOT_PATH" ]; then
    echo "ERROR: ROOT_PATH directory does not exist: $ROOT_PATH"
    exit 1
fi

echo "=== Production Ingestion ==="
echo "Root directory: $ROOT_PATH"
echo "Collection: $COLLECTION"
echo "Embedder: $OLLAMA_MODEL"
echo "Qdrant: $QDRANT_URL"
echo "Batch strategy: ${BATCH_FILES} files per run"
echo ""

# Split ingestion by directories to reduce peak memory
DEPTH=${DEPTH:-1}           # how deep to split (1 = top-level subdirs)
INCLUDE_ROOT=${INCLUDE_ROOT:-1}  # also ingest files directly under ROOT_PATH

run_ingest() {
  local root="$1"
  echo "\n=== Ingesting: $root ==="
  time NO_DOCLING_FALLBACK=${NO_DOCLING_FALLBACK} python3 ingest.py \
    --root "$root" \
    --extractor markitdown \
    --ollama-url "$OLLAMA_URL" --ollama-model "$OLLAMA_MODEL" \
    --qdrant-url "$QDRANT_URL" --qdrant-collection "$COLLECTION" \
    --changed-only --delete-changed \
    --max-chars "$MAX_CHARS" --overlap "$OVERLAP" \
    --batch-size "$BATCH_SIZE" --parallel "$PARALLEL" \
    --ollama-threads "$THREADS" --ollama-keepalive "$KEEPALIVE" \
    ${EXTS:+--ext "$EXTS"} \
    ${SKIP:+--skip "$SKIP"} \
    --max-file-mb "$MAX_FILE_MB" \
    --max-walk-depth 0 \
    ${MIN_WORDS:+--min-words "$MIN_WORDS"} \
    || echo "WARN: ingestion failed for $root; continuing"
}

if [ "$BATCH_FILES" -gt 0 ]; then
  echo "Batching by files: $BATCH_FILES per run (whole tree)"
  # Process the entire tree repeatedly in batches until fewer than BATCH_FILES are processed
  while true; do
    printf "\n=== BATCH RUN (limit %s) ===\n" "$BATCH_FILES"
    # Capture output to parse summary in a portable way
    TMP_LOG=$(mktemp)
    NO_DOCLING_FALLBACK=${NO_DOCLING_FALLBACK} python3 ingest.py \
      --root "$ROOT_PATH" \
      --extractor markitdown \
      --ollama-url "$OLLAMA_URL" --ollama-model "$OLLAMA_MODEL" \
      --qdrant-url "$QDRANT_URL" --qdrant-collection "$COLLECTION" \
      --changed-only --delete-changed \
      --max-chars "$MAX_CHARS" --overlap "$OVERLAP" \
      --batch-size "$BATCH_SIZE" --parallel "$PARALLEL" \
      --ollama-threads "$THREADS" --ollama-keepalive "$KEEPALIVE" \
      ${EXTS:+--ext "$EXTS"} \
      ${SKIP:+--skip "$SKIP"} \
      --max-file-mb "$MAX_FILE_MB" \
      --max-walk-depth -1 \
      ${MIN_WORDS:+--min-words "$MIN_WORDS"} \
      --max-docs-per-run "$BATCH_FILES" 2>&1 | tee "$TMP_LOG"

    OUT=$(cat "$TMP_LOG")
    rm -f "$TMP_LOG"

    # Parse summary from last SUMMARY line
    LAST_SUMMARY=$(echo "$OUT" | grep -E "^SUMMARY ") || true
    PROCESSED=$(echo "$LAST_SUMMARY" | sed -n 's/.*processed_docs=\([0-9]\+\).*/\1/p')
    PROCESSED=${PROCESSED:-0}
    echo "Processed in this batch: $PROCESSED"
    if [ "$PROCESSED" -lt "$BATCH_FILES" ]; then
      echo "No more work remaining (or fewer than batch size)."
      break
    fi
  done
else
  # Optionally ingest files directly under the root first
  if [ "$INCLUDE_ROOT" = "1" ]; then
    run_ingest "$ROOT_PATH"
  fi

  # Build directory list up to DEPTH
  mapfile -t DIRS < <(find "$ROOT_PATH" -mindepth 1 -maxdepth "$DEPTH" -type d | sort)

  for D in "${DIRS[@]}"; do
    run_ingest "$D"
  done
fi

echo "\nCounting vectors in $COLLECTION ..."
curl -s -X POST "$QDRANT_URL/collections/$COLLECTION/points/count" \
  -H 'content-type: application/json' \
  -d '{"exact":false}' | jq '.result.count'
