#!/usr/bin/env bash
#
# Collection Management Script
#
# Utility for managing Qdrant collections (list, delete, info)
#
# Usage:
#   ./scripts/manage_collections.sh list
#   ./scripts/manage_collections.sh delete collection_name [collection_name2 ...]
#   ./scripts/manage_collections.sh info collection_name
#   ./scripts/manage_collections.sh delete-test  # Deletes all *_test and test_* collections

set -euo pipefail

QDRANT_URL=${QDRANT_URL:-http://localhost:6333}

function list_collections() {
    echo "Collections in Qdrant at $QDRANT_URL:"
    curl -s "$QDRANT_URL/collections" | jq -r '.result.collections[] | "  - \(.name) (points: \(.points_count), vectors: \(.vectors_count // "N/A"))"'
}

function delete_collection() {
    local col=$1
    echo "Deleting collection: $col"
    curl -s -X DELETE "$QDRANT_URL/collections/$col" -H 'content-type: application/json' || {
        echo "  ⚠ Failed to delete $col (may not exist)"
        return 1
    }
    echo "  ✓ Deleted $col"
}

function collection_info() {
    local col=$1
    echo "Collection info for: $col"
    curl -s "$QDRANT_URL/collections/$col" | jq '.'
}

function delete_test_collections() {
    echo "Finding test collections (matching *_test, test_*, *_optimized, *_pilot)..."
    local collections=$(curl -s "$QDRANT_URL/collections" | jq -r '.result.collections[].name | select(test("test|pilot|optimized"))')

    if [ -z "$collections" ]; then
        echo "No test collections found."
        return 0
    fi

    echo "Found test collections:"
    echo "$collections" | sed 's/^/  - /'
    echo ""
    read -p "Delete these collections? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for col in $collections; do
            delete_collection "$col"
        done
        echo ""
        echo "Test collections deleted. Remaining collections:"
        list_collections
    else
        echo "Cancelled."
    fi
}

# Main command router
case "${1:-}" in
    list)
        list_collections
        ;;
    delete)
        if [ $# -lt 2 ]; then
            echo "Usage: $0 delete collection_name [collection_name2 ...]"
            exit 1
        fi
        shift
        for col in "$@"; do
            delete_collection "$col"
        done
        echo ""
        echo "Remaining collections:"
        list_collections
        ;;
    info)
        if [ $# -ne 2 ]; then
            echo "Usage: $0 info collection_name"
            exit 1
        fi
        collection_info "$2"
        ;;
    delete-test)
        delete_test_collections
        ;;
    *)
        echo "Qdrant Collection Management"
        echo ""
        echo "Usage:"
        echo "  $0 list                                  - List all collections"
        echo "  $0 delete col1 [col2 ...]                - Delete specific collections"
        echo "  $0 info collection_name                  - Show collection details"
        echo "  $0 delete-test                           - Interactively delete test collections"
        echo ""
        echo "Environment:"
        echo "  QDRANT_URL=${QDRANT_URL}"
        echo ""
        exit 1
        ;;
esac
