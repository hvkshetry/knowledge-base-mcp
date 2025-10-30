# Examples

This directory contains example scripts and configurations to help you get started with the Semantic Search MCP Server.

## Scripts

### simple_ingest.sh

Basic document ingestion for a single directory.

**Usage**:
```bash
# Make executable
chmod +x examples/simple_ingest.sh

# Use with environment variables
export DOCS_DIR=/path/to/your/documents
export COLLECTION_NAME=my_knowledge_base
./examples/simple_ingest.sh
```

**What it does**:
- Ingests all supported documents from a directory
- Creates a single collection
- Uses auto extractor (smart format detection)
- Creates corresponding FTS database

**Best for**: Getting started, testing, single-topic collections

### incremental_ingest.sh

Ingestion that only processes new or changed documents.

**Usage**:
```bash
chmod +x examples/incremental_ingest.sh

export DOCS_DIR=/path/to/your/documents
export COLLECTION_NAME=my_knowledge_base
./examples/incremental_ingest.sh
```

**What it does**:
- Checks content hash of each document
- Skips unchanged documents
- Deletes and reingests changed documents
- Much faster for regular updates

**Best for**: Daily/weekly updates, large collections, cron jobs

**Cron example** (daily at 2 AM):
```bash
0 2 * * * cd /path/to/knowledge-base-mcp && source .venv/bin/activate && DOCS_DIR=/your/docs COLLECTION_NAME=kb ./examples/incremental_ingest.sh >> logs/ingest.log 2>&1
```

### multi_collection_setup.sh

Set up multiple separate knowledge bases.

**Usage**:
```bash
chmod +x examples/multi_collection_setup.sh

# Customize paths
export TECHNICAL_DOCS_DIR=/docs/technical
export BUSINESS_DOCS_DIR=/docs/business
export RESEARCH_DOCS_DIR=/docs/research

./examples/multi_collection_setup.sh
```

**What it does**:
- Ingests multiple document directories
- Creates separate collections for each
- Provides MCP configuration example
- Sets up collection-specific FTS databases

**Best for**: Organizing different document types (technical, legal, sales, etc.)

## Configuration Examples

### Claude Code Configuration

Example `.mcp.json` (in project directory):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/home/user/knowledge-base-mcp/.venv/bin/python",
      "args": ["server.py", "stdio"],
      "env": {
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "snowflake-arctic-embed:xs",
        "TEI_RERANK_URL": "http://localhost:8087",
        "QDRANT_URL": "http://localhost:6333",
        "FTS_DB_PATH": "data/fts.db",
        "NOMIC_KB_SCOPES": "{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Main Knowledge Base\"}}"
      },
      "autoApprove": ["search_kb"]
    }
  }
}
```

### Claude Desktop Configuration

Example `claude_desktop_config.json` for macOS (`~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/Users/yourname/knowledge-base-mcp/.venv/bin/python",
      "args": ["/Users/yourname/knowledge-base-mcp/server.py", "stdio"],
      "env": {
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "snowflake-arctic-embed:xs",
        "TEI_RERANK_URL": "http://localhost:8087",
        "QDRANT_URL": "http://localhost:6333",
        "FTS_DB_PATH": "/Users/yourname/knowledge-base-mcp/data/fts.db",
        "NOMIC_KB_SCOPES": "{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Knowledge Base\"}}"
      },
      "autoApprove": ["search_kb"]
    }
  }
}
```

Example for Windows (using WSL) (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "wsl",
      "args": [
        "-e",
        "bash",
        "-c",
        "cd /home/your-username/knowledge-base-mcp && OLLAMA_URL='http://localhost:11434' OLLAMA_MODEL='snowflake-arctic-embed:xs' TEI_RERANK_URL='http://localhost:8087' QDRANT_URL='http://localhost:6333' FTS_DB_PATH='/home/your-username/knowledge-base-mcp/data/fts.db' NOMIC_KB_SCOPES='{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Main Knowledge Base\"}}' /home/your-username/knowledge-base-mcp/.venv/bin/python /home/your-username/knowledge-base-mcp/server.py stdio"
      ],
      "autoApprove": ["search_kb"]
    }
  }
}
```

### Codex CLI Configuration

Example `.codex/config.toml`:

```toml
[mcp.knowledge-base]
command = "/home/user/knowledge-base-mcp/.venv/bin/python"
args = ["server.py", "stdio"]
env = {
  OLLAMA_URL = "http://localhost:11434",
  OLLAMA_MODEL = "snowflake-arctic-embed:xs",
  TEI_RERANK_URL = "http://localhost:8087",
  QDRANT_URL = "http://localhost:6333",
  FTS_DB_PATH = "data/fts.db",
  NOMIC_KB_SCOPES = "{\"kb\":{\"collection\":\"main_kb\",\"title\":\"Knowledge Base\"}}"
}

[approval]
allowed_mcp_tools = [
  "mcp__knowledge-base__search_kb"
]
```

### Multi-Collection Configuration

For multiple knowledge bases:

```json
{
  "env": {
    "NOMIC_KB_SCOPES": "{
      \"technical\": {
        \"collection\": \"engineering_kb\",
        \"title\": \"Technical Documentation\"
      },
      \"legal\": {
        \"collection\": \"legal_kb\",
        \"title\": \"Legal Research\"
      },
      \"sales\": {
        \"collection\": \"sales_kb\",
        \"title\": \"Sales Materials\"
      }
    }"
  },
  "autoApprove": [
    "search_technical",
    "search_legal",
    "search_sales"
  ]
}
```

This creates three separate search tools.

## Example Queries

### Using Claude Desktop

Once configured, you can use natural language:

```
"Search my technical documentation for information about corrosion resistance"

"Find sales materials about pricing strategies"

"Search legal documents for contract terms related to indemnification"
```

Claude will automatically use the appropriate search tool and mode.

### Using validate_search.py

Test searches directly:

```bash
# Semantic search
python validate_search.py \
  --query "biological wastewater treatment" \
  --collection engineering_kb \
  --mode semantic \
  --top-k 5

# Hybrid search with custom parameters
python validate_search.py \
  --query "316L stainless steel chloride corrosion" \
  --collection engineering_kb \
  --mode hybrid \
  --retrieve-k 48 \
  --return-k 16 \
  --top-k 10
```

## Customization Tips

### Adjust Chunk Size

For different document types:

```bash
# Legal documents (larger context)
python ingest.py --chunk-size 2400 --chunk-overlap 200 ...

# Technical documentation (default)
python ingest.py --chunk-size 1800 --chunk-overlap 150 ...

# Short-form content (smaller chunks)
python ingest.py --chunk-size 1200 --chunk-overlap 100 ...
```

### Filter Documents

Skip certain files or directories:

```bash
python ingest.py \
  --root /docs \
  --skip "*/drafts/*,*/archive/*,*.tmp" \
  --include "*/final/*,*/published/*" \
  ...
```

### Batch Processing

For very large collections:

```bash
# Process 50 documents at a time
python ingest.py \
  --root /large_collection \
  --max-docs-per-run 50 \
  --skip-existing \
  ...

# Run repeatedly until all documents processed
```

### Memory-Constrained Systems

Reduce memory usage:

```bash
python ingest.py \
  --embed-batch-size 8 \
  --workers 1 \
  ...
```

## Troubleshooting Examples

### Check Collection Status

```bash
# List all collections
curl http://localhost:6333/collections

# Get specific collection info
curl http://localhost:6333/collections/engineering_kb

# Check point count
curl http://localhost:6333/collections/engineering_kb | jq '.result.points_count'
```

### Test Search Without MCP Client

```bash
# Test vector search
python validate_search.py \
  --query "test query" \
  --collection your_kb \
  --mode semantic

# Test hybrid search (requires FTS database)
python validate_search.py \
  --query "test query" \
  --collection your_kb \
  --mode hybrid
```

### Verify FTS Database

```bash
# Check if FTS database exists and has content
sqlite3 data/your_kb_fts.db "SELECT COUNT(*) FROM fts_chunks;"
```

## Next Steps

- Read [USAGE.md](../USAGE.md) for comprehensive documentation
- See [INSTALLATION.md](../INSTALLATION.md) for setup details
- Check [ARCHITECTURE.md](../ARCHITECTURE.md) for technical details
- Review [FAQ.md](../FAQ.md) for common questions
