# Semantic Search MCP Server

A production-grade **Model Context Protocol (MCP)** server that provides hybrid semantic search over private document collections using local, open-source tools. Built for privacy-conscious users who want powerful document search without cloud dependencies.

## 🌟 Features

- **Hybrid Search**: Combines dense vector embeddings (semantic) with BM25 lexical search using Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Improves result quality with Hugging Face TEI reranker
- **Multi-Collection Support**: Organize documents into separate knowledge bases with dedicated search tools
- **Context-Aware Results**: Automatically expands results with neighboring chunks for better context
- **Incremental Ingestion**: Smart update detection only reprocesses changed documents
- **Local & Private**: All processing happens on your machine - no API keys, no cloud services
- **MCP Integration**: Works seamlessly with Claude Desktop, Claude Code, and Codex CLI

## 🏗️ Architecture

```
┌──────────────────────────────┐
│  MCP Clients                 │
│  Claude Desktop / Code / CLI │
└────────────┬─────────────────┘
             │ MCP Protocol (stdio)
         ↓
┌────────────────────────────────────────┐
│         server.py (FastMCP)            │
│  ┌──────────────────────────────────┐  │
│  │  Search Modes:                   │  │
│  │  • Semantic (vector only)        │  │
│  │  • Rerank (vector + reranker)    │  │
│  │  • Hybrid (RRF + rerank)         │  │
│  └──────────────────────────────────┘  │
└─────┬──────────────┬────────────┬──────┘
      │              │            │
      ↓              ↓            ↓
┌──────────┐  ┌──────────┐  ┌────────────┐
│  Qdrant  │  │ SQLite   │  │  Ollama    │
│  Vector  │  │  FTS5    │  │ Embeddings │
│   DB     │  │  (BM25)  │  └────────────┘
└──────────┘  └──────────┘
      │
      ↓
┌──────────────┐
│ TEI Reranker │
│ (Hugging     │
│  Face)       │
└──────────────┘
```

**Processing Pipeline**:
```
Documents → Extract → Chunk → Embed → [Qdrant + SQLite FTS] → Search → Rerank → Results
```

## 📋 Use Cases

- **Engineering Documentation**: Search technical manuals, specifications, and handbooks (e.g., water treatment engineering, chemical engineering)
- **Legal Research**: Query case law, contracts, and regulatory documents
- **Medical Literature**: Search research papers, clinical guidelines, and medical textbooks
- **Academic Research**: Build searchable libraries of papers and books
- **Corporate Knowledge Bases**: Make internal documentation and reports searchable
- **Personal Research**: Organize and query your personal document collection

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (for Qdrant + TEI reranker)
- Ollama (for embeddings)
- Python 3.9+

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/knowledge-base-mcp.git
cd knowledge-base-mcp
```

2. **Install Ollama** (if not already installed):
   - Visit [ollama.com](https://ollama.com) and download for your platform
   - Pull the embedding model:
   ```bash
   ollama pull snowflake-arctic-embed:xs
   ```

3. **Start Docker services** (Qdrant + TEI Reranker):
```bash
docker-compose up -d
```

4. **Set up Python environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

5. **Configure your MCP client**:

Choose the configuration for your MCP client:

**For Claude Code**:
```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json and update the Python venv path
```

**For Claude Desktop**:
- Copy `claude_desktop_config.json.example` contents to:
  - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
  - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- Update the paths to match your system (use `wsl` command on Windows)

**For Codex CLI**:
- Use `.codex/config.toml.example` as a template

6. **Ingest your first documents**:
```bash
python ingest.py \
  --root /path/to/your/documents \
  --collection my_docs \
  --ext .pdf,.docx,.txt \
  --fts-db data/my_docs_fts.db
```

7. **Test the search**:
```bash
python validate_search.py \
  --query "your search query" \
  --collection my_docs \
  --mode hybrid
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

## 🔍 Search Modes

### Semantic Search (`mode="semantic"`)
Pure vector similarity search using dense embeddings.

**Best for**: Conceptual queries, finding related content even without exact keyword matches

**Example**: "How do biological systems remove nitrogen?" will find relevant content even if it uses terms like "nitrification" or "denitrification"

### Rerank Search (`mode="rerank"`, default)
Vector retrieval followed by cross-encoder reranking.

**Best for**: Most use cases - good balance of speed and accuracy

**Example**: Standard searches where you want better precision than pure vector search

### Hybrid Search (`mode="hybrid"`)
Combines vector search + BM25 lexical search using RRF, then reranks.

**Best for**: Complex queries with both conceptual and specific keyword requirements

**Example**: "stainless steel 316L corrosion in chloride environments" benefits from both semantic understanding and exact term matching

## 📊 Search Parameters

```python
{
  "query": str,           # Search query text
  "mode": str,            # "semantic" | "rerank" | "hybrid" (default: "rerank")
  "top_k": int,           # Final results to return (1-100, default: 8)
  "retrieve_k": int,      # Initial retrieval size (1-256, default: 24)
  "return_k": int         # Post-rerank results (1-retrieve_k, default: 8)
}
```

### Parameter Tuning Guide

| Scenario | retrieve_k | return_k | top_k | mode |
|----------|-----------|----------|-------|------|
| **Quick search** | 12 | 8 | 5 | rerank |
| **Comprehensive** | 48 | 16 | 10 | hybrid |
| **High precision** | 24 | 12 | 5 | hybrid |
| **Exploratory** | 32 | 12 | 8 | semantic |

## 📚 Usage

See [USAGE.md](USAGE.md) for comprehensive documentation including:
- Ingestion parameters and examples
- Multi-collection setup
- Advanced search features (neighbor expansion, time decay)
- Incremental ingestion patterns
- Performance tuning

## 🏛️ Architecture Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for deep dive into:
- Reciprocal Rank Fusion (RRF) algorithm
- Cross-encoder reranking strategy
- Neighbor context expansion
- Embedding model selection rationale
- Chunking strategy

## 🛠️ Configuration

### Environment Variables

All configuration can be customized via environment variables. See [.env.example](.env.example) for full documentation.

Key variables:
- `OLLAMA_MODEL`: Embedding model (default: `snowflake-arctic-embed:xs`)
- `QDRANT_URL`: Qdrant server (default: `http://localhost:6333`)
- `TEI_RERANK_URL`: Reranker endpoint (default: `http://localhost:8087`)
- `HYBRID_RRF_K`: RRF parameter (default: 60)
- `NEIGHBOR_CHUNKS`: Context expansion (default: 1)

### Multi-Collection Setup

Configure multiple knowledge bases with independent search tools:

```json
{
  "NOMIC_KB_SCOPES": "{
    \"technical_docs\": {
      \"collection\": \"engineering_kb\",
      \"title\": \"Engineering Documentation\"
    },
    \"legal_docs\": {
      \"collection\": \"legal_kb\",
      \"title\": \"Legal Research\"
    }
  }"
}
```

This creates two MCP tools: `search_technical_docs` and `search_legal_docs`.

## 🔧 Troubleshooting

**Services not starting?**
```bash
# Check Docker services
docker-compose ps

# Check Ollama
curl http://localhost:11434/api/tags

# Check Qdrant
curl http://localhost:6333/collections
```

**Embeddings failing?**
- Ensure Ollama model is pulled: `ollama list`
- Check Ollama is running: `ollama serve` (usually auto-starts)
- Try reducing batch size: Add `--embed-batch-size 16` to ingest command

**Search returning no results?**
- Verify collection name matches ingestion
- Check Qdrant collection exists: `curl http://localhost:6333/collections/{collection_name}`
- Confirm FTS database path is correct

See [FAQ.md](FAQ.md) for more common issues.

## 📈 Performance

**Hardware recommendations**:
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Optimal**: 16GB RAM, 8+ CPU cores

**Benchmarks** (approximate, varies by hardware):
- Ingestion: 5-10 pages/second (with Ollama embeddings)
- Search latency: 100-300ms (hybrid mode with reranking)
- Storage: ~2-3KB per chunk (vector + payload)

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
- [Qdrant](https://qdrant.tech) - Vector database
- [Ollama](https://ollama.com) - Local LLM and embeddings
- [Hugging Face TEI](https://github.com/huggingface/text-embeddings-inference) - Reranking
- [MarkItDown](https://github.com/microsoft/markitdown) - Document extraction
- [Docling](https://github.com/DS4SD/docling) - High-fidelity PDF processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/knowledge-base-mcp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/knowledge-base-mcp/discussions)
- **Documentation**: See docs in this repository

---

**Why local semantic search?**
- 🔒 **Privacy**: Your documents never leave your machine
- 💰 **Cost**: No API fees, unlimited searches
- 🎯 **Control**: Full customization of models and parameters
- ⚡ **Speed**: No network latency for searches
- 🔌 **Offline**: Works without internet connection
