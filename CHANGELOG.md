# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Hybrid semantic search with three modes (semantic, rerank, hybrid)
- Multi-collection support via MCP scopes
- Incremental ingestion with change detection
- Neighbor context expansion
- Time decay scoring for recency boost
- Comprehensive documentation (README, INSTALLATION, USAGE, ARCHITECTURE, FAQ)
- Example scripts for common use cases
- Docker Compose setup for Qdrant and TEI reranker
- Support for multiple document formats (PDF, DOCX, TXT, HTML, etc.)
- Production ingestion script with batch processing
- SQLite FTS5 lexical search integration
- RRF (Reciprocal Rank Fusion) for hybrid search

### Technical Details
- FastMCP-based MCP server
- Qdrant vector database with HNSW indexing
- Ollama embeddings (snowflake-arctic-embed:xs default)
- Hugging Face TEI cross-encoder reranker
- MarkItDown and Docling document extractors
- Character-based sliding window chunking
- Configurable search parameters (retrieve_k, return_k, top_k)

### Fixed
- Hardened ingestion MCP tools against path traversal by validating chunk and metadata artifact inputs with `_validate_artifact_path`.
- Updated agent prompts and documentation to reflect client-authored HyDE retries and hierarchical summary workflows.

## [1.0.0] - TBD

Initial public release.
