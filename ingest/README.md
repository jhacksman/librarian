# Librarian Ingest Module

A book ingestion system for the Hackerspace Librarian that processes DRM-free EPUBs and PDFs, extracts metadata and content using an LLM committee, and stores everything in a Qdrant vector database for RAG-based retrieval.

## Features

- **Document Extraction**: Parse EPUBs and PDFs to extract text, images, and metadata
- **LLM Committee**: 5-7 specialized agents for comprehensive metadata extraction:
  - Genre/Tags classification
  - Bibliography extraction (author, year, publisher, ISBN)
  - Book overview summaries
  - Chapter-by-chapter summaries
  - Key quotes and themes extraction
  - Technical concepts identification
- **VLM Integration**: Qwen2.5-VL for OCR and image summarization
- **Semantic Chunking**: 500-2000 token chunks with 20% overlap
- **Vector Storage**: Qdrant with BGE-base-en-v1.5 embeddings

## Hardware Requirements

Designed for GB10 hardware (DGX Spark/ASUS Ascent GX10) with 128GB unified coherent memory:
- VLM (Qwen2.5-VL-72B-Instruct-AWQ): ~40GB
- LLM (Qwen2.5-72B-Instruct-AWQ): ~40GB
- Embeddings (BGE-base-en-v1.5): ~0.5GB
- System/KV cache overhead: ~20GB

## Installation

```bash
cd ingest
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Copy the example config and adjust for your environment:

```bash
cp config/config.example.yaml config/config.yaml
```

Key configuration options:
- `llm.base_url`: vLLM server endpoint
- `llm.model`: LLM model name (default: Qwen/Qwen2.5-32B-Instruct)
- `vlm.base_url`: VLM server endpoint
- `vlm.model`: VLM model name (default: Qwen/Qwen2.5-VL-72B-Instruct)
- `qdrant.url`: Qdrant server URL
- `embedding.model`: Embedding model (default: BAAI/bge-base-en-v1.5)

## Usage

### Single Book Ingestion

```bash
librarian-ingest process /path/to/book.epub
librarian-ingest process /path/to/book.pdf
```

### Batch Ingestion

```bash
librarian-ingest batch /path/to/books/directory --recursive
```

### Check Status

```bash
librarian-ingest status
```

## Development

```bash
# Run tests
pytest

# Run linting
ruff check src tests

# Run type checking
mypy src
```

## Architecture

```
ingest/
├── src/ingest/
│   ├── extractors/      # EPUB/PDF text and image extraction
│   ├── agents/          # LLM committee agents
│   ├── chunking/        # Semantic text chunking
│   ├── storage/         # Qdrant vector storage
│   └── utils/           # Shared utilities
├── config/              # Configuration files
└── tests/               # Test suite
```

## Debug Logging

Enable verbose logging for debugging:

```bash
librarian-ingest --verbose process /path/to/book.epub
```

Logs are written to `logs/ingest.log` with detailed information about each processing step.
