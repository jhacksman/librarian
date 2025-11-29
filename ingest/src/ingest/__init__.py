"""Librarian Ingest Module.

A book ingestion system that processes EPUBs/PDFs with an LLM committee
and stores results in a Qdrant vector database.
"""

__version__ = "0.1.0"

from ingest.config import IngestConfig
from ingest.models import (
    BookAnalysis,
    BookDocument,
    EmbeddedChunk,
    ExtractedContent,
    SearchResult,
    TextChunk,
)
from ingest.pipeline import IngestPipeline

__all__ = [
    "__version__",
    "IngestConfig",
    "IngestPipeline",
    "BookAnalysis",
    "BookDocument",
    "ExtractedContent",
    "TextChunk",
    "EmbeddedChunk",
    "SearchResult",
]
