"""Librarian ingestion; load optional service dependencies only when requested."""

from importlib import import_module

__version__ = "0.1.0"
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


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = {"IngestConfig": "config", "IngestPipeline": "pipeline"}.get(name, "models")
    value = getattr(import_module(f"ingest.{module}"), name)
    globals()[name] = value
    return value
