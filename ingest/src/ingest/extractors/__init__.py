"""Document extractors for EPUB and PDF files."""

from ingest.extractors.base import BaseExtractor
from ingest.extractors.epub import EPUBExtractor
from ingest.extractors.pdf import PDFExtractor

__all__ = ["BaseExtractor", "EPUBExtractor", "PDFExtractor"]


def get_extractor(file_path: str) -> BaseExtractor:
    """Get the appropriate extractor for a file based on its extension.

    Args:
        file_path: Path to the file to extract

    Returns:
        Appropriate extractor instance

    Raises:
        ValueError: If the file type is not supported
    """
    file_path_lower = file_path.lower()
    if file_path_lower.endswith(".epub"):
        return EPUBExtractor()
    elif file_path_lower.endswith(".pdf"):
        return PDFExtractor()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
