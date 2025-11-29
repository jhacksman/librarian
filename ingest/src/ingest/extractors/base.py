"""Base extractor interface for document processing."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from ingest.models import DocumentType, ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("extractors.base")


class BaseExtractor(ABC):
    """Abstract base class for document extractors."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass

    @property
    @abstractmethod
    def document_type(self) -> DocumentType:
        """Return the document type this extractor handles."""
        pass

    def can_extract(self, file_path: str | Path) -> bool:
        """Check if this extractor can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this extractor can handle the file
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract(self, file_path: str | Path) -> ExtractedContent:
        """Extract content from a document.

        Args:
            file_path: Path to the document to extract

        Returns:
            ExtractedContent with text, images, and metadata
        """
        pass

    def compute_file_hash(self, file_path: str | Path) -> str:
        """Compute SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of the file's SHA-256 hash
        """
        file_path = Path(file_path)
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def get_file_size(self, file_path: str | Path) -> int:
        """Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
