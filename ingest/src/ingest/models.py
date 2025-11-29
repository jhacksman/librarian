"""Data models for the ingest module."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Type of document being processed."""

    EPUB = "epub"
    PDF = "pdf"


class ProcessingStatus(str, Enum):
    """Status of document processing."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageInfo(BaseModel):
    """Information about an extracted image."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    content_type: str
    width: int | None = None
    height: int | None = None
    page_number: int | None = None
    chapter: str | None = None
    caption: str | None = None
    description: str | None = None
    base64_data: str | None = None


class ChapterInfo(BaseModel):
    """Information about a book chapter."""

    number: int
    title: str
    start_page: int | None = None
    end_page: int | None = None
    content: str = ""
    summary: str | None = None
    word_count: int = 0


class ExtractedContent(BaseModel):
    """Content extracted from a document."""

    title: str | None = None
    raw_text: str = ""
    chapters: list[ChapterInfo] = Field(default_factory=list)
    images: list[ImageInfo] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    toc: list[dict[str, Any]] = Field(default_factory=list)
    page_count: int | None = None
    word_count: int = 0


class Bibliography(BaseModel):
    """Bibliographic information about a book."""

    title: str
    authors: list[str] = Field(default_factory=list)
    publisher: str | None = None
    publication_year: int | None = None
    isbn: str | None = None
    isbn13: str | None = None
    language: str | None = None
    subjects: list[str] = Field(default_factory=list)
    description: str | None = None
    edition: str | None = None
    series: str | None = None
    series_number: int | None = None


class Quote(BaseModel):
    """A notable quote from a book."""

    text: str
    chapter: str | None = None
    page: int | None = None
    context: str | None = None
    significance: str | None = None


class TechnicalConcept(BaseModel):
    """A technical concept extracted from a book."""

    term: str
    definition: str
    chapter: str | None = None
    related_terms: list[str] = Field(default_factory=list)


class BookAnalysis(BaseModel):
    """Complete analysis of a book by the LLM committee."""

    bibliography: Bibliography | None = None
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    book_summary: str | None = None
    chapter_summaries: dict[str, str] = Field(default_factory=dict)
    key_quotes: list[Quote] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    technical_concepts: list[TechnicalConcept] = Field(default_factory=list)
    target_audience: str | None = None
    difficulty_level: str | None = None
    prerequisites: list[str] = Field(default_factory=list)


class TextChunk(BaseModel):
    """A chunk of text for embedding."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    book_id: str
    content: str
    chapter: str | None = None
    chapter_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int
    total_chunks: int | None = None
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    """A text chunk with its embedding."""

    chunk: TextChunk
    embedding: list[float]


class BookDocument(BaseModel):
    """Complete book document ready for storage."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str
    file_name: str
    document_type: DocumentType
    file_hash: str
    file_size: int

    extracted_content: ExtractedContent
    analysis: BookAnalysis
    chunks: list[TextChunk] = Field(default_factory=list)

    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    processing_error: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class BatchProgress(BaseModel):
    """Progress tracking for batch processing."""

    total_books: int
    processed_books: int = 0
    failed_books: int = 0
    current_book: str | None = None
    current_status: ProcessingStatus | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_books: list[str] = Field(default_factory=list)
    failed_book_paths: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_books == 0:
            return 0.0
        return (self.processed_books / self.total_books) * 100

    @property
    def is_complete(self) -> bool:
        """Check if batch processing is complete."""
        return self.processed_books + self.failed_books >= self.total_books


class SearchResult(BaseModel):
    """Result from a vector search."""

    chunk_id: str
    book_id: str
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
