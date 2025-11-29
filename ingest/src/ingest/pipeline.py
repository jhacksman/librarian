"""Main ingestion pipeline orchestrator."""

import hashlib
from pathlib import Path

from ingest.agents.committee import LLMCommittee
from ingest.chunking.chunker import SemanticChunker
from ingest.config import IngestConfig
from ingest.extractors import get_extractor
from ingest.models import BookDocument
from ingest.storage.embeddings import EmbeddingGenerator
from ingest.storage.qdrant import QdrantStorage
from ingest.utils.logging import LogContext, get_logger

logger = get_logger("pipeline")


class IngestPipeline:
    """Main pipeline for ingesting books into the vector database."""

    def __init__(self, config: IngestConfig) -> None:
        """Initialize the ingestion pipeline.

        Args:
            config: Ingest configuration
        """
        self.config = config

        logger.info("Initializing ingestion pipeline components")

        self.chunker = SemanticChunker(config.chunking)
        logger.info("Initialized semantic chunker")

        self.embedding_generator = EmbeddingGenerator(config.embedding)
        logger.info("Initialized embedding generator")

        self.storage = QdrantStorage(config.qdrant, self.embedding_generator)
        logger.info("Initialized Qdrant storage")

        self.committee = LLMCommittee(config)
        logger.info("Initialized LLM committee")

        logger.info("Pipeline initialization complete")

    def generate_book_id(self, file_path: Path) -> str:
        """Generate a unique book ID based on file content.

        Args:
            file_path: Path to the book file

        Returns:
            Unique book ID
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()[:16]

    async def ingest_book(
        self,
        file_path: str | Path,
        book_id: str | None = None,
    ) -> BookDocument:
        """Ingest a single book into the vector database.

        Args:
            file_path: Path to the book file (EPUB or PDF)
            book_id: Optional custom book ID

        Returns:
            BookDocument with ingestion results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        book_id = book_id or self.generate_book_id(file_path)

        with LogContext(logger, "ingest_book", book_id=book_id, file=file_path.name):
            logger.info(f"Starting ingestion for: {file_path.name}")

            logger.info("Step 1/5: Extracting content")
            extractor = get_extractor(str(file_path))
            content = extractor.extract(file_path)
            logger.info(
                f"Extracted {len(content.chapters)} chapters, "
                f"{len(content.images)} images, {content.word_count} words"
            )

            logger.info("Step 2/5: Running LLM committee analysis")
            analysis = await self.committee.analyze(content)
            logger.info(
                f"Analysis complete: {len(analysis.tags)} tags, "
                f"{len(analysis.key_quotes)} quotes, "
                f"{len(analysis.technical_concepts)} concepts"
            )

            logger.info("Step 3/5: Creating semantic chunks")
            if analysis.image_descriptions:
                chunks = self.chunker.chunk_with_images(content, analysis.image_descriptions)
            else:
                chunks = self.chunker.chunk_content(content)

            chunk_stats = self.chunker.get_chunk_stats(chunks)
            logger.info(
                f"Created {chunk_stats['total_chunks']} chunks, "
                f"avg {chunk_stats['avg_tokens']} tokens"
            )

            logger.info("Step 4/5: Generating embeddings")
            embedded_chunks = self.embedding_generator.embed_chunks(chunks)
            logger.info(f"Generated {len(embedded_chunks)} embeddings")

            logger.info("Step 5/5: Storing in Qdrant")
            book_doc = self.storage.store_book(book_id, analysis, embedded_chunks)
            logger.info(f"Stored book with {book_doc.total_chunks} chunks")

            logger.info(f"Ingestion complete for: {analysis.bibliography.title}")
            return book_doc

    async def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[BookDocument]:
        """Ingest all books in a directory.

        Args:
            directory: Path to directory containing books
            recursive: Whether to search subdirectories

        Returns:
            List of BookDocument results
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        files = []
        for ext in [".epub", ".pdf"]:
            files.extend(directory.glob(f"{pattern}{ext}"))

        logger.info(f"Found {len(files)} books in {directory}")

        results = []
        for i, file_path in enumerate(files):
            logger.info(f"Processing {i + 1}/{len(files)}: {file_path.name}")
            try:
                book_doc = await self.ingest_book(file_path)
                results.append(book_doc)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")

        logger.info(f"Ingested {len(results)}/{len(files)} books successfully")
        return results

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
    ) -> list:
        """Search for relevant content.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters

        Returns:
            List of search results
        """
        return self.storage.search(query, limit, filters)

    def search_books(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
    ) -> list:
        """Search for books by metadata.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters

        Returns:
            List of book search results
        """
        return self.storage.search_books(query, limit, filters)

    def list_books(self) -> list[dict]:
        """List all ingested books.

        Returns:
            List of book metadata
        """
        return self.storage.list_books()

    def delete_book(self, book_id: str) -> None:
        """Delete a book from the database.

        Args:
            book_id: Book identifier
        """
        self.storage.delete_book(book_id)

    def get_stats(self) -> dict:
        """Get pipeline statistics.

        Returns:
            Dictionary with statistics
        """
        collection_info = self.storage.get_collection_info()
        books = self.storage.list_books()

        return {
            "collection": collection_info,
            "total_books": len(books),
            "books": books,
        }
