"""Qdrant vector database storage."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from ingest.config import QdrantConfig
from ingest.models import BookAnalysis, BookDocument, EmbeddedChunk, SearchResult
from ingest.storage.embeddings import EmbeddingGenerator
from ingest.utils.logging import get_logger

logger = get_logger("storage.qdrant")


class QdrantStorage:
    """Qdrant vector database storage for book documents."""

    def __init__(
        self,
        config: QdrantConfig,
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        """Initialize Qdrant storage.

        Args:
            config: Qdrant configuration
            embedding_generator: Embedding generator instance
        """
        self.config = config
        self.embedding_generator = embedding_generator
        self.collection_name = config.collection_name

        logger.info(f"Connecting to Qdrant at {config.host}:{config.port}")

        if config.host == "memory":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                host=config.host,
                port=config.port,
                api_key=config.api_key,
                https=config.https,
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_generator.dimension,
                    distance=Distance.COSINE,
                ),
            )

            self._create_payload_indexes()
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        indexes = [
            ("book_id", models.PayloadSchemaType.KEYWORD),
            ("title", models.PayloadSchemaType.TEXT),
            ("authors", models.PayloadSchemaType.KEYWORD),
            ("genres", models.PayloadSchemaType.KEYWORD),
            ("tags", models.PayloadSchemaType.KEYWORD),
            ("difficulty_level", models.PayloadSchemaType.KEYWORD),
            ("chapter_title", models.PayloadSchemaType.TEXT),
        ]

        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                logger.debug(f"Created index for {field_name}")
            except Exception as e:
                logger.debug(f"Index for {field_name} may already exist: {e}")

    def store_book(
        self,
        book_id: str,
        analysis: BookAnalysis,
        chunks: list[EmbeddedChunk],
    ) -> BookDocument:
        """Store a complete book with analysis and chunks.

        Args:
            book_id: Unique identifier for the book
            analysis: Book analysis from LLM committee
            chunks: Embedded text chunks

        Returns:
            BookDocument with storage information
        """
        logger.info(f"Storing book {book_id}: {analysis.bibliography.title}")

        points = []
        chunk_ids = []

        for chunk in chunks:
            point_id = str(uuid.uuid4())
            chunk_ids.append(point_id)

            payload = {
                "book_id": book_id,
                "title": analysis.bibliography.title,
                "authors": analysis.bibliography.authors,
                "genres": analysis.genres,
                "tags": analysis.tags,
                "difficulty_level": analysis.difficulty_level,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "chapter_title": chunk.chapter_title,
                "chapter_number": chunk.chapter_number,
                **chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(f"Stored batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")

        book_doc = BookDocument(
            id=book_id,
            analysis=analysis,
            chunk_ids=chunk_ids,
            total_chunks=len(chunks),
        )

        self._store_book_metadata(book_id, analysis, book_doc)

        logger.info(f"Stored {len(chunks)} chunks for book {book_id}")
        return book_doc

    def _store_book_metadata(
        self,
        book_id: str,
        analysis: BookAnalysis,
        book_doc: BookDocument,
    ) -> None:
        """Store book-level metadata as a separate point.

        Args:
            book_id: Book identifier
            analysis: Book analysis
            book_doc: Book document
        """
        metadata_text = f"""
Title: {analysis.bibliography.title}
Authors: {', '.join(analysis.bibliography.authors)}
Summary: {analysis.summary}
Topics: {', '.join(analysis.main_topics)}
Tags: {', '.join(analysis.tags)}
"""

        embedding = self.embedding_generator.embed_text(metadata_text)

        metadata_point = models.PointStruct(
            id=f"{book_id}_metadata",
            vector=embedding,
            payload={
                "book_id": book_id,
                "is_metadata": True,
                "title": analysis.bibliography.title,
                "authors": analysis.bibliography.authors,
                "publisher": analysis.bibliography.publisher,
                "publication_year": analysis.bibliography.publication_year,
                "isbn": analysis.bibliography.isbn,
                "isbn13": analysis.bibliography.isbn13,
                "language": analysis.bibliography.language,
                "subjects": analysis.bibliography.subjects,
                "description": analysis.bibliography.description,
                "genres": analysis.genres,
                "tags": analysis.tags,
                "target_audience": analysis.target_audience,
                "difficulty_level": analysis.difficulty_level,
                "prerequisites": analysis.prerequisites,
                "summary": analysis.summary,
                "main_topics": analysis.main_topics,
                "key_takeaways": analysis.key_takeaways,
                "unique_value": analysis.unique_value,
                "writing_style": analysis.writing_style,
                "practical_applications": analysis.practical_applications,
                "primary_themes": [t["name"] for t in analysis.primary_themes],
                "secondary_themes": [t["name"] for t in analysis.secondary_themes],
                "total_chunks": book_doc.total_chunks,
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[metadata_point],
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        include_metadata_only: bool = False,
    ) -> list[SearchResult]:
        """Search for relevant chunks.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (e.g., {"tags": ["Python"]})
            include_metadata_only: If True, only search book metadata points

        Returns:
            List of SearchResult objects
        """
        logger.debug(f"Searching for: {query[:50]}...")

        query_embedding = self.embedding_generator.embed_query(query)

        filter_conditions = []
        if filters:
            for field, values in filters.items():
                if isinstance(values, list):
                    filter_conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchAny(any=values),
                        )
                    )
                else:
                    filter_conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=values),
                        )
                    )

        if include_metadata_only:
            filter_conditions.append(
                models.FieldCondition(
                    key="is_metadata",
                    match=models.MatchValue(value=True),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    id=str(result.id),
                    score=result.score,
                    content=result.payload.get("content", ""),
                    book_id=result.payload.get("book_id", ""),
                    title=result.payload.get("title", ""),
                    chunk_index=result.payload.get("chunk_index"),
                    chapter_title=result.payload.get("chapter_title"),
                    metadata=result.payload,
                )
            )

        logger.debug(f"Found {len(search_results)} results")
        return search_results

    def search_books(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Search for books by their metadata.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            List of SearchResult objects for book metadata
        """
        return self.search(
            query=query,
            limit=limit,
            filters=filters,
            include_metadata_only=True,
        )

    def get_book_chunks(
        self,
        book_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get all chunks for a specific book.

        Args:
            book_id: Book identifier
            limit: Maximum chunks to return
            offset: Offset for pagination

        Returns:
            List of chunk payloads
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="book_id",
                        match=models.MatchValue(value=book_id),
                    ),
                    models.FieldCondition(
                        key="is_metadata",
                        match=models.MatchValue(value=False),
                    ),
                ]
            ),
            limit=limit,
            offset=offset,
            with_payload=True,
        )

        return [r.payload for r in results]

    def delete_book(self, book_id: str) -> int:
        """Delete all data for a book.

        Args:
            book_id: Book identifier

        Returns:
            Number of points deleted
        """
        logger.info(f"Deleting book: {book_id}")

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_id",
                            match=models.MatchValue(value=book_id),
                        )
                    ]
                )
            ),
        )

        logger.info(f"Deleted book {book_id}")
        return result

    def get_collection_info(self) -> dict:
        """Get information about the collection.

        Returns:
            Collection information dictionary
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    def list_books(self) -> list[dict]:
        """List all books in the collection.

        Returns:
            List of book metadata dictionaries
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_metadata",
                        match=models.MatchValue(value=True),
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
        )

        books = []
        for r in results:
            books.append({
                "book_id": r.payload.get("book_id"),
                "title": r.payload.get("title"),
                "authors": r.payload.get("authors"),
                "genres": r.payload.get("genres"),
                "tags": r.payload.get("tags"),
                "total_chunks": r.payload.get("total_chunks"),
            })

        return books
