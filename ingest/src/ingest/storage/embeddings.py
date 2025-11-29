"""Embedding generation using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest.config import EmbeddingConfig
from ingest.models import EmbeddedChunk, TextChunk
from ingest.utils.logging import get_logger

logger = get_logger("storage.embeddings")


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the embedding generator.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.dimension = config.dimension

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != self.dimension:
            logger.warning(
                f"Model dimension ({actual_dim}) differs from config ({self.dimension}). "
                f"Using actual dimension."
            )
            self.dimension = actual_dim

        logger.info(f"Loaded embedding model with dimension {self.dimension}")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.debug(f"Generating embeddings for {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

        return [emb.tolist() for emb in embeddings]

    def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for text chunks.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of EmbeddedChunk objects with embeddings
        """
        if not chunks:
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            embedded_chunks.append(
                EmbeddedChunk(
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                    metadata=chunk.metadata,
                    chapter_title=chunk.chapter_title,
                    chapter_number=chunk.chapter_number,
                    embedding=embedding,
                )
            )

        logger.info(f"Generated {len(embedded_chunks)} embeddings")
        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return self.embed_text(query)

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
