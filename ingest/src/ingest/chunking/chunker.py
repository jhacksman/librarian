"""Semantic text chunker for RAG applications."""


from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingest.config import ChunkingConfig
from ingest.models import ChapterInfo, ExtractedContent, TextChunk
from ingest.utils.logging import get_logger

logger = get_logger("chunking")


class SemanticChunker:
    """Semantic text chunker with configurable size and overlap."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize the chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.min_chunk_size = config.min_chunk_size
        self.max_chunk_size = config.max_chunk_size
        self.overlap = config.overlap
        self.preserve_structure = config.preserve_structure

        avg_chunk_size = (self.min_chunk_size + self.max_chunk_size) // 2
        overlap_chars = int(avg_chunk_size * self.overlap)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=avg_chunk_size * 4,
            chunk_overlap=overlap_chars * 4,
            length_function=self._token_length,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

        logger.info(
            f"Initialized chunker: {self.min_chunk_size}-{self.max_chunk_size} tokens, "
            f"{int(self.overlap * 100)}% overlap"
        )

    def _token_length(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic of ~4 characters per token.

        Args:
            text: Text to measure

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _char_to_tokens(self, chars: int) -> int:
        """Convert character count to estimated tokens.

        Args:
            chars: Character count

        Returns:
            Estimated token count
        """
        return chars // 4

    def _tokens_to_chars(self, tokens: int) -> int:
        """Convert token count to estimated characters.

        Args:
            tokens: Token count

        Returns:
            Estimated character count
        """
        return tokens * 4

    def chunk_text(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        """Chunk text into semantic segments.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            token_count = self._token_length(chunk_text)

            if token_count < self.min_chunk_size and i < len(raw_chunks) - 1:
                continue

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    chunk_index=i,
                    token_count=token_count,
                    metadata={
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(raw_chunks),
                    },
                )
            )

        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.debug(f"Created {len(chunks)} chunks from {len(text)} chars")
        return chunks

    def chunk_content(self, content: ExtractedContent) -> list[TextChunk]:
        """Chunk extracted content, preserving document structure.

        Args:
            content: Extracted document content

        Returns:
            List of TextChunk objects
        """
        logger.info(f"Chunking content for: {content.title}")

        all_chunks = []

        if self.preserve_structure and content.chapters:
            for chapter in content.chapters:
                chapter_chunks = self._chunk_chapter(chapter, content.title)
                all_chunks.extend(chapter_chunks)
        else:
            base_metadata = {
                "title": content.title,
                "source_type": "full_document",
            }
            all_chunks = self.chunk_text(content.raw_text, base_metadata)

        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.metadata["global_chunk_index"] = i
            chunk.metadata["total_document_chunks"] = len(all_chunks)

        logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks

    def _chunk_chapter(
        self,
        chapter: ChapterInfo,
        book_title: str | None,
    ) -> list[TextChunk]:
        """Chunk a single chapter.

        Args:
            chapter: Chapter information
            book_title: Title of the book

        Returns:
            List of TextChunk objects for this chapter
        """
        metadata = {
            "title": book_title,
            "chapter_number": chapter.number,
            "chapter_title": chapter.title,
            "source_type": "chapter",
        }

        if chapter.start_page:
            metadata["start_page"] = chapter.start_page
        if chapter.end_page:
            metadata["end_page"] = chapter.end_page

        chunks = self.chunk_text(chapter.content, metadata)

        for chunk in chunks:
            chunk.chapter_title = chapter.title
            chunk.chapter_number = chapter.number

        return chunks

    def chunk_with_images(
        self,
        content: ExtractedContent,
        image_descriptions: list[dict],
    ) -> list[TextChunk]:
        """Chunk content and integrate image descriptions.

        Args:
            content: Extracted document content
            image_descriptions: VLM-generated image descriptions

        Returns:
            List of TextChunk objects with integrated image info
        """
        chunks = self.chunk_content(content)

        if not image_descriptions:
            return chunks

        image_by_page = {}
        for img in image_descriptions:
            page = img.get("page_number")
            if page:
                if page not in image_by_page:
                    image_by_page[page] = []
                image_by_page[page].append(img)

        for chunk in chunks:
            start_page = chunk.metadata.get("start_page")
            end_page = chunk.metadata.get("end_page")

            if start_page and end_page:
                relevant_images = []
                for page in range(start_page, end_page + 1):
                    if page in image_by_page:
                        relevant_images.extend(image_by_page[page])

                if relevant_images:
                    chunk.metadata["images"] = relevant_images
                    image_text = self._format_image_context(relevant_images)
                    chunk.content = f"{chunk.content}\n\n[Image Context]\n{image_text}"

        return chunks

    def _format_image_context(self, images: list[dict]) -> str:
        """Format image descriptions for inclusion in chunk.

        Args:
            images: List of image description dictionaries

        Returns:
            Formatted string with image context
        """
        parts = []
        for img in images:
            desc = img.get("description", "")
            ocr = img.get("ocr_text", "")
            img_type = img.get("image_type", "image")

            part = f"[{img_type.upper()}] {desc}"
            if ocr:
                part += f"\nText in image: {ocr[:500]}"
            parts.append(part)

        return "\n\n".join(parts)

    def get_chunk_stats(self, chunks: list[TextChunk]) -> dict:
        """Get statistics about chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }

        token_counts = [c.token_count for c in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) // len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }
