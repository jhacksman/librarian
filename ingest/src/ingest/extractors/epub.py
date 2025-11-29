"""EPUB document extractor."""

import base64
import io
import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub
from PIL import Image

from ingest.extractors.base import BaseExtractor
from ingest.models import ChapterInfo, DocumentType, ExtractedContent, ImageInfo
from ingest.utils.logging import get_logger

logger = get_logger("extractors.epub")


class EPUBExtractor(BaseExtractor):
    """Extractor for EPUB documents."""

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return [".epub"]

    @property
    def document_type(self) -> DocumentType:
        """Return the document type this extractor handles."""
        return DocumentType.EPUB

    def extract(self, file_path: str | Path) -> ExtractedContent:
        """Extract content from an EPUB file.

        Args:
            file_path: Path to the EPUB file

        Returns:
            ExtractedContent with text, chapters, images, and metadata
        """
        file_path = Path(file_path)
        logger.info(f"Extracting content from EPUB: {file_path}")

        book = epub.read_epub(str(file_path))

        metadata = self._extract_metadata(book)
        title = metadata.get("title", file_path.stem)

        chapters = self._extract_chapters(book)
        images = self._extract_images(book)
        toc = self._extract_toc(book)

        raw_text = "\n\n".join(chapter.content for chapter in chapters)
        word_count = len(raw_text.split())

        logger.info(
            f"Extracted {len(chapters)} chapters, {len(images)} images, "
            f"{word_count} words from {title}"
        )

        return ExtractedContent(
            title=title,
            raw_text=raw_text,
            chapters=chapters,
            images=images,
            metadata=metadata,
            toc=toc,
            word_count=word_count,
        )

    def _extract_metadata(self, book: epub.EpubBook) -> dict:
        """Extract metadata from an EPUB book.

        Args:
            book: The EPUB book object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        title = book.get_metadata("DC", "title")
        if title:
            metadata["title"] = title[0][0]

        creators = book.get_metadata("DC", "creator")
        if creators:
            metadata["authors"] = [c[0] for c in creators]

        publisher = book.get_metadata("DC", "publisher")
        if publisher:
            metadata["publisher"] = publisher[0][0]

        date = book.get_metadata("DC", "date")
        if date:
            metadata["date"] = date[0][0]

        language = book.get_metadata("DC", "language")
        if language:
            metadata["language"] = language[0][0]

        identifier = book.get_metadata("DC", "identifier")
        if identifier:
            for ident in identifier:
                ident_value = ident[0]
                ident_attrs = ident[1] if len(ident) > 1 else {}
                scheme = ident_attrs.get("scheme", "").lower()
                if scheme == "isbn" or "isbn" in ident_value.lower():
                    isbn = re.sub(r"[^0-9X]", "", ident_value.upper())
                    if len(isbn) == 10:
                        metadata["isbn"] = isbn
                    elif len(isbn) == 13:
                        metadata["isbn13"] = isbn

        description = book.get_metadata("DC", "description")
        if description:
            soup = BeautifulSoup(description[0][0], "html.parser")
            metadata["description"] = soup.get_text(strip=True)

        subjects = book.get_metadata("DC", "subject")
        if subjects:
            metadata["subjects"] = [s[0] for s in subjects]

        return metadata

    def _extract_chapters(self, book: epub.EpubBook) -> list[ChapterInfo]:
        """Extract chapters from an EPUB book.

        Args:
            book: The EPUB book object

        Returns:
            List of ChapterInfo objects
        """
        chapters = []
        chapter_num = 0

        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                content = item.get_content().decode("utf-8", errors="ignore")
                soup = BeautifulSoup(content, "html.parser")

                text = soup.get_text(separator="\n", strip=True)
                if not text or len(text.strip()) < 100:
                    continue

                title_tag = soup.find(["h1", "h2", "h3", "title"])
                title = title_tag.get_text(strip=True) if title_tag else f"Chapter {chapter_num + 1}"

                if len(title) > 200:
                    title = title[:200] + "..."

                chapter_num += 1
                chapters.append(
                    ChapterInfo(
                        number=chapter_num,
                        title=title,
                        content=text,
                        word_count=len(text.split()),
                    )
                )

        return chapters

    def _extract_images(self, book: epub.EpubBook) -> list[ImageInfo]:
        """Extract images from an EPUB book.

        Args:
            book: The EPUB book object

        Returns:
            List of ImageInfo objects
        """
        images = []

        for item in book.get_items():
            if item.get_type() == epub.ITEM_IMAGE:
                try:
                    image_data = item.get_content()
                    content_type = item.media_type

                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size

                    if width < 50 or height < 50:
                        continue

                    base64_data = base64.b64encode(image_data).decode("utf-8")

                    images.append(
                        ImageInfo(
                            filename=item.get_name(),
                            content_type=content_type,
                            width=width,
                            height=height,
                            base64_data=base64_data,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract image {item.get_name()}: {e}")

        return images

    def _extract_toc(self, book: epub.EpubBook) -> list[dict]:
        """Extract table of contents from an EPUB book.

        Args:
            book: The EPUB book object

        Returns:
            List of TOC entries
        """
        toc = []

        def process_toc_item(item: epub.Link | tuple, level: int = 0) -> dict | None:
            if isinstance(item, epub.Link):
                return {
                    "title": item.title,
                    "href": item.href,
                    "level": level,
                }
            elif isinstance(item, tuple):
                section, children = item
                entry = {
                    "title": section.title if hasattr(section, "title") else str(section),
                    "href": section.href if hasattr(section, "href") else None,
                    "level": level,
                    "children": [],
                }
                for child in children:
                    child_entry = process_toc_item(child, level + 1)
                    if child_entry:
                        entry["children"].append(child_entry)
                return entry
            return None

        for item in book.toc:
            entry = process_toc_item(item)
            if entry:
                toc.append(entry)

        return toc
