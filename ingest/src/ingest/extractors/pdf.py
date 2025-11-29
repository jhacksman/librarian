"""PDF document extractor."""

import base64
import io
import re
from pathlib import Path

import fitz
from PIL import Image

from ingest.extractors.base import BaseExtractor
from ingest.models import ChapterInfo, DocumentType, ExtractedContent, ImageInfo
from ingest.utils.logging import get_logger

logger = get_logger("extractors.pdf")


class PDFExtractor(BaseExtractor):
    """Extractor for PDF documents."""

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return [".pdf"]

    @property
    def document_type(self) -> DocumentType:
        """Return the document type this extractor handles."""
        return DocumentType.PDF

    def extract(self, file_path: str | Path) -> ExtractedContent:
        """Extract content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            ExtractedContent with text, chapters, images, and metadata
        """
        file_path = Path(file_path)
        logger.info(f"Extracting content from PDF: {file_path}")

        doc = fitz.open(str(file_path))

        metadata = self._extract_metadata(doc)
        title = metadata.get("title", file_path.stem)

        chapters = self._extract_chapters(doc)
        images = self._extract_images(doc)
        toc = self._extract_toc(doc)

        raw_text = "\n\n".join(chapter.content for chapter in chapters)
        word_count = len(raw_text.split())

        logger.info(
            f"Extracted {len(chapters)} sections, {len(images)} images, "
            f"{word_count} words from {title}"
        )

        doc.close()

        return ExtractedContent(
            title=title,
            raw_text=raw_text,
            chapters=chapters,
            images=images,
            metadata=metadata,
            toc=toc,
            page_count=len(doc) if doc else None,
            word_count=word_count,
        )

    def _extract_metadata(self, doc: fitz.Document) -> dict:
        """Extract metadata from a PDF document.

        Args:
            doc: The PyMuPDF document object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        pdf_metadata = doc.metadata
        if pdf_metadata:
            if pdf_metadata.get("title"):
                metadata["title"] = pdf_metadata["title"]

            if pdf_metadata.get("author"):
                authors = pdf_metadata["author"]
                if "," in authors:
                    metadata["authors"] = [a.strip() for a in authors.split(",")]
                elif ";" in authors:
                    metadata["authors"] = [a.strip() for a in authors.split(";")]
                else:
                    metadata["authors"] = [authors]

            if pdf_metadata.get("subject"):
                metadata["description"] = pdf_metadata["subject"]

            if pdf_metadata.get("keywords"):
                keywords = pdf_metadata["keywords"]
                if "," in keywords:
                    metadata["subjects"] = [k.strip() for k in keywords.split(",")]
                elif ";" in keywords:
                    metadata["subjects"] = [k.strip() for k in keywords.split(";")]
                else:
                    metadata["subjects"] = [keywords]

            if pdf_metadata.get("creationDate"):
                date_str = pdf_metadata["creationDate"]
                match = re.search(r"D:(\d{4})(\d{2})?(\d{2})?", date_str)
                if match:
                    year = match.group(1)
                    month = match.group(2) or "01"
                    day = match.group(3) or "01"
                    metadata["date"] = f"{year}-{month}-{day}"

            if pdf_metadata.get("producer"):
                metadata["producer"] = pdf_metadata["producer"]

            if pdf_metadata.get("creator"):
                metadata["creator"] = pdf_metadata["creator"]

        metadata["page_count"] = len(doc)

        return metadata

    def _extract_chapters(self, doc: fitz.Document) -> list[ChapterInfo]:
        """Extract chapters/sections from a PDF document.

        Uses the PDF's table of contents if available, otherwise
        treats each page as a section.

        Args:
            doc: The PyMuPDF document object

        Returns:
            List of ChapterInfo objects
        """
        chapters = []
        toc = doc.get_toc()

        if toc:
            for i, (level, title, page_num) in enumerate(toc):
                if level > 2:
                    continue

                start_page = page_num - 1
                end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)

                text_parts = []
                for page_idx in range(start_page, min(end_page, len(doc))):
                    page = doc[page_idx]
                    text_parts.append(page.get_text())

                content = "\n".join(text_parts)
                if len(content.strip()) < 50:
                    continue

                chapters.append(
                    ChapterInfo(
                        number=len(chapters) + 1,
                        title=title,
                        start_page=start_page + 1,
                        end_page=end_page,
                        content=content,
                        word_count=len(content.split()),
                    )
                )
        else:
            all_text = []
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                all_text.append(page.get_text())

            full_text = "\n".join(all_text)

            chapters.append(
                ChapterInfo(
                    number=1,
                    title="Full Document",
                    start_page=1,
                    end_page=len(doc),
                    content=full_text,
                    word_count=len(full_text.split()),
                )
            )

        return chapters

    def _extract_images(self, doc: fitz.Document) -> list[ImageInfo]:
        """Extract images from a PDF document.

        Args:
            doc: The PyMuPDF document object

        Returns:
            List of ImageInfo objects
        """
        images = []
        seen_xrefs = set()

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]

                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_data = base_image["image"]
                    content_type = f"image/{base_image['ext']}"

                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size

                    if width < 50 or height < 50:
                        continue

                    base64_data = base64.b64encode(image_data).decode("utf-8")

                    images.append(
                        ImageInfo(
                            filename=f"page{page_idx + 1}_img{xref}.{base_image['ext']}",
                            content_type=content_type,
                            width=width,
                            height=height,
                            page_number=page_idx + 1,
                            base64_data=base64_data,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract image xref={xref} from page {page_idx + 1}: {e}")

        return images

    def _extract_toc(self, doc: fitz.Document) -> list[dict]:
        """Extract table of contents from a PDF document.

        Args:
            doc: The PyMuPDF document object

        Returns:
            List of TOC entries
        """
        toc = []
        pdf_toc = doc.get_toc()

        for level, title, page_num in pdf_toc:
            toc.append({
                "title": title,
                "page": page_num,
                "level": level - 1,
            })

        return toc
