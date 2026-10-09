"""Text-only EPUB/PDF adapters using the shared extraction models.

EPUB entries are read in memory, never extracted to disk or executed.
"""

import posixpath
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from ingest.models import ChapterInfo, ExtractedContent

MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_EPUB_BYTES = 128 * 1024 * 1024


class TextParser(HTMLParser):
    """Keep block boundaries and omit non-reading content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head"}:
            self.hidden.append(tag)
        if not self.hidden and tag in {"p", "div", "br", "li", "h1", "h2", "h3", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()
        if not self.hidden:
            self.parts.append("\n" if tag in {"p", "div", "li", "h1", "h2", "h3"} else "")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        return "\n".join(
            " ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip()
        )


def member_path(base, href):
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        raise ValueError("EPUB references must be local archive members")
    path = posixpath.normpath(posixpath.join(base, unquote(parts.path)))
    if path.startswith(("/", "../")) or path == ".." or "\\" in path:
        raise ValueError("Unsafe EPUB member path")
    return path


def extract_epub(path: Path) -> ExtractedContent:
    with ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_EPUB_BYTES:
            raise ValueError("EPUB exceeds the pilot's expanded-size limit")

        def read(name):
            if archive.getinfo(name).file_size > MAX_MEMBER_BYTES:
                raise ValueError("EPUB member exceeds the pilot's size limit")
            data = archive.read(name)
            # Do not resolve or expand custom XML entities.
            if b"<!ENTITY" in data.upper():
                raise ValueError("EPUB custom XML entities are unsupported")
            return data

        container = ET.fromstring(read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        if rootfile is None:
            raise ValueError("EPUB has no package document")
        package = member_path("", rootfile.attrib["full-path"])
        root = ET.fromstring(read(package))
        metadata = {}
        for key in ("title", "creator", "publisher", "language", "identifier", "date"):
            metadata[key] = [
                el.text.strip()
                for el in root.findall(f"{{*}}metadata/{{*}}{key}")
                if el.text and el.text.strip()
            ]
        metadata["authors"] = metadata.pop("creator")
        items = {el.attrib["id"]: el.attrib for el in root.findall("{*}manifest/{*}item")}
        chapters, sources = [], {}
        for number, ref in enumerate(root.findall("{*}spine/{*}itemref"), 1):
            if ref.get("linear", "yes") == "no":
                continue
            item = items[ref.attrib["idref"]]
            if item.get("media-type") not in {"application/xhtml+xml", "text/html"}:
                continue
            if "nav" in item.get("properties", "").split():
                continue
            name = member_path(posixpath.dirname(package), item["href"])
            parser = TextParser()
            parser.feed(read(name).decode("utf-8-sig", errors="replace"))
            text = parser.text()
            if not text:
                continue
            sources[str(number)] = name
            chapters.append(
                ChapterInfo(
                    number=number,
                    title=text.splitlines()[0][:200],
                    content=text,
                    word_count=len(text.split()),
                )
            )
        metadata["section_sources"] = sources
    text = "\n\n".join(ch.content for ch in chapters)
    return ExtractedContent(
        title=next(iter(metadata["title"]), path.stem),
        raw_text=text,
        chapters=chapters,
        metadata=metadata,
        word_count=len(text.split()),
    )


def extract_pdf(path: Path) -> ExtractedContent:
    from pypdf import PdfReader

    with path.open("rb") as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported by the pilot")
        info = reader.metadata
        title = (info.title if info else None) or path.stem
        authors = [info.author] if info and info.author else []
        chapters = []
        empty_pages = []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                empty_pages.append(number)
                continue
            chapters.append(
                ChapterInfo(
                    number=number,
                    title=f"Page {number}",
                    start_page=number,
                    end_page=number,
                    content=text,
                    word_count=len(text.split()),
                )
            )
        page_count = len(reader.pages)
    text = "\n\n".join(ch.content for ch in chapters)
    return ExtractedContent(
        title=title,
        raw_text=text,
        chapters=chapters,
        page_count=page_count,
        word_count=len(text.split()),
        metadata={"authors": authors, "pages_without_text": empty_pages},
    )


def extract(path: Path) -> ExtractedContent:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Expected a non-empty file: {path}")
    if path.suffix.lower() == ".epub":
        content = extract_epub(path)
    elif path.suffix.lower() == ".pdf":
        content = extract_pdf(path)
    else:
        raise ValueError("Only EPUB and PDF files are supported")
    if not content.raw_text.strip():
        raise ValueError(
            "No extractable text; scanned books require OCR, which this pilot does not run"
        )
    return content
