"""Ephemeral loopback-only human interface. No LAN transport or background service."""

import argparse
import re
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ingest.local.coverage import coverage_note
from ingest.local.paths import index_path
from ingest.local.service import LibraryRequestError, LibraryService, SQLiteRetriever

CATALOG_PAGE_SIZE = 20


def selected_book(service, book_id):
    if book_id is None:
        return None
    books = service.books(limit=1, offset=0, book_id=book_id)["books"]
    if not books:
        raise LibraryRequestError("Selected book is no longer indexed. Choose another book or All books.")
    return books[0]


def render_page(service, question="", answer=None, error=None, *, book_id=None, catalog_offset=0):
    status = service.status()
    catalog = service.books(limit=CATALOG_PAGE_SIZE, offset=catalog_offset)
    selection_error = None
    try:
        selected = selected_book(service, book_id)
    except LibraryRequestError as caught:
        selected = None
        selection_error = str(caught)
    library_warnings = "".join(f"<p>{escape(warning)}</p>"
                               for warning in status.get("extraction_coverage", {}).get("warnings", []))
    if selection_error:
        scope = f'<h2>Unavailable book selection</h2><p role="alert">{escape(selection_error)}</p>'
    elif selected:
        scope = (f'<h2>Search this book: {escape(selected["title"])}</h2>'
                 f'<p class="book-id">Book ID: {escape(selected["book_id"])}</p>'
                 f'<p>{escape(coverage_note(selected["extraction_coverage"]))}</p>')
    else:
        scope = '<h2>Search scope: All books</h2>'
    entries = []
    for book in catalog["books"]:
        entries.append(f'''<li><h3>{escape(book['title'])}</h3>
            <p class="book-id">Book ID: {escape(book['book_id'])}</p>
            <p>{book['chunk_count']} indexed passages</p>
            <p>{escape(coverage_note(book['extraction_coverage']))}</p>
            <button type="submit" name="choose_book" value="{escape(book['book_id'], quote=True)}"
                formaction="/browse" formnovalidate>Choose book</button></li>''')
    paging = []
    if catalog_offset:
        for label, offset in (("First books", 0), ("Previous books", max(0, catalog_offset - CATALOG_PAGE_SIZE))):
            paging.append(f'<button name="page_offset" value="{offset}" formaction="/browse" formnovalidate>{label}</button>')
    if catalog["next_offset"] is not None:
        paging.append(f'<button name="page_offset" value="{catalog["next_offset"]}" formaction="/browse" formnovalidate>Next books</button>')
    catalog_html = ('<ul class="catalog">' + "".join(entries) + '</ul>' if entries
                    else '<p>No books on this catalog page.</p>')
    sections = []
    if error:
        sections.append(f'<p role="alert">{escape(error)}</p>')
    if answer:
        sections.append(f'<h2>{escape(answer["message"])}</h2><p>Search terms: {escape(answer["retrieval_query"])}</p>')
        if answer.get("limitations"):
            sections.append(f'<p>{escape(answer["limitations"])}</p>')
        if answer.get("abstained"):
            sections.append("<p>Try a few keywords from the book, or change the book selection. Paraphrases may not match.</p>")
        if "extraction_coverage" in answer:
            sections.append("<h3>Search coverage</h3>" + "".join(
                f"<p>{escape(warning)}</p>" for warning in answer["extraction_coverage"]["warnings"]))
        for passage in answer["passages"]:
            location = f'PDF page {passage["page_start"]}' if passage["page_start"] else f'EPUB {passage["epub_member"]}'
            shortened = "<p>Excerpt shortened; source offsets below cover exactly the displayed quotation.</p>" if passage["excerpt_truncated"] else ""
            navigation = []
            for label, offset in (("Read from start", 0), ("Continue excerpt", passage["next_offset"])):
                if offset is not None and offset != passage["excerpt_offset"]:
                    location_value = f"{passage['chunk_id']}@{offset}"
                    navigation.append(f'<button name="passage_location" value="{escape(location_value, quote=True)}" '
                                      f'formaction="/passage" formnovalidate>{label}</button>')
            sections.append(f'''<article><h3>{escape(passage['citation_id'])} {escape(passage['title'])}</h3>
                <p>{escape(location)}</p><p>{escape(coverage_note(passage.get('extraction_coverage')))}</p>
                <blockquote>{escape(passage['quote'])}</blockquote>{shortened}<p>{" · ".join(navigation)}</p>
                <details><summary>Source details</summary><p>Passage: {escape(passage['chunk_id'])}</p>
                <p>SHA-256: {escape(passage['source_sha256'])}</p>
                <p>Source characters: {passage["char_start"]}–{passage["char_end"]}</p></details></article>''')
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Ask Librarian</title>
    <style>body{{font:18px/1.6 system-ui;max-width:850px;margin:4rem auto;padding:0 1.3rem;background:#faf8f2;color:#20312a}}
    h1{{font-size:2.7rem;line-height:1.15}}label{{display:block;font-weight:600}}textarea{{box-sizing:border-box;width:100%;font:inherit;padding:.8rem;border:1px solid #899d90;border-radius:6px}}
    button{{font:inherit;background:#214e3a;color:white;border:0;border-radius:6px;padding:.7rem 1.1rem;margin-top:.8rem}}
    article,.catalog li{{background:white;padding:1rem 1.4rem;margin:1rem 0;border:1px solid #ddd8ce;border-radius:8px}}
    .catalog{{padding:0;list-style:none}}.book-id{{font-size:.8rem;overflow-wrap:anywhere}}button:disabled{{opacity:.5}}
    blockquote{{white-space:pre-wrap;margin:1rem 0}}details{{font-size:.8rem;overflow-wrap:anywhere}}h2{{font-size:1.2rem}}</style>
    <main><h1>Ask Librarian</h1><p>Search your indexed books and inspect cited passages.</p>
    <p>{status['indexed_books']} books · {status['indexed_chunks']} passages indexed</p>
    <details><summary>Library coverage</summary>{library_warnings}</details>
    <form method="post" action="/ask">
    <input type="hidden" name="book_id" value="{escape(book_id or '', quote=True)}">
    <input type="hidden" name="catalog_offset" value="{catalog_offset}">
    {scope}<label for="question">What would you like to find?</label>
    <textarea id="question" name="question" rows="3" maxlength="2000" required>{escape(question)}</textarea>
    <button type="submit"{' disabled' if selection_error else ''}>Find source passages</button>
    <button name="choose_book" value="" formaction="/browse" formnovalidate>All books</button>
    <p>Preview: keyword retrieval, quoted source evidence. No generated answer or semantic model is running.</p>
    {''.join(sections)}
    <details><summary>Browse indexed books</summary><p>Choose a book to limit your next search. Full book IDs distinguish books with the same title.</p>
    {catalog_html}<nav aria-label="Catalog pages">{' '.join(paging)}</nav></details>
    </form></main></html>'''


def make_server(service, port=0):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, format, *args):
            pass  # Do not write questions or source excerpts to an access log.

        def _allowed(self):
            addresses = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in addresses and (origin is None or origin in {f"http://{address}" for address in addresses})

        def _send(self, content, status=200):
            data = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _page(self, question="", answer=None, error=None, status=200, *, book_id=None, catalog_offset=0):
            try:
                content = render_page(service, question, answer, error, book_id=book_id, catalog_offset=catalog_offset)
            except Exception:
                self._send("Library unavailable; inspect the local index before retrying", 503)
                return
            self._send(content, status)

        def do_GET(self):  # noqa: N802 - stdlib handler method name
            if not self._allowed():
                self._send("Host or Origin not allowed", 403)
            elif urlsplit(self.path).path == "/passage":
                try:
                    fields = parse_qs(urlsplit(self.path).query, max_num_fields=2)
                    offset = int(fields.get("offset", ["0"])[0])
                except ValueError:
                    self._page(error="Invalid passage location", status=400)
                    return
                try:
                    passage = service.get_passage(fields.get("chunk_id", [""])[0], offset)
                except LibraryRequestError as error:
                    self._page(error=str(error), status=400)
                except Exception:
                    self._send("Library unavailable; inspect the local index before retrying", 503)
                else:
                    self._page(answer={"message": "Indexed source excerpt", "retrieval_query": "Direct passage lookup",
                                       "passages": [{**passage, "citation_id": "[1]"}]})
            elif self.path != "/":
                self._send("Not found", 404)
            else:
                self._page()

        def do_POST(self):  # noqa: N802 - stdlib handler method name
            if not self._allowed():
                self._send("Host or Origin not allowed", 403)
                return
            if self.path not in {"/ask", "/browse", "/passage"}:
                self._send("Not found", 404)
                return
            fields = {}
            question, book_id, catalog_offset = "", None, 0
            try:
                if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                    raise ValueError("Invalid request framing")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16000:
                    raise ValueError("Request body must be 1-16000 bytes")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request body")
                fields = parse_qs(body.decode("utf-8"), max_num_fields=6, keep_blank_values=True)
                question = fields.get("question", [""])[0]
                # A repeated selection is invalid state, never an implicit All books choice.
                book_values = fields.get("book_id", [""])
                book_id = (book_values[0] or None) if len(book_values) == 1 else "!invalid-selection!"
                allowed = {"question", "book_id", "catalog_offset"}
                if self.path == "/browse":
                    allowed |= {"choose_book", "page_offset"}
                elif self.path == "/passage":
                    allowed.add("passage_location")
                if set(fields) - allowed or any(len(values) != 1 for values in fields.values()):
                    raise ValueError("Unknown or repeated form field")
                raw_offset = fields.get("catalog_offset", ["0"])[0]
                if not re.fullmatch(r"[0-9]{1,7}", raw_offset) or int(raw_offset) > 1_000_000:
                    raise ValueError("Invalid catalog page")
                catalog_offset = int(raw_offset)
            except (ValueError, TimeoutError):
                self._page(question, error="Invalid or incomplete request. No search was run.", status=400,
                           book_id=book_id, catalog_offset=catalog_offset)
                return
            try:
                if self.path == "/browse":
                    if ("choose_book" in fields) == ("page_offset" in fields):
                        raise LibraryRequestError("Choose one catalog action. No search was run.")
                    if "choose_book" in fields:
                        book_id = fields["choose_book"][0] or None
                    else:
                        page = fields["page_offset"][0]
                        if not re.fullmatch(r"[0-9]{1,7}", page) or int(page) > 1_000_000:
                            raise LibraryRequestError("Invalid catalog page. No search was run.")
                        catalog_offset = int(page)
                    selected_book(service, book_id)
                    answer = None
                elif self.path == "/passage":
                    selected_book(service, book_id)
                    location = fields.get("passage_location", [""])[0]
                    if "@" not in location:
                        raise LibraryRequestError("Invalid passage location")
                    chunk_id, raw_offset = location.rsplit("@", 1)
                    if not re.fullmatch(r"[0-9]{1,10}", raw_offset):
                        raise LibraryRequestError("Invalid excerpt offset")
                    passage = service.get_passage(chunk_id, int(raw_offset))
                    if book_id is not None and passage["book_id"] != book_id:
                        raise LibraryRequestError("Passage is outside the selected book")
                    answer = {"message": "Indexed source excerpt", "retrieval_query": "Direct passage lookup",
                              "passages": [{**passage, "citation_id": "[1]"}]}
                else:
                    selected_book(service, book_id)
                    answer = service.ask(question, book_id=book_id)
            except LibraryRequestError as error:
                self._page(question, error=str(error), status=400, book_id=book_id, catalog_offset=catalog_offset)
            except Exception:
                self._send("Library unavailable; inspect the local index before retrying", 503)
            else:
                self._page(question, answer, book_id=book_id, catalog_offset=catalog_offset)

    # Binding is intentionally fixed; LAN deployment is a separate approved change.
    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    service = LibraryService(SQLiteRetriever(index_path(index=args.index)))
    service.status()  # Fail before binding if the existing index cannot be opened.
    server = make_server(service, args.port)
    print(f"Ask Librarian: http://127.0.0.1:{server.server_port} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
