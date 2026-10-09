"""Real SDK stdio checks with synthetic books; execute only on Spark CI."""

import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from test_local_pilot import epub, pdf

from ingest.local.extract import extract
from ingest.local.index import LocalIndex
from ingest.local.mcp_adapter import READ_ONLY
from ingest.local.service import LibraryService, SQLiteRetriever


@pytest.fixture(scope="module")
def sdk():
    pytest.importorskip("mcp", reason="Install the pinned MCP requirements on Spark for wire tests")
    assert version("mcp") == "2.3.0", "Wire verification requires the reviewed MCP 2.3.0 lock"
    import anyio
    from mcp import Client, StdioServerParameters
    from mcp_types import ToolAnnotations

    return anyio, Client, StdioServerParameters, ToolAnnotations


def child_environment():
    return {
        "PYTHONPATH": os.environ.get("PYTHONPATH", str(Path(__file__).parents[1] / "src")),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def save_evidence(name, report):
    destination = os.environ.get("LIBRARIAN_EVIDENCE_DIR")
    if destination:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_citation(passage, sources):
    source = sources[passage["book_id"]]
    content = extract(source)
    if passage["page_start"] is not None:
        section = next(ch for ch in content.chapters if ch.start_page == passage["page_start"])
        assert passage["page_start"] == passage["page_end"] == 2
    else:
        section_number = next(
            number for number, member in content.metadata["section_sources"].items()
            if member == passage["epub_member"]
        )
        section = next(ch for ch in content.chapters if ch.number == int(section_number))
        assert passage["epub_member"] == "OPS/one.xhtml"
    assert section.content[passage["char_start"]:passage["char_end"]] == passage["quote"]
    assert passage["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert passage["source_sha256"] == passage["book_id"]
    assert passage["content_kind"] == "source_excerpt"
    assert passage["source_uri"] == (
        f"librarian://books/{passage['book_id']}/chunks/{passage['chunk_id'].split(':')[1]}"
    )
    assert "source_path" not in passage


@pytest.mark.parametrize("mode, protocol", [("auto", "2026-07-28"), ("legacy", "2025-11-25")])
def test_real_stdio_tools_and_evidence(tmp_path, sdk, mode, protocol):
    anyio, client_type, parameters_type, annotations_type = sdk
    path = tmp_path / "index.sqlite"
    sources = {}
    index = LocalIndex(path, create=True)
    try:
        for source in (epub(tmp_path / "synthetic.epub"), pdf(tmp_path / "synthetic.pdf")):
            sources[index.ingest(source)["book_id"]] = source
    finally:
        index.close()
    service = LibraryService(SQLiteRetriever(path))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    parameters = parameters_type(
        command=sys.executable,
        args=["-m", "ingest.local.mcp_server", "--index", str(path)],
        env=child_environment(),
        cwd=Path.cwd(),
    )
    transcript = {"fixture_provenance": "Original synthetic EPUB/PDF fixtures; no private library data",
                  "sdk_version": version("mcp"), "transport": "stdio", "mode": mode, "calls": []}

    async def round_trip():
        with anyio.fail_after(45):
            async with client_type(parameters, mode=mode, read_timeout_seconds=5, cache=None) as client:
                assert client.protocol_version == protocol
                transcript["protocol_version"] = client.protocol_version
                listing = await client.list_tools()
                tools = {tool.name: tool for tool in listing.tools}
                assert set(tools) == {"ask_library", "get_passage", "list_books", "library_status"}
                assert listing.next_cursor is None
                transcript["tools"] = listing.model_dump(mode="json", by_alias=True, exclude_none=True)
                expected_annotations = annotations_type(**READ_ONLY).model_dump(by_alias=True)
                for tool in tools.values():
                    assert tool.annotations is not None
                    annotations = tool.annotations.model_dump(by_alias=True)
                    for key in READ_ONLY:
                        assert annotations[key] == expected_annotations[key] == READ_ONLY[key]
                    assert tool.input_schema["type"] == "object"
                    assert tool.output_schema is not None
                    assert tool.output_schema["type"] == "object"
                ask_fields = tools["ask_library"].input_schema["properties"]
                assert ask_fields["question"]["minLength"] == 1
                assert ask_fields["question"]["maxLength"] == 2000
                assert ask_fields["limit"]["type"] == "integer"
                assert ask_fields["limit"]["minimum"] == 1
                assert ask_fields["limit"]["maximum"] == 10
                book_types = ask_fields["book_id"].get("anyOf", [ask_fields["book_id"]])
                assert any(field.get("type") == "string" and "pattern" in field for field in book_types)
                chunk_field = tools["get_passage"].input_schema["properties"]["chunk_id"]
                assert chunk_field["type"] == "string" and "pattern" in chunk_field
                book_fields = tools["list_books"].input_schema["properties"]
                assert book_fields["limit"]["minimum"] == 1
                assert book_fields["limit"]["maximum"] == 50
                assert book_fields["offset"]["minimum"] == 0
                assert book_fields["offset"]["maximum"] == 1_000_000

                async def call(name, arguments=None):
                    result = await client.call_tool(name, arguments or {})
                    assert not result.is_error
                    assert isinstance(result.structured_content, dict)
                    assert len(result.content) == 1 and result.content[0].type == "text"
                    assert json.loads(result.content[0].text) == result.structured_content
                    serialized = result.model_dump_json(by_alias=True)
                    assert str(tmp_path) not in serialized and "source_path" not in serialized
                    assert len(serialized.encode("utf-8")) < 128 * 1024
                    transcript["calls"].append({"tool": name, "arguments": arguments or {},
                                                "result": result.structured_content})
                    return result.structured_content

                status = await call("library_status")
                assert status == service.status()
                assert status["indexed_books"] == 2 and status["indexed_chunks"] == 4
                assert status["read_only"] and not status["semantic_model_loaded"]
                assert status["answer_mode"] == "evidence_only"
                results = []
                for question in ("What are regular expressions?", "cybersecurity"):
                    answer = await call("ask_library", {"question": question})
                    assert answer == service.ask(question)
                    assert not answer["abstained"] and answer["answer_mode"] == "evidence_only"
                    assert len(answer["passages"]) == 1
                    passage = answer["passages"][0]
                    assert passage["citation_id"] == "[1]"
                    assert_citation(passage, sources)
                    fetched = await call("get_passage", {"chunk_id": passage["chunk_id"]})
                    assert fetched == service.get_passage(passage["chunk_id"])
                    assert fetched == {key: value for key, value in passage.items() if key != "citation_id"}
                    results.append(answer)
                pdf_book = results[1]["passages"][0]["book_id"]
                for question, book_id in (("quasar spectroscopy", None), ("How can I?", None),
                                          ("regular expressions", pdf_book)):
                    arguments = {"question": question}
                    if book_id is not None:
                        arguments["book_id"] = book_id
                    answer = await call("ask_library", arguments)
                    assert answer == service.ask(question, book_id=book_id)
                    assert answer["abstained"] and answer["passages"] == []

                first = await call("list_books", {"limit": 1})
                assert len(first["books"]) == 1 and first["next_offset"] == 1
                second = await call("list_books", {"limit": 1, "offset": first["next_offset"]})
                assert len(second["books"]) == 1 and second["next_offset"] is None
                combined = first["books"] + second["books"]
                assert {book["book_id"] for book in combined} == set(sources)
                assert service.books() == {"books": combined, "next_offset": None}
                assert await call("list_books") == service.books()
                assert await call("list_books", {"offset": 2}) == {"books": [], "next_offset": None}

                invalid = [
                    ("boolean_limit", "ask_library", {"question": "regular", "limit": True}, None),
                    ("string_limit", "ask_library", {"question": "regular", "limit": "5"}, None),
                    ("limit_overflow", "ask_library", {"question": "regular", "limit": 11}, None),
                    ("empty_question", "ask_library", {"question": ""}, None),
                    ("blank_question", "ask_library", {"question": "   "}, "Question"),
                    ("long_question", "ask_library", {"question": "x" * 2001}, None),
                    ("bad_book_id", "ask_library", {"question": "regular", "book_id": "A" * 64}, None),
                    ("string_null_filter", "ask_library", {"question": "regular", "book_id": "null"}, None),
                    ("explicit_null_filter", "ask_library", {"question": "regular", "book_id": None}, None),
                    ("missing_passage", "get_passage", {"chunk_id": "0" * 64 + ":0"}, "not found"),
                    ("path_as_passage", "get_passage", {"chunk_id": "/etc/passwd"}, None),
                    ("long_chunk_id", "get_passage", {"chunk_id": "0" * 64 + ":" + "1" * 11}, None),
                    ("book_limit", "list_books", {"limit": 51}, None),
                    ("boolean_book_limit", "list_books", {"limit": True}, None),
                    ("negative_offset", "list_books", {"offset": -1}, None),
                    ("large_offset", "list_books", {"offset": 1_000_001}, None),
                    ("unknown_tool", "unregistered_tool", {}, "Unknown tool"),
                ]
                for label, name, arguments, message in invalid:
                    result = await client.call_tool(name, arguments)
                    assert result.is_error, label
                    text = "\n".join(block.text for block in result.content if block.type == "text")
                    assert text and "Traceback" not in text and str(tmp_path) not in text
                    if message:
                        assert message in text, label
                    # Error text may echo the deliberately invalid path input; omit it from evidence.
                    transcript["calls"].append({"case": label, "tool": name, "is_error": True})
                    assert await call("library_status") == status

    anyio.run(round_trip)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    transcript["database_unchanged"] = True
    transcript["passed"] = True
    save_evidence(f"mcp-stdio-{mode}.json", transcript)


def test_stdio_missing_index_fails_without_creating_files(tmp_path, sdk):
    missing = tmp_path / "absent-directory" / "index.sqlite"
    before = set(tmp_path.rglob("*"))
    result = subprocess.run(
        [sys.executable, "-m", "ingest.local.mcp_server", "--index", str(missing)],
        env={**os.environ, **child_environment()}, input="", capture_output=True, text=True,
        timeout=15, check=False,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert set(tmp_path.rglob("*")) == before
    save_evidence("mcp-stdio-missing-index.json", {
        "fixture_provenance": "Absent synthetic index path", "transport": "stdio",
        "nonzero_exit": True, "stdout_empty": True, "files_created": False, "passed": True,
    })


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_stdio_late_match_and_excerpt_continuation(tmp_path, sdk, mode):
    anyio, client_type, parameters_type, _ = sdk
    source_text = ("a" * 30 + " ") * 399 + "needle"
    source = epub(tmp_path / "long.epub", text=source_text)
    path = tmp_path / "index.sqlite"
    index = LocalIndex(path, create=True)
    try:
        receipt = index.ingest(source)
        assert receipt["chunks"] == 1
    finally:
        index.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    parameters = parameters_type(command=sys.executable,
                                 args=["-m", "ingest.local.mcp_server", "--index", str(path)],
                                 env=child_environment())

    async def round_trip():
        with anyio.fail_after(30):
            async with client_type(parameters, mode=mode, read_timeout_seconds=5, cache=None) as client:
                answer = await client.call_tool("ask_library", {"question": "needle"})
                assert not answer.is_error and not answer.structured_content["abstained"]
                passage = answer.structured_content["passages"][0]
                assert "needle" in passage["quote"] and len(passage["quote"]) <= 8000
                assert source_text[passage["char_start"]:passage["char_end"]] == passage["quote"]
                exact = await client.call_tool("get_passage", {"chunk_id": passage["chunk_id"], "offset": passage["excerpt_offset"]})
                assert not exact.is_error
                assert exact.structured_content["quote"] == passage["quote"]
                offset, parts = 0, []
                while offset is not None:
                    response = await client.call_tool("get_passage", {"chunk_id": passage["chunk_id"], "offset": offset})
                    assert not response.is_error
                    part = response.structured_content
                    assert source_text[part["char_start"]:part["char_end"]] == part["quote"]
                    parts.append(part["quote"])
                    offset = part["next_offset"]
                    assert len(parts) <= 2
                assert "".join(parts) == source_text
                for bad_offset in (True, "0", -1, 2_147_483_648, len(source_text)):
                    error = await client.call_tool("get_passage", {"chunk_id": passage["chunk_id"], "offset": bad_offset})
                    assert error.is_error
                save_evidence(f"mcp-late-match-{mode}.json", {
                    "fixture_provenance": "Original synthetic400-word EPUB section",
                    "protocol_version": client.protocol_version,
                    "matching_term_in_excerpt": True, "source_spans_exact": True,
                    "continuation_reconstructs_source": True, "passed": True,
                })

    anyio.run(round_trip)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
