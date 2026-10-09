"""Original deterministic source fixtures for the frozen literal experiment.

The expected literals and outcomes below are authored independently of the query
planner.  Each long section fits in one 400-word chunk while exceeding the
8,000-character excerpt limit.  The Unicode padding is deliberately one word per
item: word-window chunking must not turn a late match into a short candidate.
"""

SEED = 20261004
UNICODE_WORD = "界" * 96
FLAGS = tuple(f"--flag{number:03}" for number in range(128))


def _padding(words):
    return " ".join([UNICODE_WORD] * words)


def _document(label, title, sections, source_format="epub"):
    return {"id": label, "format": source_format, "title": title, "sections": sections}


def _case(label, question, book, literals, *, no_match=False, quotes=(), **checks):
    return {
        "id": label,
        "question": question,
        "book_filter": book,
        "expected_no_match": no_match,
        "expected_book": None if no_match else book,
        "expected_literals": list(literals),
        "required_quotes": list(quotes),
        "expected_fts_candidates": 1,
        **checks,
    }


def fixtures():
    """Return fresh source documents and book-filtered independent probe cases."""
    long_padding = _padding(360)
    flag_padding = _padding(250)
    early_quote = "Sparkle++ marks the first observation in this original notebook."
    late_quote = "Later::signal closes this original notebook after its long silent interval."
    unicode_quote = (
        "Atlas::open records naïve café, coöperation, 東京, and decomposed cafe\u0301 "
        "without changing their source spelling."
    )
    continuation_quote = "The final Unicode continuation reaches its independently authored endpoint."
    flags_text = " ".join(FLAGS)
    invalid_flags_text = " ".join((*FLAGS[:-1], "--flag127longer"))
    needle_quote = "Needle++ identifies the only whole literal in this collection of pages."

    documents = [
        _document("literal-early", "First Observation", [f"{early_quote} {long_padding}"]),
        _document("literal-late", "Last Observation", [f"{long_padding} {late_quote}"]),
        _document(
            "literal-unicode",
            "Unicode Continuation Notebook",
            [f"{unicode_quote} {long_padding} {continuation_quote}"],
        ),
        _document(
            "reject-larger-flag",
            "Longer Flag Notebook",
            [f"The only switch is --verify-full while verify names its ordinary anchor. {long_padding}"],
        ),
        _document(
            "reject-qualified",
            "Qualified Name Notebook",
            [f"The only qualified identifier is outer.Vault::key in this notation. {long_padding}"],
        ),
        _document(
            "reject-suffix",
            "Longer Suffix Notebook",
            [f"Tone++17 is the entire identifier while Tone remains an ordinary anchor. {long_padding}"],
        ),
        _document(
            "reject-combining-before",
            "Leading Combining Mark Notebook",
            [f"The marked spelling is \u0301Boundary++ and Boundary supplies a plain anchor. {long_padding}"],
        ),
        _document(
            "reject-combining-after",
            "Trailing Combining Mark Notebook",
            [f"The marked spelling is Boundary++\u0301 and Boundary supplies a plain anchor. {long_padding}"],
        ),
        _document(
            "literal-one-absent",
            "Missing Qualified Name Notebook",
            [f"Latch++ is present; Missing and piece appear only as separate ordinary words. {long_padding}"],
        ),
        _document("flags-early", "Early Flag Register", [f"{flags_text} {flag_padding}"]),
        _document("flags-late", "Late Flag Register", [f"{flag_padding} {flags_text}"]),
        _document(
            "flags-invalid-boundary",
            "Longer Final Flag Register",
            [f"flag127 {flag_padding} {invalid_flags_text}"],
        ),
        _document(
            "needle-ranked-last",
            "Dense Needle Pages",
            [" ".join(["Needle"] * 40) for _ in range(35)]
            + [" ".join(["quiet"] * 380) + " " + needle_quote],
            source_format="pdf",
        ),
    ]

    queries = [
        _case("literal-early-positive", "Sparkle++", "literal-early", ["Sparkle++"], quotes=[early_quote]),
        _case(
            "literal-late-positive", "Later::signal", "literal-late", ["Later::signal"],
            quotes=[late_quote], expected_excerpt_offset_positive=True,
        ),
        _case(
            "long-unicode-continuation", "Atlas::open", "literal-unicode", ["Atlas::open"],
            quotes=[unicode_quote], expected_continuation=True,
            required_continuation_quotes=[continuation_quote],
        ),
        _case("larger-flag-reject", "--verify", "reject-larger-flag", ["--verify"], no_match=True),
        _case("qualified-name-reject", "Vault::key", "reject-qualified", ["Vault::key"], no_match=True),
        _case("longer-suffix-reject", "Tone++", "reject-suffix", ["Tone++"], no_match=True),
        _case(
            "combining-before-reject", "Boundary++", "reject-combining-before", ["Boundary++"],
            no_match=True,
        ),
        _case(
            "combining-after-reject", "Boundary++", "reject-combining-after", ["Boundary++"],
            no_match=True,
        ),
        _case(
            "multi-literal-absent-last", "Latch++ Missing::piece", "literal-one-absent",
            ["Latch++", "Missing::piece"], no_match=True,
        ),
        _case(
            "multi-literal-absent-first", "Missing::piece Latch++", "literal-one-absent",
            ["Missing::piece", "Latch++"], no_match=True,
        ),
        _case(
            "128-flags-early", flags_text, "flags-early", FLAGS,
            quotes=[FLAGS[0], FLAGS[-1]],
        ),
        _case(
            "128-flags-late", flags_text, "flags-late", FLAGS,
            quotes=[FLAGS[0], FLAGS[-1]], expected_excerpt_offset_positive=True,
        ),
        _case(
            "128-flags-invalid-boundary", flags_text, "flags-invalid-boundary", FLAGS,
            no_match=True,
        ),
        _case(
            "valid-after-35-rejects", "Needle++", "needle-ranked-last", ["Needle++"],
            quotes=[needle_quote], expected_fts_candidates=36, expected_candidate_rank_minimum=36,
        ),
    ]
    return {"seed": SEED, "documents": documents, "queries": queries}


def broad_cases(variant):
    """Return scale probes for ordinary or substring-collision backgrounds.

    The caller owns background generation and replaces ordinary ``archive``
    words with ``archive++17`` only for the collision variant.  These expected
    outcomes therefore do not depend on the tested predicate implementation.
    """
    if variant not in {"plain", "collision"}:
        raise ValueError("Expected plain or collision background variant")
    cases = [
        {
            "id": "primary-archive-miss", "question": "archive++", "book_filter": None,
            "expected_no_match": True, "expected_book": None, "expected_literals": ["archive++"],
        },
        {
            "id": "ordinary-archive", "question": "archive", "book_filter": None,
            "expected_no_match": False, "expected_book": None, "expected_literals": [],
            "required_quotes": ["archive"],
        },
        {
            "id": "filtered-background-000", "question": "archive", "book_filter": "background-000",
            "expected_no_match": False, "expected_book": "background-000", "expected_literals": [],
            "required_quotes": ["archive"],
        },
    ]
    if variant == "collision":
        cases.insert(1, {
            "id": "collision-positive", "question": "archive++17", "book_filter": None,
            "expected_no_match": False, "expected_book": None, "expected_literals": ["archive++17"],
            "required_quotes": ["archive++17"],
        })
    return cases
