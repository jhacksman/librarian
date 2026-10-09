# Literal technical queries: frozen policy v1

Policy fixed before the implementer opens the independent held-out cases. This
adds precise filtering for explicitly typed technical tokens to the existing
SQLite lexical backend. It does not add stemming, synonyms, semantic retrieval,
answer synthesis, or ranking adjustments. The existing recovery diagnostics stay
unchanged.

## Recognized forms and matching

Literal recognition precedes lowercasing and question-word removal. V1 recognizes
ASCII tokens with at least one letter or digit in these forms:

- Short or long flags: one or two leading hyphens, an ASCII letter, then ASCII
  letters, digits, underscores or hyphens.
- Underscore identifiers, including leading/trailing underscores.
- Dot- or double-colon-qualified chains of ASCII word components, including
  numeric components. A trailing prose period is not part of a chain.
- ASCII word tokens with a run of `+` or `#` suffix characters and an optional
  ASCII word suffix, including version suffixes.

Literal comparisons are case-sensitive. Ordinary words retain existing Unicode
word extraction, lowercasing and filler-word handling. Components inside a
literal are never removed as filler words. A semantic adapter still receives the
original question. Unsupported syntax receives ordinary lexical handling; v1
does not claim exact matching for arbitrary programming-language syntax or
Unicode identifiers.

The same lexer recognizes maximal technical tokens in questions and source text.
A flag prefix cannot match a longer flag; an identifier cannot match a longer or
more qualified token. Parentheses, brackets, braces, quotes, commas, semicolons,
assignment signs, and ordinary terminal punctuation can surround tokens. A
qualification separator or suffix operator adjacent to a token cannot be silently
discarded to create a shorter literal. A flag cannot start inside a longer run
of hyphens.

At a token's outer boundary, an adjacent non-ASCII non-whitespace character is
conservatively excluded. This deliberate v1 restriction prevents a Python/SQLite
Unicode-tokenizer mismatch from creating an accepted literal match that FTS
cannot retrieve. Unicode elsewhere in a passage remains intact, and offsets are
always measured in the original extracted Unicode string. Supporting a broader
Unicode grammar requires a separate reviewed policy and evaluation.

## Candidate generation and filtering

Each literal contributes its ASCII alphanumeric runs as required FTS anchors.
Every accepted literal occurrence contains these runs separated at compatible
token boundaries. Ordinary query terms retain their existing candidate behavior.
All candidate terms are quoted and joined with AND in a parameterized MATCH
expression. This relies on the existing default
[SQLite unicode61 tokenizer](https://www.sqlite.org/fts5.html#unicode61_tokenizer),
where ASCII technical punctuation separates alphanumeric tokens. It does not
assume that Python's Unicode word definition equals SQLite's.

A pure, connection-local SQLite predicate checks all literal constraints against
each candidate's original content. Exact substring absence can reject a candidate
before lexing; substring presence still requires at most one maximal-token lexer
pass. These checks run before
the final hit limit. The book predicate, BM25 order and chunk-ID tie-breaker remain
in place. There is no hidden top-candidate cap or partial-result fallback. An
execution failure remains an error, never a no-match abstention. V1 supports no
literal without a usable alphanumeric anchor.

Returned provenance records the preserved retrieval query, actual FTS candidate
expression, literal constraints, literal case policy and normalization version.
Normalization version is separate from extraction/chunking version. Stored index
schema, passage identities and citation offsets do not change; no reindexing is
required.

When a long passage needs clipping, prefer an actual matched literal occurrence
identified by the same source lexer. Never find offsets on a lowercased copy.
One bounded excerpt need not contain every conjunct; continuation retains access
to the rest of the matching chunk.

## Verification contract

Keep development labels frozen. Freeze the application candidate before opening
the independently authored cases; record their predeclared digest and report all
outcomes. Do not relabel failures to obtain a pass. These are independent synthetic
development checks, not a human-authored or representative semantic benchmark.

Verify technical-token prefix/suffix and case distinctions, ordinary-word and
filter compatibility, names containing question words, multiple literals, Unicode
offsets, long excerpts, and a valid result below many rejected FTS candidates.
Measure predicate cost on the bounded Spark corpus. Tests and measurements run
only on Spark with the existing reviewed dependencies. No private corpus,
model, source upload, LAN service or publication is involved.
