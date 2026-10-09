"""Conservative literal technical tokens, shared by query and source matching."""

import re
from dataclasses import dataclass

NORMALIZATION_VERSION = "literal-technical-ascii-v1"
QUESTION_WORDS = {
    "what", "why", "how", "can", "could", "would", "should", "is", "are", "do", "does",
    "the", "a", "an", "about", "tell", "me", "please", "library", "book", "books", "find",
    "learn", "i", "we", "you", "to", "of", "in", "and", "for", "with",
}
# This scans maximal ASCII token candidates, including ordinary words. There are
# no nested unbounded repetitions that can repeatedly reconsider a word's prefix.
TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:--?[A-Za-z][A-Za-z0-9_-]*|"
    r"[A-Za-z0-9_]+(?:(?:::|\.)[A-Za-z0-9_]+)*(?:[+#]+[A-Za-z0-9_]*)?)"
)
ALPHANUMERIC = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class LiteralSpan:
    text: str
    start: int
    end: int


def literal_spans(text):
    """Yield recognized whole literals with offsets into the unchanged source."""
    for match in TOKEN.finditer(text):
        token = match.group()
        if not ALPHANUMERIC.search(token) or not any(char in token for char in "_-.:+#"):
            continue
        start, end = match.span()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if any(char and not char.isascii() and not char.isspace() for char in (before, after)):
            continue
        # Qualification and suffix syntax cannot be discarded at the edge of a
        # smaller candidate. A terminal prose period/colon remains a delimiter.
        if before in {".", ":", "+", "#"} or after in {"+", "#"}:
            continue
        if text[end:end + 2] == "::":
            continue
        if after == "." and end + 1 < len(text) and (text[end + 1].isascii() and (text[end + 1].isalnum() or text[end + 1] == "_")):
            continue
        if token.startswith("-") and (before == "-" or after == "-"):
            continue
        yield LiteralSpan(token, start, end)


@dataclass(frozen=True)
class QueryPlan:
    preserved_query: str
    candidate_terms: tuple[str, ...]
    literals: tuple[str, ...]

    @property
    def expression(self):
        # Terms originate only from word extraction or ASCII alphanumeric runs.
        return " AND ".join('"' + term + '"' for term in self.candidate_terms)

    def provenance(self):
        return {
            "version": NORMALIZATION_VERSION,
            "fts_candidate_expression": self.expression,
            "literal_constraints": list(self.literals),
            "literal_case_policy": "case_sensitive",
            "literal_grammar": "conservative_ascii_technical_tokens",
        }


def plan_query(text, *, drop_question_words=False):
    preserved, candidates, literals = [], [], []

    def ordinary(fragment):
        words = re.findall(r"\w+", fragment.lower() if drop_question_words else fragment)
        for word in words:
            if not drop_question_words or word not in QUESTION_WORDS:
                preserved.append(word)
                candidates.append(word)

    cursor = 0
    for literal in literal_spans(text):
        ordinary(text[cursor:literal.start])
        preserved.append(literal.text)
        candidates.extend(part.lower() for part in ALPHANUMERIC.findall(literal.text))
        if literal.text not in literals:
            literals.append(literal.text)
        cursor = literal.end
    ordinary(text[cursor:])
    return QueryPlan(" ".join(preserved), tuple(candidates), tuple(literals))


def contains_literals(text, literals):
    """Substring absence checks, then at most one maximal-token lexer pass."""
    remaining = set(literals)
    if not remaining:
        return True
    if any(literal not in text for literal in remaining):
        return False
    for literal in literal_spans(text):
        remaining.discard(literal.text)
        if not remaining:
            return True
    return False


def first_literal_match(text, literals):
    required = set(literals)
    return next((span for span in literal_spans(text) if span.text in required), None)
