from __future__ import annotations

import re

ROUTE_ORDER = (
    "hard_text",
    "label_text",
    "numeric_text",
    "counting",
    "chart_table",
    "coarse",
)

_BINARY_RE = re.compile(r"^(is|are|does|do|did|was|were|can|could|will|would|has|have|had)\b")
_SPACE_RE = re.compile(r"\s+")

_CHART_HINTS = ("chart", "graph", "table", "plot")
_COUNT_HINTS = ("how many", "count", "number of")
_HARD_TEXT_HINTS = (
    "small",
    "spell",
    "written",
    "write",
    "says",
    "say",
    "text",
    "word",
    "letter",
    "read",
    "screen",
    "sticker",
    "paragraph",
    "page",
    "menu",
    "caption",
    "website",
    "web site",
    "phone",
    "license",
    "plate",
    "serial",
    "barcode",
    "qr code",
    "what does",
    "what is written",
    "what is the first",
    "what is the last",
    "what is the top",
    "what is the bottom",
)
_LABEL_HINTS = (
    "brand",
    "name",
    "title",
    "logo",
    "author",
    "photographer",
    "publisher",
    "sponsor",
    "company",
    "manufacturer",
    "make",
    "model",
    "kind of",
)
_NUMERIC_HINTS = (
    "time",
    "date",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "percent",
    "price",
    "cost",
    "calories",
    "score",
    "age",
    "how much",
    "what number",
    "what is the number",
)


def normalize_question(text: str) -> str:
    return _SPACE_RE.sub(" ", text.lower()).strip()


def question_route(text: str) -> str:
    q = normalize_question(text)
    if not q:
        return "coarse"
    if any(hint in q for hint in _CHART_HINTS):
        return "chart_table"
    if "how many" in q or q.startswith("count ") or " number of " in f" {q} ":
        return "counting"

    binary = bool(_BINARY_RE.match(q))
    if any(hint in q for hint in _HARD_TEXT_HINTS):
        return "hard_text"
    if any(hint in q for hint in _NUMERIC_HINTS) or q.startswith("when "):
        return "numeric_text"
    if any(hint in q for hint in _LABEL_HINTS):
        return "label_text"
    if binary:
        return "coarse"
    return "coarse"
