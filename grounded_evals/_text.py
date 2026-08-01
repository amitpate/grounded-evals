"""Shared text utilities: tokenization and numeral extraction.

Kept provider-free and dependency-free. Both the lexical alignment stage and
the mock judge use these, so their notions of "content token" and "numeral"
never drift apart.
"""
from __future__ import annotations

import re

_STOP = frozenset(
    "a an the of to in on for and or is are was were be been with by at as it its "
    "this that from into over under after before not no".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_NUMERAL_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def content_tokens(text: str) -> set[str]:
    """Lowercased tokens with stopwords and near-empty fragments removed."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOP and len(t) > 2
    }


def numerals(text: str) -> set[str]:
    """Numerals normalized for comparison: thousands separators stripped.

    "90,000" and "90000" compare equal; "315.5" stays distinct from "315".
    """
    return {n.replace(",", "") for n in _NUMERAL_RE.findall(text)}
