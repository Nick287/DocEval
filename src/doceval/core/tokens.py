"""Single source of truth for what counts as a "structured token".

Both the OCR reader and the Markdown reader call :func:`iter_token_matches`
so they agree byte-for-byte on which substrings should be considered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---- patterns --------------------------------------------------------------
# fmt: off
PATTERNS: dict[str, re.Pattern[str]] = {
    "long_number":   re.compile(r"\b\d{4,}(?:[.,]\d+)?\b"),
    "alnum_id":      re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b"),
    "mixed_id":      re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9-]*\b|\b\d+[A-Za-z]+[A-Za-z0-9-]*\b"),
    "currency":      re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    "date_compact":  re.compile(r"\b\d{1,2}[A-Z]{3}\d{2,4}\b"),
    "date_dmy":      re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"),
    "date_ymd":      re.compile(r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"),
    "date_dmonY":    re.compile(r"\b\d{1,2}-[A-Z][a-z]{2}-\d{2,4}\b"),
}
# fmt: on


# Same patterns but with ``\b`` removed; used on `"".join(words)` where word
# boundaries rarely fall inside long digit runs that OCR split apart.
PATTERNS_RELAXED: dict[str, re.Pattern[str]] = {
    name: re.compile(p.pattern.replace(r"\b", "").replace(r"\B", ""))
    for name, p in PATTERNS.items()
}


# ---- markdown sanitization -------------------------------------------------
_MD_NOISE = re.compile(
    r"<figure>[^<]*</figure>"   # figure placeholders
    r"|```[\s\S]*?```"            # fenced code
    r"|`[^`]*`"                    # inline code
    r"|!?\[[^\]]*\]\([^)]*\)"      # markdown links and images
)


def strip_markdown(text: str) -> str:
    """Remove markdown syntax that would otherwise pollute token extraction."""
    text = _MD_NOISE.sub(" ", text)
    text = re.sub(r"^\s*#{1,6}\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~|>]", " ", text)
    return text


# ---- extraction ------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TokenMatch:
    """A regex hit. ``span`` is in the input string's coordinate system."""

    surface: str
    span: tuple[int, int]
    pattern_name: str


def iter_token_matches(text: str, *, relaxed: bool = False) -> list[TokenMatch]:
    """Find all structured tokens in ``text``.

    With ``relaxed=True`` the regexes drop word boundaries; that mode is meant
    for the OCR "no-space" join pass where boundaries don't fire reliably.
    """
    patterns = PATTERNS_RELAXED if relaxed else PATTERNS
    hits: list[TokenMatch] = []
    for name, pat in patterns.items():
        for m in pat.finditer(text):
            hits.append(TokenMatch(surface=m.group(0), span=m.span(), pattern_name=name))
    return hits
