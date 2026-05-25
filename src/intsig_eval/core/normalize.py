"""Surface → normalized form, plus capped Levenshtein distance.

Both functions are tight little utilities that the OCR reader, the Markdown
reader, and the consensus stage all depend on. Keeping them here means
**there is exactly one definition of equality between tokens** in the project.
"""
from __future__ import annotations

import re
import unicodedata

_STRIP = re.compile(r"[\s,]")


def normalize(surface: str) -> str:
    """Map raw text into a canonical comparable form.

    - NFKC unicode normalization (全角 → 半角)
    - Uppercase
    - Drop whitespace and commas (so ``5,555 5777`` → ``55555777``)
    - Strip leading/trailing punctuation that often comes from MD context
    """
    t = unicodedata.normalize("NFKC", surface).upper().strip()
    t = _STRIP.sub("", t)
    return t.strip(".-/:;")


def edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Capped Levenshtein distance.

    Returns ``cap + 1`` as soon as the partial distance exceeds ``cap``,
    which makes it cheap to use as a near-miss check inside tight loops.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1

    # Ensure a is the shorter string for memory efficiency.
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        cur = [i] + [0] * len(a)
        row_min = cur[0]
        for j, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost, # substitution
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > cap:
            return cap + 1
        prev = cur
    return prev[-1]
