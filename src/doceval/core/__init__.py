"""Pure logic. No I/O, no Azure, no Agent Framework — easy to unit test."""
from doceval.core.normalize import edit_distance, normalize
from doceval.core.tokens import (
    PATTERNS,
    PATTERNS_RELAXED,
    TokenMatch,
    iter_token_matches,
    strip_markdown,
)
from doceval.core.types import (
    BBox,
    Cluster,
    ImageEvaluation,
    SourceJudgement,
    SourceName,
    TokenHit,
    Verdict,
)

__all__ = [
    "BBox",
    "Cluster",
    "ImageEvaluation",
    "PATTERNS",
    "PATTERNS_RELAXED",
    "SourceJudgement",
    "SourceName",
    "TokenHit",
    "TokenMatch",
    "Verdict",
    "edit_distance",
    "iter_token_matches",
    "normalize",
    "strip_markdown",
]
