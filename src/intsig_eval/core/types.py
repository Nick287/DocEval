"""Shared dataclasses used across the pipeline.

The pipeline is deliberately oriented around a single value object — :class:`TokenHit` —
that uniformly represents "source S saw this token at location L on page P".
Both the OCR reader and the Markdown reader emit ``list[TokenHit]``.

A :class:`Cluster` groups together hits that refer to the same underlying datum
(possibly with character-level disagreements between sources).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]
"""Normalized axis-aligned bounding box ``(x0, y0, x1, y1)``, each in [0, 1]."""

SourceName = str
"""Free-form source identifier — e.g. ``"ocr"``, ``"gemini"``, ``"gpt"``."""

Verdict = Literal["correct", "typo", "omission", "hallucination", "ambiguous"]
"""Per-source verdict against a cluster's canonical form."""


@dataclass(frozen=True, slots=True)
class TokenHit:
    """A single observation of a structured token by one source."""

    source: SourceName
    surface: str                 # the raw text as the source wrote it
    norm: str                    # normalized form (uppercase, no separators)
    bbox: BBox | None = None     # only OCR populates this
    confidence: float | None = None


@dataclass(slots=True)
class Cluster:
    """A group of hits that refer to the same underlying token.

    Membership is decided by edit distance on ``norm``. A cluster typically
    contains 1 or 2 hits per source. ``canonical_norm`` is derived by majority
    vote across all hits.
    """

    members: list[TokenHit] = field(default_factory=list)
    canonical_norm: str = ""
    canonical_surface: str = ""
    bbox: BBox | None = None

    @property
    def sources(self) -> set[SourceName]:
        return {h.source for h in self.members}

    def hits_for(self, source: SourceName) -> list[TokenHit]:
        return [h for h in self.members if h.source == source]


@dataclass(slots=True)
class SourceJudgement:
    """Per-cluster, per-source verdict produced by the consensus stage."""

    source: SourceName
    cluster: Cluster
    verdict: Verdict
    surface_observed: str | None  # what THIS source wrote (None for omission)
    canonical: str                # canonical surface for comparison
    distance: int = 0             # edit distance to canonical (0 for correct/omission)
    evidence: str = ""            # free-form explanation (from agent if used)


@dataclass(slots=True)
class ImageEvaluation:
    """End-to-end evaluation result for one image."""

    stem: str
    image_path: str
    clusters: list[Cluster]
    judgements: list[SourceJudgement]
    elapsed_seconds: float = 0.0
    verifier_model: str | None = None
    """Served model string returned by the vision verifier, e.g.
    ``gpt-5.4-2025-09-xx``. ``None`` when the verifier was disabled."""

    def by_source(self, source: SourceName) -> list[SourceJudgement]:
        return [j for j in self.judgements if j.source == source]
