"""Decide a canonical form per cluster and a verdict per source.

Voting rule
-----------
For each cluster:

1. **Tally normalized forms** across all sources (one vote per *source*,
   not per hit, so a source seeing the same number twice doesn't dominate).
2. The form with the most votes wins. Ties are broken in this order:
   - OCR's form (it observed pixels, not text)
   - longest form (fewer truncations)
   - lexicographic
3. The canonical *surface* is picked from one of the hits whose norm equals
   the canonical norm. OCR is preferred when present.
4. The canonical *bbox* is taken from OCR's hit if any cluster member is OCR.

Per-source verdict per cluster
------------------------------
- Source has a hit whose norm == canonical_norm  → ``correct``
- Source has a hit whose norm differs (≤ ``max_distance``) → ``typo``
  (with ``distance`` = edit distance to canonical)
- Source has **no hit** in the cluster *and* cluster has ≥ 2 sources
  agreeing → ``omission``
- Singleton clusters (1 source only) yield ``hallucination`` for that single
  source (unless the verifier later confirms it; that's the next stage).
"""
from __future__ import annotations

from collections import Counter

from intsig_eval.core import (
    Cluster,
    SourceJudgement,
    SourceName,
    TokenHit,
    edit_distance,
)


def _pick_canonical_norm(cluster: Cluster, ocr_source: SourceName) -> str:
    """Vote on the canonical normalized form (one vote per source)."""
    # source → set of norms that source produced in this cluster
    by_source: dict[SourceName, set[str]] = {}
    for h in cluster.members:
        by_source.setdefault(h.source, set()).add(h.norm)

    # source-weighted tally
    tally: Counter[str] = Counter()
    for norms in by_source.values():
        for n in norms:
            tally[n] += 1

    top = tally.most_common()
    if not top:
        return ""
    max_votes = top[0][1]
    contenders = [n for n, v in top if v == max_votes]
    if len(contenders) == 1:
        return contenders[0]

    # Tie break ----------------------------------------------------------------
    ocr_norms = by_source.get(ocr_source, set())
    for n in contenders:
        if n in ocr_norms:
            return n
    contenders.sort(key=lambda n: (-len(n), n))
    return contenders[0]


def _pick_canonical_surface(
    cluster: Cluster, canonical_norm: str, ocr_source: SourceName
) -> tuple[str, tuple[float, float, float, float] | None]:
    """Pick the surface form to display and the bbox to draw, preferring OCR."""
    ocr_candidates = [
        h for h in cluster.members if h.source == ocr_source and h.norm == canonical_norm
    ]
    if ocr_candidates:
        h = ocr_candidates[0]
        return h.surface, h.bbox

    matching = [h for h in cluster.members if h.norm == canonical_norm]
    if matching:
        h = matching[0]
        # bbox can still come from any OCR hit in the cluster (even with a
        # different norm — e.g. typo case) — we'd rather show the wrong location
        # than no location.
        ocr_any = next((m for m in cluster.members if m.source == ocr_source and m.bbox), None)
        return h.surface, (ocr_any.bbox if ocr_any else h.bbox)

    h = cluster.members[0]
    return h.surface, h.bbox


def finalize_cluster(cluster: Cluster, *, ocr_source: SourceName = "ocr") -> Cluster:
    """Populate ``canonical_norm``, ``canonical_surface`` and ``bbox`` in place."""
    cluster.canonical_norm = _pick_canonical_norm(cluster, ocr_source)
    surface, bbox = _pick_canonical_surface(cluster, cluster.canonical_norm, ocr_source)
    cluster.canonical_surface = surface
    cluster.bbox = bbox
    return cluster


def judge_cluster(
    cluster: Cluster,
    all_sources: list[SourceName],
    *,
    cap: int = 2,
) -> list[SourceJudgement]:
    """Emit one :class:`SourceJudgement` per source per cluster.

    Singleton (1-source) clusters are tentatively flagged ``hallucination`` —
    the optional vision verifier may later upgrade them to ``correct``.
    """
    assert cluster.canonical_norm, "finalize_cluster() must be called first"

    sources_present = cluster.sources
    is_singleton = len(sources_present) <= 1

    judgements: list[SourceJudgement] = []
    for src in all_sources:
        hits = cluster.hits_for(src)
        if not hits:
            if is_singleton:
                # nothing to say about this source — it just didn't see a
                # singleton's lonely token; skip rather than emit "omission".
                continue
            judgements.append(
                SourceJudgement(
                    source=src,
                    cluster=cluster,
                    verdict="omission",
                    surface_observed=None,
                    canonical=cluster.canonical_surface,
                )
            )
            continue

        # take the first hit whose norm matches canonical, else the first hit
        best = next((h for h in hits if h.norm == cluster.canonical_norm), hits[0])

        if best.norm == cluster.canonical_norm:
            judgements.append(
                SourceJudgement(
                    source=src,
                    cluster=cluster,
                    verdict="hallucination" if is_singleton else "correct",
                    surface_observed=best.surface,
                    canonical=cluster.canonical_surface,
                )
            )
        else:
            dist = edit_distance(best.norm, cluster.canonical_norm, cap=cap)
            judgements.append(
                SourceJudgement(
                    source=src,
                    cluster=cluster,
                    verdict="typo",
                    surface_observed=best.surface,
                    canonical=cluster.canonical_surface,
                    distance=dist,
                )
            )
    return judgements


def vote(
    clusters: list[Cluster],
    all_sources: list[SourceName],
    *,
    ocr_source: SourceName = "ocr",
    cap: int = 2,
) -> tuple[list[Cluster], list[SourceJudgement]]:
    """Run :func:`finalize_cluster` then :func:`judge_cluster` over the list."""
    judgements: list[SourceJudgement] = []
    for c in clusters:
        finalize_cluster(c, ocr_source=ocr_source)
        judgements.extend(judge_cluster(c, all_sources, cap=cap))
    return clusters, judgements


# Helper used by the agent verifier to update a judgement after vision check.
def apply_vision_verdict(
    judgement: SourceJudgement,
    visible: bool,
    evidence: str = "",
) -> None:
    """Mutate ``judgement`` based on a vision verifier outcome.

    Only meaningful for singleton-source judgements that were tentatively
    flagged ``hallucination``. If the verifier says the token IS visible in
    the image, the verdict becomes ``correct``.
    """
    if judgement.verdict != "hallucination":
        return
    judgement.evidence = evidence
    if visible:
        judgement.verdict = "correct"
