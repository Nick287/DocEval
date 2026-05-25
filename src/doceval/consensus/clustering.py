"""Cluster TokenHits that refer to the same underlying token.

Two hits belong to the same cluster when:

  * they share the same ``norm`` exactly, **or**
  * one normalized form is within ``max_distance`` edits of the other

We use a Union-Find over the unique normalized forms, then fold every hit
sharing a parent into one :class:`Cluster`. The check is intentionally
asymmetric-friendly: we only merge across sources, never within one source's
own near-misses (those are usually distinct entities).
"""
from __future__ import annotations

from collections.abc import Iterable

from intsig_eval.core import Cluster, TokenHit, edit_distance


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, items: Iterable[str]) -> None:
        self.parent: dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_clusters(hits: list[TokenHit], *, max_distance: int = 1) -> list[Cluster]:
    """Group hits by ``norm`` proximity. Returns clusters sorted by canonical_norm.

    Canonicalization (canonical_norm / canonical_surface / bbox) is **not** done
    here — see :mod:`intsig_eval.consensus.voting`. We only handle membership.
    """
    if not hits:
        return []

    # collect unique norms and the sources that produced each
    norm_sources: dict[str, set[str]] = {}
    for h in hits:
        norm_sources.setdefault(h.norm, set()).add(h.source)

    uf = _UnionFind(norm_sources.keys())
    norms = sorted(norm_sources.keys(), key=len)

    # cross-source merge only — within a single source two near-miss norms are
    # likely distinct entities (e.g. two different invoice numbers that happen
    # to differ by one digit).
    for i, a in enumerate(norms):
        for b in norms[i + 1 :]:
            if abs(len(a) - len(b)) > max_distance:
                continue
            # require at least one source to differ — same source pair → skip
            if norm_sources[a].isdisjoint(norm_sources[b]) or (
                norm_sources[a] | norm_sources[b]
            ) - (norm_sources[a] & norm_sources[b]):
                if edit_distance(a, b, cap=max_distance) <= max_distance:
                    uf.union(a, b)

    groups: dict[str, list[TokenHit]] = {}
    for h in hits:
        groups.setdefault(uf.find(h.norm), []).append(h)

    clusters = [Cluster(members=ms) for ms in groups.values()]
    clusters.sort(key=lambda c: min(h.norm for h in c.members))
    return clusters
