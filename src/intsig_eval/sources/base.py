"""Source adapters convert one reader's raw output into ``list[TokenHit]``.

Add a new source by subclassing :class:`TokenReader` and implementing
``read(stem)``. The pipeline doesn't care where hits came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from intsig_eval.core import TokenHit


class TokenReader(ABC):
    """A reader that turns one document instance into structured token hits."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def read(self, stem: str) -> list[TokenHit]:
        """Return all structured tokens this source sees in ``<stem>``.

        Hits should be tagged with ``source=self.name``.
        """

    def available_stems(self) -> set[str]:  # pragma: no cover - default impl
        return set()


def discover_stems(*readers: TokenReader) -> list[str]:
    """Intersection of stems across all readers, sorted for determinism."""
    if not readers:
        return []
    sets = [r.available_stems() for r in readers]
    common = set.intersection(*sets) if all(sets) else set()
    return sorted(common)


__all__ = ["TokenReader", "discover_stems", "Path"]
