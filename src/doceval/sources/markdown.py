"""Read structured tokens from a markdown file.

A ``MarkdownReader`` reads ``<root>/<source_name>/<stem>.md`` and emits
``TokenHit`` objects tagged with the source name (e.g. ``"gemini"``, ``"gpt"``).
Bounding boxes are always ``None`` because markdown carries no layout info.
"""
from __future__ import annotations

from pathlib import Path

from intsig_eval.core import TokenHit, iter_token_matches, normalize, strip_markdown
from intsig_eval.sources.base import TokenReader


class MarkdownReader(TokenReader):
    def __init__(self, name: str, root: Path) -> None:
        super().__init__(name)
        self.root = Path(root)

    # -- TokenReader API ---------------------------------------------------
    def available_stems(self) -> set[str]:
        if not self.root.is_dir():
            return set()
        return {p.stem for p in self.root.glob("*.md")}

    def read(self, stem: str) -> list[TokenHit]:
        path = self.root / f"{stem}.md"
        if not path.exists():
            return []
        text = strip_markdown(path.read_text(encoding="utf-8"))

        hits: dict[tuple[str, str], TokenHit] = {}
        for m in iter_token_matches(text):
            norm = normalize(m.surface)
            if not norm:
                continue
            key = (norm, m.surface)
            if key in hits:
                continue
            hits[key] = TokenHit(source=self.name, surface=m.surface, norm=norm)
        return list(hits.values())
