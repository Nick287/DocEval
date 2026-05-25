"""OCR reader powered by Azure Document Intelligence ``prebuilt-layout``.

This module is structured so that the **expensive part (the network call)**
is isolated behind a content-addressed file cache (``ocr_cache_dir``). The
token extraction step that follows is pure and re-runs instantly.

Hits emitted from here always carry a real pixel-accurate ``bbox``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from doceval.config import get_settings
from doceval.core import (
    BBox,
    TokenHit,
    iter_token_matches,
    normalize,
)
from doceval.sources.base import TokenReader


# ---------------------------------------------------------------------------
# Pure helpers (testable without Azure)
# ---------------------------------------------------------------------------
def _polygon_to_bbox(polygon: list[float], w: float, h: float) -> BBox:
    """8-float polygon → normalized axis-aligned [x0,y0,x1,y1]."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (
        max(0.0, min(1.0, min(xs) / w)),
        max(0.0, min(1.0, min(ys) / h)),
        max(0.0, min(1.0, max(xs) / w)),
        max(0.0, min(1.0, max(ys) / h)),
    )


def _union_bbox(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _join_with_index(words: list[dict], sep: str) -> tuple[str, list[int]]:
    """Concatenate word texts and return a character-position → word-index map."""
    parts: list[str] = []
    char_to_word: list[int] = []
    for i, w in enumerate(words):
        text = w["text"]
        if i > 0 and sep:
            parts.append(sep)
            char_to_word.extend([-1] * len(sep))
        parts.append(text)
        char_to_word.extend([i] * len(text))
    return "".join(parts), char_to_word


def _word_indices_for_span(idx_map: list[int], start: int, end: int) -> list[int]:
    out: list[int] = []
    last = -2
    for ch in range(start, end):
        wi = idx_map[ch]
        if wi >= 0 and wi != last:
            out.append(wi)
            last = wi
    return out


def extract_token_hits_from_words(
    words: list[dict],
    source_name: str = "ocr",
) -> list[TokenHit]:
    """Two-pass token extraction over OCR words.

    Pass 1 joins words with a single space (handles tokens contained in one
    word or across whitespace-separated cells).
    Pass 2 joins with no separator (handles tokens that OCR fragmented into
    multiple words, like credit-card groups).
    """
    if not words:
        return []

    out: dict[tuple[str, tuple[int, ...]], TokenHit] = {}

    def _record(surface: str, indices: list[int]) -> None:
        norm = normalize(surface)
        if not norm or not indices:
            return
        bbox = _union_bbox([words[i]["bbox_norm"] for i in indices])
        key = (norm, tuple(indices))
        if key in out:
            return
        confidence = min(
            (words[i].get("confidence", 1.0) for i in indices), default=None
        )
        out[key] = TokenHit(
            source=source_name,
            surface=surface,
            norm=norm,
            bbox=bbox,
            confidence=confidence,
        )

    spaced, idx_spaced = _join_with_index(words, sep=" ")
    for m in iter_token_matches(spaced, relaxed=False):
        idx = _word_indices_for_span(idx_spaced, *m.span)
        _record(m.surface, idx)

    stripped, idx_stripped = _join_with_index(words, sep="")
    for m in iter_token_matches(stripped, relaxed=True):
        idx = _word_indices_for_span(idx_stripped, *m.span)
        # skip matches already fully contained in one word — pass 1 caught them
        if len(idx) <= 1:
            continue
        _record(m.surface, idx)

    return list(out.values())


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class AzureLayoutOCRReader(TokenReader):
    """OCR reader backed by Azure Document Intelligence prebuilt-layout.

    Pure file caching keyed by ``(stem, SHA-256(image bytes)[:16])`` keeps
    repeat runs free.
    """

    def __init__(
        self,
        name: str = "ocr",
        image_dir: Path | None = None,
        cache_dir: Path | None = None,
        client: DocumentIntelligenceClient | None = None,
    ) -> None:
        super().__init__(name)
        s = get_settings()
        self.image_dir = Path(image_dir or s.image_dir)
        self.cache_dir = Path(cache_dir or s.ocr_cache_dir)
        self._client = client

    # -- TokenReader API ---------------------------------------------------
    def available_stems(self) -> set[str]:
        if not self.image_dir.is_dir():
            return set()
        return {
            p.stem
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

    def find_image(self, stem: str) -> Path | None:
        for ext in ("jpg", "jpeg", "png"):
            p = self.image_dir / f"{stem}.{ext}"
            if p.exists():
                return p
        return None

    def read(self, stem: str) -> list[TokenHit]:
        image_path = self.find_image(stem)
        if not image_path:
            return []
        result = self.analyze(image_path)
        return extract_token_hits_from_words(result["words"], self.name)

    # -- Network + cache ---------------------------------------------------
    def analyze(self, image_path: Path, *, force: bool = False) -> dict[str, Any]:
        cache_path = self._cache_path(image_path)
        if not force and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        client = self._get_client()
        with open(image_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                content_type="application/octet-stream",
            )
        result = poller.result()
        out = self._serialize(result)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

    # -- internals ---------------------------------------------------------
    def _cache_path(self, image_path: Path) -> Path:
        digest = self._image_hash(image_path)
        return self.cache_dir / f"{image_path.stem}.{digest}.json"

    @staticmethod
    def _image_hash(image_path: Path) -> str:
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _get_client(self) -> DocumentIntelligenceClient:
        if self._client is None:
            s = get_settings()
            if not s.di_key:
                raise RuntimeError(
                    "DOCEVAL_DI_KEY is not set; cannot call Document Intelligence."
                )
            self._client = DocumentIntelligenceClient(
                endpoint=s.di_endpoint,
                credential=AzureKeyCredential(s.di_key),
            )
        return self._client

    @staticmethod
    def _serialize(result: Any) -> dict[str, Any]:
        if not result.pages:
            return {"page_width": 0, "page_height": 0, "words": [], "lines": []}
        page = result.pages[0]
        w = float(page.width or 1.0)
        h = float(page.height or 1.0)

        words = []
        for word in (page.words or []):
            poly = list(word.polygon or [])
            if len(poly) != 8:
                continue
            words.append(
                {
                    "text": word.content,
                    "bbox_norm": _polygon_to_bbox(poly, w, h),
                    "confidence": float(word.confidence or 0.0),
                }
            )

        lines = []
        for line in (page.lines or []):
            poly = list(line.polygon or [])
            if len(poly) != 8:
                continue
            lines.append(
                {"text": line.content, "bbox_norm": _polygon_to_bbox(poly, w, h)}
            )

        return {
            "page_width": w,
            "page_height": h,
            "unit": getattr(page, "unit", None),
            "words": words,
            "lines": lines,
        }
