"""Draw consensus verdicts onto the original image, one image per source."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from doceval.core import BBox, ImageEvaluation, SourceJudgement, Verdict
from doceval.utils import load_font


# (border RGB, label background RGBA)
_COLOR_BY_VERDICT: dict[Verdict, tuple[tuple[int, int, int], tuple[int, int, int, int]]] = {
    "hallucination": ((220, 40, 40), (220, 40, 40, 220)),       # 红
    "typo":          ((220, 100, 30), (220, 100, 30, 220)),     # 红橙
    "omission":      ((255, 150, 0), (255, 150, 0, 220)),       # 橙
    "ambiguous":     ((140, 140, 140), (140, 140, 140, 220)),   # 灰
    "correct":       ((40, 160, 60), (40, 160, 60, 220)),       # 绿（一般不画）
}


_LABEL_ZH: dict[Verdict, str] = {
    "hallucination": "幻觉",
    "typo":          "看错",
    "omission":      "漏读",
    "ambiguous":     "不明确",
    "correct":       "正确",
}


def _valid_bbox(bbox: BBox | None) -> BBox | None:
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 0.001 or y1 - y0 < 0.001:
        return None
    return (
        max(0.0, min(1.0, x0)),
        max(0.0, min(1.0, y0)),
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
    )


def _wrap_label(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    cur = ""
    for ch in text:
        if len(cur) >= max_chars and ch != " ":
            lines.append(cur)
            cur = ""
        cur += ch
    if cur:
        lines.append(cur)
    return lines[:3]


def _draw_judgements(
    image_path: Path,
    out_path: Path,
    judgements: list[SourceJudgement],
    legend: list[tuple[str, tuple[int, int, int]]],
) -> int:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    font_size = max(14, min(w, h) // 60)
    font = load_font(font_size)
    line_w = max(2, min(w, h) // 400)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    drawn = 0
    for i, j in enumerate(judgements, 1):
        bbox = _valid_bbox(j.cluster.bbox)
        if not bbox:
            continue
        px0, py0 = int(bbox[0] * w), int(bbox[1] * h)
        px1, py1 = int(bbox[2] * w), int(bbox[3] * h)
        border, label_bg = _COLOR_BY_VERDICT.get(
            j.verdict, ((100, 100, 100), (100, 100, 100, 220))
        )
        draw.rectangle([px0, py0, px1, py1], outline=border, width=line_w, fill=(*border, 40))

        label_text = f"{i}. {_LABEL_ZH[j.verdict]}: {j.surface_observed or j.canonical}"
        if j.verdict == "typo":
            label_text += f" ≠ {j.canonical}"
        label_lines = _wrap_label(label_text, max_chars=24)

        ascent, descent = font.getmetrics()
        line_h = ascent + descent + 2
        box_h = line_h * len(label_lines) + 6
        box_w = max(int(font.getlength(ln)) for ln in label_lines) + 12

        lx, ly = px0, py0 - box_h - 2
        if ly < 0:
            ly = py0 + 2
            lx = px0 + 2
        if lx + box_w > w:
            lx = max(0, w - box_w)

        draw.rectangle([lx, ly, lx + box_w, ly + box_h], fill=label_bg)
        for k, ln in enumerate(label_lines):
            draw.text((lx + 6, ly + 3 + k * line_h), ln, fill=(255, 255, 255), font=font)
        drawn += 1

    composed = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # legend strip
    legend_h = max(36, font_size * 2 + 12)
    legend_strip = Image.new("RGB", (w, legend_h), (245, 245, 245))
    ldraw = ImageDraw.Draw(legend_strip)
    x = 10
    swatch = font_size + 4
    y_top = (legend_h - swatch) // 2
    for text, color in legend:
        ldraw.rectangle(
            [x, y_top, x + swatch, y_top + swatch],
            fill=color,
            outline=(0, 0, 0),
            width=1,
        )
        ldraw.text((x + swatch + 6, y_top - 2), text, fill=(20, 20, 20), font=font)
        x += swatch + 6 + int(font.getlength(text)) + 18

    out_img = Image.new("RGB", (w, h + legend_h), (255, 255, 255))
    out_img.paste(composed, (0, 0))
    out_img.paste(legend_strip, (0, h))
    out_img.save(out_path, quality=92)
    return drawn


# Errors we want to visualize per source. We deliberately don't draw "correct"
# (it would flood the image) but we do draw "ambiguous" so reviewers can see
# what was uncertain.
_DRAWN_VERDICTS: set[Verdict] = {"hallucination", "typo", "omission", "ambiguous"}


def annotate(
    evaluation: ImageEvaluation,
    out_dir: Path,
    *,
    sources: list[str] | None = None,
) -> dict[str, int]:
    """Write ``annotated_<source>.jpg`` for each source. Returns drawn counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(evaluation.image_path)

    judgements_by_source: dict[str, list[SourceJudgement]] = {}
    for j in evaluation.judgements:
        if j.verdict not in _DRAWN_VERDICTS:
            continue
        judgements_by_source.setdefault(j.source, []).append(j)

    sources = sources or sorted(judgements_by_source.keys())
    counts: dict[str, int] = {}
    for src in sources:
        items = judgements_by_source.get(src, [])
        legend = [
            (f"{src}{_LABEL_ZH[v]}", _COLOR_BY_VERDICT[v][0])
            for v in ("hallucination", "typo", "omission", "ambiguous")
        ]
        counts[src] = _draw_judgements(
            image_path, out_dir / f"annotated_{src}.jpg", items, legend
        )
    return counts
