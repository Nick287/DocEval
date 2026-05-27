"""Cross-image summary CSV + markdown."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from doceval.core import ImageEvaluation, Verdict

_VERDICTS: list[Verdict] = ["correct", "typo", "omission", "hallucination", "ambiguous"]
_VERDICT_ZH = {
    "correct": "命中",
    "typo": "看错",
    "omission": "漏读",
    "hallucination": "幻觉",
    "ambiguous": "不明确",
}


@dataclass
class SkippedEntry:
    """One stem that was attempted but didn't make it into ``evaluations``."""

    stem: str
    stage: str  # 'preflight' | 'run_all' | 'discovery'
    reason: str


def _stats_for(evaluation: ImageEvaluation, source: str) -> dict[str, int]:
    out = {v: 0 for v in _VERDICTS}
    for j in evaluation.judgements:
        if j.source == source:
            out[j.verdict] += 1
    return out


def write_summary(
    evaluations: list[ImageEvaluation],
    out_dir: Path,
    sources: list[str],
    *,
    skipped: list[SkippedEntry] | None = None,
) -> None:
    """Write summary.csv + summary.md aggregating ``evaluations``.

    ``skipped`` lists stems that were known but excluded from the run
    (preflight failures, dropped due to missing sources, etc.) and is
    rendered as a dedicated section so the user can reconcile counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = list(skipped or [])

    rows: list[dict] = []
    for ev in evaluations:
        row: dict[str, str | int | float] = {
            "stem": ev.stem,
            "clusters": len(ev.clusters),
            "elapsed_s": round(ev.elapsed_seconds, 1),
        }
        for src in sources:
            stats = _stats_for(ev, src)
            for v in _VERDICTS:
                row[f"{src}_{_VERDICT_ZH[v]}"] = stats[v]
        rows.append(row)

    if not rows:
        if not skipped:
            return
        # Even without evaluations, still write a summary that surfaces the
        # skipped stems so the user has something to inspect.
        md_lines: list[str] = [
            "# 共识评估总结",
            "",
            "⚠️ 本次运行未产生任何评估结果。",
            "",
            "## 跳过/失败的图像",
            "",
            "| stem | 阶段 | 原因 |",
            "|---|---|---|",
            *[f"| {sk.stem} | {sk.stage} | {sk.reason} |" for sk in skipped],
            "",
        ]
        (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        return

    # CSV
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md_lines: list[str] = ["# 共识评估总结", ""]

    # --- run config (verifier model versions actually seen) ----------------
    models = sorted({ev.verifier_model for ev in evaluations if ev.verifier_model})
    if models:
        md_lines += [
            "## 评估配置",
            "",
            "- 视觉验证模型：" + ", ".join(f"`{m}`" for m in models),
            "",
        ]
    else:
        md_lines += [
            "## 评估配置",
            "",
            "- 视觉验证：已关闭 (`--no-verify`)",
            "",
        ]

    headers = list(rows[0].keys())
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    totals: dict[str, int | float] = {h: 0 for h in headers if h != "stem"}
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
        for h, v in r.items():
            if h == "stem":
                continue
            try:
                totals[h] += float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
    md_lines.append(
        "| **合计** | " + " | ".join(
            f"{totals[h]:.1f}" if h == "elapsed_s" else str(int(totals[h]))
            for h in headers
            if h != "stem"
        ) + " |"
    )

    # Aggregate per-source recall/precision
    md_lines.append("\n## 各来源累计指标\n")
    md_lines.append("| 来源 | 命中 | 漏读 | 看错 | 幻觉 | 不明确 | 召回率 | 准确率 |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for src in sources:
        c = sum(_stats_for(ev, src)["correct"] for ev in evaluations)
        omi = sum(_stats_for(ev, src)["omission"] for ev in evaluations)
        typ = sum(_stats_for(ev, src)["typo"] for ev in evaluations)
        hal = sum(_stats_for(ev, src)["hallucination"] for ev in evaluations)
        amb = sum(_stats_for(ev, src)["ambiguous"] for ev in evaluations)
        seen = c + typ + hal + amb  # tokens this source actually wrote
        truth = c + omi + typ       # tokens the consensus says exist that this source had a chance to write
        recall = f"{(c * 100 / truth):.1f}%" if truth else "—"
        precision = f"{(c * 100 / seen):.1f}%" if seen else "—"
        md_lines.append(
            f"| {src} | {c} | {omi} | {typ} | {hal} | {amb} | {recall} | {precision} |"
        )

    md_lines.append("\n## 每张图详细报告\n")
    for ev in evaluations:
        md_lines.append(f"- **{ev.stem}** — [report.md]({ev.stem}/report.md)")

    if skipped:
        md_lines.append("\n## 跳过/失败的图像\n")
        md_lines.append("| stem | 阶段 | 原因 |")
        md_lines.append("|---|---|---|")
        for sk in skipped:
            md_lines.append(f"| {sk.stem} | {sk.stage} | {sk.reason} |")

    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
