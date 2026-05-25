"""Per-image markdown report + clusters.json snapshot."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from doceval.core import Cluster, ImageEvaluation, SourceJudgement, Verdict


def _verdict_counts(judgements: list[SourceJudgement]) -> dict[str, dict[str, int]]:
    by_source: dict[str, Counter[Verdict]] = {}
    for j in judgements:
        by_source.setdefault(j.source, Counter())[j.verdict] += 1
    return {src: dict(c) for src, c in by_source.items()}


def _cluster_to_dict(c: Cluster) -> dict:
    return {
        "canonical_norm": c.canonical_norm,
        "canonical_surface": c.canonical_surface,
        "bbox": list(c.bbox) if c.bbox else None,
        "members": [asdict(h) for h in c.members],
        "sources": sorted(c.sources),
    }


def _judgement_to_dict(j: SourceJudgement) -> dict:
    return {
        "source": j.source,
        "verdict": j.verdict,
        "canonical_norm": j.cluster.canonical_norm,
        "canonical_surface": j.canonical,
        "surface_observed": j.surface_observed,
        "distance": j.distance,
        "evidence": j.evidence,
        "bbox": list(j.cluster.bbox) if j.cluster.bbox else None,
    }


def write_clusters_json(evaluation: ImageEvaluation, out_path: Path) -> None:
    payload = {
        "stem": evaluation.stem,
        "image": evaluation.image_path,
        "elapsed_s": round(evaluation.elapsed_seconds, 2),
        "verifier_model": evaluation.verifier_model,
        "stats": _verdict_counts(evaluation.judgements),
        "clusters": [_cluster_to_dict(c) for c in evaluation.clusters],
        "judgements": [_judgement_to_dict(j) for j in evaluation.judgements],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_VERDICT_ZH: dict[Verdict, str] = {
    "correct": "正确",
    "typo": "看错",
    "omission": "漏读",
    "hallucination": "幻觉",
    "ambiguous": "不明确",
}


def write_report(evaluation: ImageEvaluation, out_path: Path) -> None:
    stats = _verdict_counts(evaluation.judgements)
    sources = sorted(stats.keys())

    lines: list[str] = [
        f"# {evaluation.stem} — 共识评估报告",
        "",
        f"耗时：{evaluation.elapsed_seconds:.1f}s ｜ 共 {len(evaluation.clusters)} 个 token 簇",
    ]
    if evaluation.verifier_model:
        lines.append(f"视觉验证模型：`{evaluation.verifier_model}`")
    elif evaluation.verifier_model is None:
        # explicitly mark when verifier was off, easier to audit later
        pass
    lines += [
        "",
        "## 各来源得分",
        "",
        "| 来源 | " + " | ".join(_VERDICT_ZH[v] for v in ("correct", "typo", "omission", "hallucination", "ambiguous")) + " |",
        "|" + "|".join(["---"] * 6) + "|",
    ]
    for src in sources:
        c = stats[src]
        lines.append(
            "| "
            + src
            + " | "
            + " | ".join(
                str(c.get(v, 0))
                for v in ("correct", "typo", "omission", "hallucination", "ambiguous")
            )
            + " |"
        )

    # Per-cluster table — only show clusters that have at least one non-correct
    # judgement, otherwise the report explodes for big images.
    interesting = [
        c
        for c in evaluation.clusters
        if any(
            j.verdict != "correct"
            for j in evaluation.judgements
            if j.cluster is c
        )
    ]
    if interesting:
        lines += ["", "## 有分歧的 token 簇", "", "| 规范化 | " + " | ".join(sources) + " | 位置 |", "|" + "|".join(["---"] * (len(sources) + 2)) + "|"]
        for c in interesting:
            row_cells: list[str] = []
            for src in sources:
                jud = next(
                    (j for j in evaluation.judgements if j.cluster is c and j.source == src),
                    None,
                )
                if jud is None or jud.verdict == "correct":
                    row_cells.append(jud.surface_observed if jud and jud.surface_observed else "✓" if jud else "")
                else:
                    label = _VERDICT_ZH[jud.verdict]
                    obs = jud.surface_observed or "—"
                    row_cells.append(f"{label}: `{obs}`")
            bbox_str = "—"
            if c.bbox:
                bbox_str = "[" + ", ".join(f"{v:.2f}" for v in c.bbox) + "]"
            lines.append(
                f"| `{c.canonical_surface}` | " + " | ".join(row_cells) + f" | {bbox_str} |"
            )

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
