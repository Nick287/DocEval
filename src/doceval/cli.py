"""Typer-based CLI: ``doceval run [--stem ...] [--no-verify]``."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from doceval.agents import VisionVerifierAgent
from doceval.config import get_settings
from doceval.pipeline import list_available_sources, run_workflow_many
from doceval.reporting import SkippedEntry, write_summary
from doceval.sources import AzureDocIntelReader, MarkdownReader, discover_stems

app = typer.Typer(add_completion=False, help="DocEval MD verification — consensus pipeline.")
log = logging.getLogger("doceval")


@app.command()
def run(
    stem: list[str] = typer.Option(
        None, "--stem", "-s", help="Run only these stems (omit to run all)."
    ),
    sources: list[str] = typer.Option(
        None,
        "--source",
        help="MD source folder names to include; omit to auto-discover under MD/.",
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip vision verifier (no LLM calls)."
    ),
    concurrency: int = typer.Option(1, "--concurrency", "-c", min=1, max=8),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run the Agent-Framework workflow over one or more images."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = get_settings()
    md_sources = sources or list_available_sources(settings)
    if not md_sources:
        typer.echo("no MD source folders found under MD/; nothing to do.", err=True)
        raise typer.Exit(code=1)

    all_sources: list[str] = ["di", *md_sources]
    verifier = (
        None
        if no_verify or not settings.verify_singletons
        else VisionVerifierAgent()
    )

    # Stem discovery still uses the lightweight reader helpers — we just need
    # to know which stems exist on disk before we kick off the workflow.
    di_reader = AzureDocIntelReader(name="di")
    md_reader_map = {
        name: MarkdownReader(name=name, root=settings.md_root / name)
        for name in md_sources
    }
    readers = [di_reader, *md_reader_map.values()]
    if stem:
        stems = list(stem)
    else:
        stems = discover_stems(*readers)
        if not stems:
            typer.echo("no stems shared across all sources; nothing to do.", err=True)
            raise typer.Exit(code=1)

    # Surface any stems that exist on disk somewhere but aren't shared across
    # all sources, so the user understands why the run count < image count.
    dropped_entries: list[SkippedEntry] = []
    if not stem:
        image_stems = di_reader.available_stems()
        union_stems = set(image_stems)
        for r in md_reader_map.values():
            union_stems |= r.available_stems()
        dropped = sorted(union_stems - set(stems))
        if dropped:
            per_source_missing: dict[str, list[str]] = {}
            for d in dropped:
                missing_in: list[str] = []
                if d not in image_stems:
                    missing_in.append("image")
                for name, r in md_reader_map.items():
                    if d not in r.available_stems():
                        missing_in.append(name)
                per_source_missing[d] = missing_in
            typer.echo(
                f"⚠️  skipping {len(dropped)} stem(s) not present in every source:",
                err=True,
            )
            for d, srcs in per_source_missing.items():
                typer.echo(f"     - {d}  (missing in: {', '.join(srcs)})", err=True)
                dropped_entries.append(
                    SkippedEntry(
                        stem=d,
                        stage="discovery",
                        reason=f"缺少源: {', '.join(srcs)}",
                    )
                )

    typer.echo(
        f"found {len(stems)} stem(s); sources={all_sources}; "
        f"verifier={'on' if verifier else 'off'}"
    )

    out_root = settings.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    async def _go() -> None:
        evaluations = await run_workflow_many(
            stems,
            concurrency=concurrency,
            settings=settings,
            verifier=verifier,
            sources=md_sources,
        )
        for ev in evaluations:
            typer.echo(
                f"  [{ev.stem}] {ev.elapsed_seconds:.1f}s — "
                f"clusters={len(ev.clusters)} verifier_model={ev.verifier_model or '-'}"
            )
        write_summary(evaluations, out_root, sources=all_sources, skipped=dropped_entries)
        typer.echo(f"summary → {out_root / 'summary.md'}")

    asyncio.run(_go())


@app.command()
def di(
    stem: str = typer.Argument(..., help="Image stem (without extension)."),
) -> None:
    """Run Document Intelligence for one image and dump tokens it found (no LLM)."""
    from doceval.sources import AzureDocIntelReader

    reader = AzureDocIntelReader()
    hits = reader.read(stem)
    typer.echo(f"{len(hits)} structured tokens:")
    for h in hits:
        typer.echo(f"  {h.norm:30s}  surface={h.surface!r:20s}  bbox={h.bbox}")


if __name__ == "__main__":
    app()
