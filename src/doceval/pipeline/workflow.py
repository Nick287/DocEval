"""Pipeline expressed as a Microsoft Agent Framework :class:`Workflow`.

The graph uses three edge patterns from the Agent Framework cookbook:

    * **Fan-out** (``add_fan_out_edges``) — ``load_config`` broadcasts to 4
      readers; the Azure OCR call (~seconds) runs concurrently with 3 local
      MD reads (~milliseconds) instead of being added on top of them.
    * **Fan-in** (``add_fan_in_edges``) — ``aggregate_hits`` waits for all 4
      readers, merges their ``ReaderOutput`` payloads.
    * **Switch-Case** (``add_switch_case_edge_group``) — when there is
      nothing to verify (no singletons, or verifier disabled), the LLM
      step is skipped at the graph level::

                            load_config
                                │
                ┌───────────────┴────fan-out────────────┐
                ▼               ▼               ▼       ▼
       read_doc_intel  read_gemini_md  read_gpt_md  read_extra_sources
                │               │               │       │
                └────────────fan-in─────────────────────┘
                                │
                                ▼
                        aggregate_hits
                                │
                                ▼
                        cluster_and_vote
                                │
                  ┌─────────────┴───switch-case────────┐
                  ▼ Case(has_singletons)      Default ▼
           verify_singletons ───────────▶  emit_reports
                                                │
                                                ▼
                                        ImageEvaluation
                                       (workflow output)

每一个节点都打印 ``step N · name START / DONE`` 日志，方便在终端看到执行过程。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_framework import (
    Case,
    Default,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)

from intsig_eval.agents import VisionVerifierAgent
from intsig_eval.config import Settings, get_settings
from intsig_eval.consensus import apply_vision_verdict, build_clusters, vote
from intsig_eval.core import (
    Cluster,
    ImageEvaluation,
    SourceJudgement,
    SourceName,
    TokenHit,
)
from intsig_eval.sources import AzureLayoutOCRReader, MarkdownReader

log = logging.getLogger("intsig_eval.workflow")


# ---------------------------------------------------------------------------
# Shared state flowing through the workflow
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    """Mutable payload threaded through every executor."""

    stem: str
    image_path: Path | None = None
    # Sources we plan to evaluate (always includes "ocr"; MD sources only if
    # the corresponding file exists for this stem).
    sources_present: list[SourceName] = field(default_factory=list)
    # All hits collected so far, regardless of source.
    hits: list[TokenHit] = field(default_factory=list)
    # Filled by cluster_and_vote.
    clusters: list[Cluster] = field(default_factory=list)
    judgements: list[SourceJudgement] = field(default_factory=list)
    # Filled by verify_singletons.
    verifier_model: str | None = None
    # Filled by emit_reports.
    elapsed_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)


@dataclass
class ReaderOutput:
    """Partial result emitted by each parallel reader (fan-out branch).

    Each reader is *read-only* w.r.t. the shared :class:`PipelineState`;
    they build a fresh ``ReaderOutput`` and the fan-in target merges them
    back into the state.  This avoids races when the 4 readers run
    concurrently.
    """

    source_names: list[SourceName] = field(default_factory=list)
    hits: list[TokenHit] = field(default_factory=list)


# Key used to persist the PipelineState in the workflow's shared store while
# fan-out branches are in flight.
_STATE_KEY = "pipeline_state"


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------
def build_pipeline_workflow(
    *,
    settings: Settings | None = None,
    verifier: VisionVerifierAgent | None = None,
    out_root: Path | None = None,
    sources: list[str] | None = None,
) -> Workflow:
    """Construct a fresh :class:`Workflow` for one stem.

    Dependencies are captured in closures so the executors stay pure
    ``(message, ctx)``-shaped functions.
    """
    s = settings or get_settings()
    out_root = out_root or s.out_root
    discovered = (
        [p.name for p in s.md_root.iterdir() if p.is_dir()]
        if s.md_root.exists()
        else []
    )
    md_sources = sources if sources is not None else sorted(discovered)

    ocr_reader = AzureLayoutOCRReader(name="ocr")
    md_readers: dict[str, MarkdownReader] = {
        name: MarkdownReader(name=name, root=s.md_root / name) for name in md_sources
    }

    # ------------------------------------------------------------------
    # [1] load_config — broadcast state to all readers (fan-out source)
    # ------------------------------------------------------------------
    @executor(id="load_config")
    async def load_config(
        stem: str, ctx: WorkflowContext[PipelineState]
    ) -> None:
        log.info("step 1 · load_config START stem=%s", stem)
        image_path = ocr_reader.find_image(stem)
        if image_path is None:
            raise FileNotFoundError(f"image for stem {stem!r} not found")

        state = PipelineState(
            stem=stem,
            image_path=image_path,
            sources_present=[],  # populated by aggregate_hits
        )
        # Persist the state so aggregate_hits can recover it without
        # depending on the reader payload.
        ctx.set_state(_STATE_KEY, state)
        log.info(
            "step 1 · load_config DONE  image=%s available_md=%s (fan-out → 4 readers)",
            image_path.name,
            list(md_readers.keys()),
        )
        await ctx.send_message(state)

    # ------------------------------------------------------------------
    # [2a-d] Parallel readers — each emits a ReaderOutput, never mutates
    #         the shared state.  They run concurrently under fan-out.
    # ------------------------------------------------------------------
    @executor(id="read_doc_intel")
    async def read_doc_intel(
        state: PipelineState, ctx: WorkflowContext[ReaderOutput]
    ) -> None:
        log.info("step 2a · read_doc_intel START (Azure call)")
        hits = ocr_reader.read(state.stem)
        log.info("step 2a · read_doc_intel DONE  +%d hits", len(hits))
        # OCR is always considered "present" so vote() can use it as anchor.
        await ctx.send_message(ReaderOutput(source_names=["ocr"], hits=hits))

    @executor(id="read_gemini_md")
    async def read_gemini_md(
        state: PipelineState, ctx: WorkflowContext[ReaderOutput]
    ) -> None:
        reader = md_readers.get("gemini")
        path = (s.md_root / "gemini" / f"{state.stem}.md") if reader else None
        if reader and path and path.exists():
            hits = reader.read(state.stem)
            log.info("step 2b · read_gemini_md HIT  +%d hits", len(hits))
            await ctx.send_message(
                ReaderOutput(source_names=["gemini"], hits=hits)
            )
        else:
            log.info(
                "step 2b · read_gemini_md SKIP (no MD/gemini/%s.md)", state.stem
            )
            await ctx.send_message(ReaderOutput())

    @executor(id="read_gpt_md")
    async def read_gpt_md(
        state: PipelineState, ctx: WorkflowContext[ReaderOutput]
    ) -> None:
        reader = md_readers.get("gpt")
        path = (s.md_root / "gpt" / f"{state.stem}.md") if reader else None
        if reader and path and path.exists():
            hits = reader.read(state.stem)
            log.info("step 2c · read_gpt_md HIT   +%d hits", len(hits))
            await ctx.send_message(
                ReaderOutput(source_names=["gpt"], hits=hits)
            )
        else:
            log.info("step 2c · read_gpt_md SKIP (no MD/gpt/%s.md)", state.stem)
            await ctx.send_message(ReaderOutput())

    extra_names = [n for n in md_sources if n not in ("gemini", "gpt")]

    @executor(id="read_extra_sources")
    async def read_extra_sources(
        state: PipelineState, ctx: WorkflowContext[ReaderOutput]
    ) -> None:
        if not extra_names:
            log.info("step 2d · read_extra_sources SKIP (no extra MD sources)")
            await ctx.send_message(ReaderOutput())
            return
        added_sources: list[SourceName] = []
        all_hits: list[TokenHit] = []
        added_log: list[tuple[str, int]] = []
        for name in extra_names:
            reader = md_readers[name]
            path = s.md_root / name / f"{state.stem}.md"
            if not path.exists():
                continue
            hits = reader.read(state.stem)
            all_hits.extend(hits)
            added_sources.append(name)
            added_log.append((name, len(hits)))
        log.info(
            "step 2d · read_extra_sources DONE  %s",
            ", ".join(f"{n}+{c}" for n, c in added_log) or "(none matched)",
        )
        await ctx.send_message(
            ReaderOutput(source_names=added_sources, hits=all_hits)
        )

    # ------------------------------------------------------------------
    # [3] aggregate_hits — fan-in target; merges 4 ReaderOutputs into state
    # ------------------------------------------------------------------
    @executor(id="aggregate_hits")
    async def aggregate_hits(
        parts: list[ReaderOutput], ctx: WorkflowContext[PipelineState]
    ) -> None:
        state: PipelineState = ctx.get_state(_STATE_KEY)
        for part in parts:
            state.sources_present.extend(part.source_names)
            state.hits.extend(part.hits)
        log.info(
            "step 3 · aggregate_hits DONE  total_hits=%d sources=%s",
            len(state.hits),
            state.sources_present,
        )
        await ctx.send_message(state)

    # ------------------------------------------------------------------
    # [4] cluster_and_vote — pure local
    # ------------------------------------------------------------------
    @executor(id="cluster_and_vote")
    async def cluster_and_vote(
        state: PipelineState, ctx: WorkflowContext[PipelineState]
    ) -> None:
        log.info(
            "step 4 · cluster_and_vote START total_hits=%d sources=%s",
            len(state.hits),
            state.sources_present,
        )
        clusters = build_clusters(state.hits, max_distance=s.cluster_edit_distance)
        clusters, judgements = vote(
            clusters, state.sources_present, ocr_source="ocr"
        )
        state.clusters = clusters
        state.judgements = judgements
        log.info(
            "step 4 · cluster_and_vote DONE  clusters=%d judgements=%d",
            len(clusters),
            len(judgements),
        )
        await ctx.send_message(state)

    # ------------------------------------------------------------------
    # [5] verify_singletons — only LLM call in the whole workflow
    #     Reached only when the switch-case below selects this branch.
    # ------------------------------------------------------------------
    @executor(id="verify_singletons")
    async def verify_singletons(
        state: PipelineState, ctx: WorkflowContext[PipelineState]
    ) -> None:
        # By switch-case construction `verifier` is not None here and at
        # least one singleton with a surface needs verification.
        assert verifier is not None
        targets = [
            j for j in state.judgements
            if j.verdict == "hallucination" and j.surface_observed
        ]
        surfaces = list({j.surface_observed for j in targets if j.surface_observed})
        log.info(
            "step 5 · verify_singletons START surfaces=%d → calling LLM",
            len(surfaces),
        )
        verdicts = await verifier.verify(state.image_path, surfaces)
        for j in targets:
            v = verdicts.get(j.surface_observed or "")
            if not v:
                continue
            kind = v["verdict"]
            if kind == "present":
                apply_vision_verdict(j, True, v.get("evidence", ""))
            elif kind == "ambiguous":
                j.verdict = "ambiguous"
                j.evidence = v.get("evidence", "")
            else:  # absent → keep hallucination, record evidence
                j.evidence = v.get("evidence", "")
        state.verifier_model = verifier.last_model
        log.info(
            "step 5 · verify_singletons DONE  served_model=%s",
            state.verifier_model,
        )
        await ctx.send_message(state)

    # ------------------------------------------------------------------
    # [6] emit_reports — write disk + yield workflow output
    # ------------------------------------------------------------------
    # Import here to avoid a circular import at module load time
    from intsig_eval.reporting import annotate, write_clusters_json, write_report

    @executor(id="emit_reports", workflow_output=ImageEvaluation)
    async def emit_reports(
        state: PipelineState, ctx: WorkflowContext[Any, ImageEvaluation]
    ) -> None:
        state.elapsed_seconds = time.time() - state.started_at
        evaluation = ImageEvaluation(
            stem=state.stem,
            image_path=str(state.image_path),
            clusters=state.clusters,
            judgements=state.judgements,
            elapsed_seconds=state.elapsed_seconds,
            verifier_model=state.verifier_model,
        )

        out_dir = out_root / state.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        write_clusters_json(evaluation, out_dir / "clusters.json")
        write_report(evaluation, out_dir / "report.md")
        annotate(evaluation, out_dir, sources=state.sources_present)
        log.info(
            "step 6 · emit_reports DONE  %.1fs → %s/",
            state.elapsed_seconds,
            out_dir,
        )
        await ctx.yield_output(evaluation)

    # ------------------------------------------------------------------
    # Switch-case predicate: do we actually need the LLM step?
    # ------------------------------------------------------------------
    def has_singletons_to_verify(message: Any) -> bool:
        if verifier is None:
            return False
        if not isinstance(message, PipelineState):
            return False
        return any(
            j.verdict == "hallucination" and j.surface_observed
            for j in message.judgements
        )

    # ------------------------------------------------------------------
    # Wire the graph (fan-out → fan-in → switch-case)
    # ------------------------------------------------------------------
    readers = [read_doc_intel, read_gemini_md, read_gpt_md, read_extra_sources]
    return (
        WorkflowBuilder(start_executor=load_config)
        # [1] → [2a..d] : parallel readers (fan-out)
        .add_fan_out_edges(load_config, readers)
        # [2a..d] → [3] : aggregate partial outputs (fan-in)
        .add_fan_in_edges(readers, aggregate_hits)
        # [3] → [4]
        .add_edge(aggregate_hits, cluster_and_vote)
        # [4] → [5] or [6] : switch-case skips the LLM step when not needed
        .add_switch_case_edge_group(
            cluster_and_vote,
            [
                Case(condition=has_singletons_to_verify, target=verify_singletons),
                Default(target=emit_reports),
            ],
        )
        # [5] → [6]
        .add_edge(verify_singletons, emit_reports)
        .build()
    )


# ---------------------------------------------------------------------------
# Convenience runners — mirror the old ``evaluate_many`` shape
# ---------------------------------------------------------------------------
async def run_workflow_for(
    stem: str,
    *,
    settings: Settings | None = None,
    verifier: VisionVerifierAgent | None = None,
    sources: list[str] | None = None,
) -> ImageEvaluation:
    """Build + run one workflow for a single stem; return the evaluation."""
    workflow = build_pipeline_workflow(
        settings=settings, verifier=verifier, sources=sources
    )
    result = await workflow.run(stem)
    outputs = result.get_outputs()
    if not outputs:
        raise RuntimeError(f"workflow produced no output for stem={stem!r}")
    return outputs[0]


async def run_workflow_many(
    stems: list[str],
    *,
    concurrency: int = 1,
    settings: Settings | None = None,
    verifier: VisionVerifierAgent | None = None,
    sources: list[str] | None = None,
) -> list[ImageEvaluation]:
    """Run one workflow per stem, optionally with bounded concurrency."""
    import asyncio

    if concurrency <= 1:
        return [
            await run_workflow_for(
                stem, settings=settings, verifier=verifier, sources=sources
            )
            for stem in stems
        ]

    sem = asyncio.Semaphore(concurrency)

    async def _one(stem: str) -> ImageEvaluation:
        async with sem:
            return await run_workflow_for(
                stem, settings=settings, verifier=verifier, sources=sources
            )

    return list(await asyncio.gather(*[_one(stem) for stem in stems]))


def list_available_sources(settings: Settings | None = None) -> list[str]:
    """Return the MD source directory names that exist under ``MD/``."""
    s = settings or get_settings()
    if not s.md_root.exists():
        return []
    return sorted(p.name for p in s.md_root.iterdir() if p.is_dir())
