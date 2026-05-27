"""Batch wrapper around the per-stem :mod:`doceval.pipeline.workflow`.

One ``workflow.run(BatchRequest(...))`` = process **all** stems in one shot,
write the aggregated ``summary.md`` / ``summary.csv``, and yield a single
:class:`BatchReport`. This mirrors what ``doceval run`` does in
:mod:`doceval.cli`, but exposes it as a DevUI entity so you can kick off
the whole batch from the browser instead of a shell loop.

Graph::

				BatchRequest
					 │
					 ▼
				  discover           ← resolve stems + MD sources
					 │
					 ▼
			  inspect_prereqs        ← per-stem check: image present?
					 │               DI cache present?  gpt MD present?
					 │               (no network calls here)
					 │
					 ├── add_edge(condition=any_needs_di) ──────→ ensure_di  ─┐
					 ├── add_edge(condition=any_needs_gpt) ─────→ ensure_gpt ─┤
					 └── add_edge(condition=nothing_needed) ──────────────────┤
																			  ▼
																  aggregate_preflight
																			  │
																			  ▼
																		  run_all
																			  │
																			  ▼
																	   summarize
																			  │
																			  ▼
																	   BatchReport

The two source-specific generators (``ensure_di`` / ``ensure_gpt``) sit on
independent conditional edges from ``inspect_prereqs`` so they run in
parallel whenever both prerequisites are missing. A third bypass edge fires
only when nothing is missing. ``aggregate_preflight`` acts as an idempotent
barrier that reads branch results from shared state and only emits one final
``PreflightResult``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_framework import Workflow, WorkflowBuilder, WorkflowContext, executor
from pydantic import BaseModel, Field

from doceval.agents import VisionVerifierAgent
from doceval.config import Settings, get_settings
from doceval.core import ImageEvaluation
from doceval.pipeline.workflow import build_pipeline_workflow
from doceval.reporting import SkippedEntry, write_summary
from doceval.sources import AzureDocIntelReader, MarkdownReader, discover_stems
from doceval.sources.gpt_md_generator import ensure_gpt_markdown

log = logging.getLogger("doceval.batch")


class BatchRequest(BaseModel):
	"""Input form for one batch run — mirrors ``doceval run`` defaults."""

	stems: list[str] = Field(
		default_factory=list,
		description=(
			"Specific stems to evaluate (folder names under MD/<source>/). "
			"Leave empty to auto-discover all stems shared across sources."
		),
	)
	sources: list[str] = Field(
		default_factory=list,
		description=(
			"MD source folder names under MD/ (e.g. `gemini`, `gpt`). "
			"Leave empty to auto-discover every subfolder."
		),
	)
	concurrency: int = Field(
		default=4,
		ge=1,
		le=8,
		description="Max number of stems running in parallel (1–8).",
	)


class SkippedStem(BaseModel):
	"""One stem that was dropped before/during processing."""

	stem: str
	stage: str
	reason: str


class BatchReport(BaseModel):
	"""Aggregated result yielded by the batch workflow."""

	stems: list[str]
	sources: list[str]
	summary_md: str
	summary_csv: str
	out_root: str
	elapsed_seconds: float
	evaluation_count: int
	skipped: list[SkippedStem] = []
	gemini_missing: list[str] = []


@dataclass
class BatchPlan:
	"""Resolved batch configuration produced by ``discover``."""

	stems: list[str]
	md_sources: list[str]
	all_sources: list[str]
	concurrency: int


@dataclass
class StemPrereq:
	"""Per-stem disk inventory produced by ``inspect_prereqs``."""

	stem: str
	image_path: Path | None
	needs_di: bool
	needs_gpt: bool
	gemini_present: bool


@dataclass
class PreflightPlan:
	"""Per-stem inventory carried through the conditional preflight edges."""

	plan: BatchPlan
	tasks: list[StemPrereq]
	no_image: list[SkippedStem]


@dataclass
class PreflightSourceResult:
	"""Per-source preflight outcome stored in shared state."""

	source: str
	successes: list[str]
	failures: list[SkippedStem]


@dataclass
class PreflightResult:
	"""Output of ``aggregate_preflight`` — partitions plan into ready/skipped."""

	plan: BatchPlan
	ready_stems: list[str]
	skipped: list[SkippedStem] = field(default_factory=list)
	gemini_missing: list[str] = field(default_factory=list)


@dataclass
class BatchResults:
	"""Output of ``run_all`` — handed to ``summarize`` to write reports."""

	plan: BatchPlan
	evaluations: list[ImageEvaluation]
	skipped: list[SkippedStem]
	gemini_missing: list[str]
	elapsed_seconds: float


_PREFLIGHT_KEY = "preflight_plan"
_DI_RESULT_KEY = "preflight_di_result"
_GPT_RESULT_KEY = "preflight_gpt_result"
_EMITTED_KEY = "preflight_emitted"


def build_batch_workflow(
	*,
	settings: Settings | None = None,
	verifier: VisionVerifierAgent | None = None,
) -> Workflow:
	"""Build a workflow that processes *all* stems in a single ``run`` call."""
	s = settings or get_settings()

	@executor(id="discover")
	async def discover(request: BatchRequest, ctx: WorkflowContext[BatchPlan]) -> None:
		log.info(
			"batch · discover START stems=%s sources=%s concurrency=%d",
			request.stems or "(auto)",
			request.sources or "(auto)",
			request.concurrency,
		)

		if request.sources:
			md_sources = list(request.sources)
		else:
			md_sources = s.list_md_sources()
		# The active vision-LLM (``s.model_name``) is an implicit source:
		# ``inspect_prereqs`` + ``ensure_gpt`` will always (re)generate
		# ``MD/<model_name>/<stem>.md`` for every stem, regardless of whether
		# that folder already exists on disk. Make sure it's in the source list
		# so the per-stem pipeline actually reads those files and the summary
		# report includes the corresponding columns.
		gpt_name = s.model_name
		if gpt_name not in md_sources:
			md_sources.append(gpt_name)
		# Ensure the directory exists so MarkdownReader.available_stems() in the
		# stem-intersection step below doesn't crash on a missing folder.
		s.gpt_md_dir.mkdir(parents=True, exist_ok=True)
		if not md_sources:
			raise RuntimeError(f"no MD source folders found under {s.md_root}; nothing to do")

		if request.stems:
			stems = list(request.stems)
		else:
			# Build readers only for sources that actually have files on disk —
			# the stem intersection is meant to find stems present everywhere we
			# can *currently* read. The active model dir is excluded even if it
			# already contains some files (preflight + ensure_gpt will (re-)fill
			# in the missing stems), so a half-populated MD/<model>/ folder
			# can't silently shrink the batch.
			intersect_sources = [
				n for n in md_sources
				if n != gpt_name and any((s.md_root / n).glob("*.md"))
			]
			readers = [
				AzureDocIntelReader(name="di"),
				*[MarkdownReader(name=name, root=s.md_root / name) for name in intersect_sources],
			]
			stems = discover_stems(*readers)
		if not stems:
			raise RuntimeError("no stems shared across all sources; nothing to do")

		plan = BatchPlan(stems=stems, md_sources=md_sources, all_sources=["di", *md_sources], concurrency=request.concurrency)
		log.info("batch · discover DONE  %d stem(s) × sources=%s", len(stems), plan.all_sources)
		await ctx.send_message(plan)

	@executor(id="inspect_prereqs")
	async def inspect_prereqs(plan: BatchPlan, ctx: WorkflowContext[PreflightPlan]) -> None:
		log.info("preflight · inspect_prereqs START  %d stem(s)", len(plan.stems))
		di_reader_local = AzureDocIntelReader(name="di")
		gpt_dir = s.gpt_md_dir
		gemini_dir = s.md_root / "gemini"
		cache_dir = s.di_cache_dir

		tasks: list[StemPrereq] = []
		no_image: list[SkippedStem] = []
		for stem in plan.stems:
			image_path = di_reader_local.find_image(stem)
			if image_path is None:
				no_image.append(SkippedStem(stem=stem, stage="preflight", reason=f"no image for stem {stem!r} under {di_reader_local.image_dir}"))
				continue
			has_di_cache = cache_dir.is_dir() and any(cache_dir.glob(f"{stem}.*.json"))
			tasks.append(
				StemPrereq(
					stem=stem,
					image_path=image_path,
					needs_di=not has_di_cache,
					needs_gpt=not (gpt_dir / f"{stem}.md").exists(),
					gemini_present=(gemini_dir / f"{stem}.md").exists(),
				)
			)

		pre = PreflightPlan(plan=plan, tasks=tasks, no_image=no_image)
		ctx.set_state(_PREFLIGHT_KEY, pre)
		log.info(
			"preflight · inspect_prereqs DONE  no_image=%d need_di=%d need_gpt=%d gemini_missing=%d",
			len(no_image),
			sum(1 for t in tasks if t.needs_di),
			sum(1 for t in tasks if t.needs_gpt),
			sum(1 for t in tasks if not t.gemini_present),
		)
		await ctx.send_message(pre)

	def any_needs_di(message: Any) -> bool:
		return isinstance(message, PreflightPlan) and any(t.needs_di for t in message.tasks)

	def any_needs_gpt(message: Any) -> bool:
		return isinstance(message, PreflightPlan) and any(t.needs_gpt for t in message.tasks)

	def nothing_needed(message: Any) -> bool:
		return isinstance(message, PreflightPlan) and not any(t.needs_di for t in message.tasks) and not any(t.needs_gpt for t in message.tasks)

	@executor(id="ensure_di")
	async def ensure_di(pre: PreflightPlan, ctx: WorkflowContext[PreflightPlan]) -> None:
		reader = AzureDocIntelReader(name="di")
		todo = [t for t in pre.tasks if t.needs_di and t.image_path is not None]
		already = [t for t in pre.tasks if not t.needs_di]
		log.info("step 3a · ensure_di START  %d to generate, %d already cached", len(todo), len(already))
		sem = asyncio.Semaphore(pre.plan.concurrency)

		async def _one(task: StemPrereq) -> tuple[str, SkippedStem | None]:
			async with sem:
				assert task.image_path is not None
				try:
					log.info("step 3a · ensure_di  %s  calling Doc Intel…", task.stem)
					await asyncio.to_thread(reader.analyze, task.image_path)
					log.info("step 3a · ensure_di  %s  OK", task.stem)
					return task.stem, None
				except Exception as exc:  # noqa: BLE001
					log.warning("step 3a · ensure_di  %s  FAIL: %s", task.stem, exc)
					return task.stem, SkippedStem(stem=task.stem, stage="preflight", reason=f"Doc Intelligence call failed: {exc}")

		rows = await asyncio.gather(*[_one(t) for t in todo])
		successes = [t.stem for t in already]
		failures: list[SkippedStem] = []
		for stem, err in rows:
			if err is None:
				successes.append(stem)
			else:
				failures.append(err)
		log.info("step 3a · ensure_di DONE  ok=%d fail=%d", len(successes), len(failures))
		ctx.set_state(_DI_RESULT_KEY, PreflightSourceResult(source="di", successes=successes, failures=failures))
		await ctx.send_message(pre)

	@executor(id="ensure_gpt")
	async def ensure_gpt(pre: PreflightPlan, ctx: WorkflowContext[PreflightPlan]) -> None:
		gpt_dir = s.gpt_md_dir
		gpt_name = s.model_name
		todo = [t for t in pre.tasks if t.needs_gpt and t.image_path is not None]
		already = [t for t in pre.tasks if not t.needs_gpt]
		log.info("step 3b · ensure_gpt START  %d to generate, %d already present  (model=%s)", len(todo), len(already), gpt_name)
		sem = asyncio.Semaphore(pre.plan.concurrency)

		async def _one(task: StemPrereq) -> tuple[str, SkippedStem | None]:
			async with sem:
				assert task.image_path is not None
				out_path = gpt_dir / f"{task.stem}.md"
				try:
					log.info("step 3b · ensure_gpt  %s  calling vision LLM…", task.stem)
					await ensure_gpt_markdown(task.image_path, out_path)
					log.info("step 3b · ensure_gpt  %s  OK", task.stem)
					return task.stem, None
				except Exception as exc:  # noqa: BLE001
					log.warning("step 3b · ensure_gpt  %s  FAIL: %s", task.stem, exc)
					return task.stem, SkippedStem(stem=task.stem, stage="preflight", reason=f"GPT MD generation failed: {exc}")

		rows = await asyncio.gather(*[_one(t) for t in todo])
		successes = [t.stem for t in already]
		failures: list[SkippedStem] = []
		for stem, err in rows:
			if err is None:
				successes.append(stem)
			else:
				failures.append(err)
		log.info("step 3b · ensure_gpt DONE  ok=%d fail=%d", len(successes), len(failures))
		ctx.set_state(_GPT_RESULT_KEY, PreflightSourceResult(source=gpt_name, successes=successes, failures=failures))
		await ctx.send_message(pre)

	@executor(id="aggregate_preflight")
	async def aggregate_preflight(pre: PreflightPlan, ctx: WorkflowContext[PreflightResult]) -> None:
		if ctx.get_state(_EMITTED_KEY):
			return

		expect_di = any(t.needs_di for t in pre.tasks)
		expect_gpt = any(t.needs_gpt for t in pre.tasks)
		di_result: PreflightSourceResult | None = ctx.get_state(_DI_RESULT_KEY)
		gpt_result: PreflightSourceResult | None = ctx.get_state(_GPT_RESULT_KEY)

		if expect_di and di_result is None:
			log.info("step 4 · aggregate_preflight  waiting on DI branch…")
			return
		if expect_gpt and gpt_result is None:
			log.info("step 4 · aggregate_preflight  waiting on GPT branch…")
			return

		if di_result is None:
			di_ok = {t.stem for t in pre.tasks}
			di_failures: list[SkippedStem] = []
		else:
			di_ok = set(di_result.successes)
			di_failures = list(di_result.failures)

		if gpt_result is None:
			gpt_ok = {t.stem for t in pre.tasks}
			gpt_failures: list[SkippedStem] = []
		else:
			gpt_ok = set(gpt_result.successes)
			gpt_failures = list(gpt_result.failures)

		failures: dict[str, SkippedStem] = {}
		for f in di_failures + gpt_failures:
			failures.setdefault(f.stem, f)
		for f in pre.no_image:
			failures.setdefault(f.stem, f)

		ready_set = (di_ok & gpt_ok) - set(failures.keys())
		ready = [t.stem for t in pre.tasks if t.stem in ready_set]
		skipped = list(failures.values())
		gemini_missing = [t.stem for t in pre.tasks if t.stem in ready_set and not t.gemini_present]
		if gemini_missing:
			log.warning("step 4 · aggregate_preflight  MD/gemini missing for %d stem(s) (continuing): %s", len(gemini_missing), gemini_missing)
		log.info("step 4 · aggregate_preflight DONE  ready=%d skipped=%d", len(ready), len(skipped))
		ctx.set_state(_EMITTED_KEY, True)
		await ctx.send_message(PreflightResult(plan=pre.plan, ready_stems=ready, skipped=skipped, gemini_missing=gemini_missing))

	@executor(id="run_all")
	async def run_all(pre: PreflightResult, ctx: WorkflowContext[BatchResults]) -> None:
		plan = pre.plan
		log.info("batch · run_all START %d stem(s) concurrency=%d (skipped=%d)", len(pre.ready_stems), plan.concurrency, len(pre.skipped))
		started = time.time()
		sem = asyncio.Semaphore(plan.concurrency)

		async def _one(stem: str) -> ImageEvaluation | SkippedStem:
			async with sem:
				log.info("batch · stem START %s", stem)
				try:
					sub = build_pipeline_workflow(settings=s, verifier=verifier, sources=plan.md_sources)
					result = await sub.run(stem)
					outputs = result.get_outputs()
					if not outputs:
						raise RuntimeError("per-stem workflow produced no output")
					log.info("batch · stem DONE  %s (%.1fs)", stem, outputs[0].elapsed_seconds)
					return outputs[0]
				except Exception as exc:  # noqa: BLE001
					log.warning("batch · stem FAIL  %s: %s", stem, exc)
					return SkippedStem(stem=stem, stage="run_all", reason=str(exc))

		results = await asyncio.gather(*[_one(stem) for stem in pre.ready_stems])
		evaluations: list[ImageEvaluation] = []
		skipped = list(pre.skipped)
		for item in results:
			if isinstance(item, SkippedStem):
				skipped.append(item)
			else:
				evaluations.append(item)

		elapsed = time.time() - started
		log.info("batch · run_all DONE  %d evaluations  %d skipped  in %.1fs", len(evaluations), len(skipped), elapsed)
		await ctx.send_message(BatchResults(plan=plan, evaluations=evaluations, skipped=skipped, gemini_missing=pre.gemini_missing, elapsed_seconds=elapsed))

	@executor(id="summarize", workflow_output=BatchReport)
	async def summarize(results: BatchResults, ctx: WorkflowContext[Any, BatchReport]) -> None:
		log.info("batch · summarize START  %d evaluation(s)  %d skipped", len(results.evaluations), len(results.skipped))
		out_root = s.out_root
		out_root.mkdir(parents=True, exist_ok=True)
		skipped_entries = [
			SkippedEntry(stem=sk.stem, stage=sk.stage, reason=sk.reason)
			for sk in results.skipped
		]
		write_summary(
			results.evaluations,
			out_root,
			sources=results.plan.all_sources,
			skipped=skipped_entries,
		)

		report = BatchReport(
			stems=[ev.stem for ev in results.evaluations],
			sources=results.plan.all_sources,
			summary_md=str(out_root / "summary.md"),
			summary_csv=str(out_root / "summary.csv"),
			out_root=str(out_root),
			elapsed_seconds=results.elapsed_seconds,
			evaluation_count=len(results.evaluations),
			skipped=results.skipped,
			gemini_missing=results.gemini_missing,
		)
		log.info("batch · summarize DONE  %s  (skipped=%d, gemini_missing=%d)", report.summary_md, len(report.skipped), len(report.gemini_missing))
		await ctx.yield_output(report)

	return (
		WorkflowBuilder(start_executor=discover)
		.add_edge(discover, inspect_prereqs)
		.add_edge(inspect_prereqs, ensure_di, condition=any_needs_di)
		.add_edge(inspect_prereqs, ensure_gpt, condition=any_needs_gpt)
		.add_edge(inspect_prereqs, aggregate_preflight, condition=nothing_needed)
		.add_edge(ensure_di, aggregate_preflight)
		.add_edge(ensure_gpt, aggregate_preflight)
		.add_edge(aggregate_preflight, run_all)
		.add_edge(run_all, summarize)
		.build()
	)
