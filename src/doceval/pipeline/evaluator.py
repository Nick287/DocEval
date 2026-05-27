"""End-to-end evaluation of one image.

The evaluator is plain Python — readers and the verifier are passed in.
That keeps it trivial to test (swap in fake readers) and to extend
(add a third markdown source by registering one more :class:`MarkdownReader`).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from doceval.agents import VisionVerifierAgent
from doceval.config import get_settings
from doceval.consensus import apply_vision_verdict, build_clusters, vote
from doceval.core import (
    Cluster,
    ImageEvaluation,
    SourceJudgement,
    SourceName,
    TokenHit,
)
from doceval.sources import AzureDocIntelReader, MarkdownReader, TokenReader


class Evaluator:
    """Run the DI + multi-source consensus pipeline for one stem at a time."""

    def __init__(
        self,
        di_reader: AzureDocIntelReader,
        markdown_readers: list[MarkdownReader],
        verifier: VisionVerifierAgent | None = None,
        *,
        max_cluster_distance: int = 1,
    ) -> None:
        self.di_reader = di_reader
        self.markdown_readers = markdown_readers
        self.verifier = verifier
        self.max_cluster_distance = max_cluster_distance

    # ------------------------------------------------------------------
    @property
    def all_sources(self) -> list[SourceName]:
        return [self.di_reader.name] + [r.name for r in self.markdown_readers]

    @property
    def readers(self) -> list[TokenReader]:
        return [self.di_reader, *self.markdown_readers]

    # ------------------------------------------------------------------
    async def evaluate(self, stem: str) -> ImageEvaluation:
        """Evaluate one image; the optional verifier runs concurrently per stem."""
        t0 = time.time()
        image_path = self.di_reader.find_image(stem)
        if image_path is None:
            raise FileNotFoundError(f"image for stem {stem!r} not found")

        # --- 1. Collect hits from every source ----------------------------
        hits: list[TokenHit] = []
        for reader in self.readers:
            hits.extend(reader.read(stem))

        # --- 2. Cluster + vote -------------------------------------------
        clusters = build_clusters(hits, max_distance=self.max_cluster_distance)
        clusters, judgements = vote(
            clusters, self.all_sources, di_source=self.di_reader.name
        )

        # --- 3. Vision verify singletons --------------------------------
        if self.verifier is not None:
            await self._verify_singletons(image_path, judgements)

        return ImageEvaluation(
            stem=stem,
            image_path=str(image_path),
            clusters=clusters,
            judgements=judgements,
            elapsed_seconds=time.time() - t0,
            verifier_model=getattr(self.verifier, "last_model", None)
            if self.verifier is not None
            else None,
        )

    # ------------------------------------------------------------------
    async def _verify_singletons(
        self,
        image_path: Path,
        judgements: list[SourceJudgement],
    ) -> None:
        """For each ``hallucination`` judgement ask the verifier; upgrade in place."""
        targets: list[SourceJudgement] = [
            j for j in judgements if j.verdict == "hallucination" and j.surface_observed
        ]
        if not targets:
            return

        surfaces = list({j.surface_observed for j in targets if j.surface_observed})
        verdicts = await self.verifier.verify(image_path, surfaces)

        for j in targets:
            verdict_obj = verdicts.get(j.surface_observed or "")
            if not verdict_obj:
                continue
            v = verdict_obj["verdict"]
            if v == "present":
                apply_vision_verdict(j, True, verdict_obj.get("evidence", ""))
            elif v == "ambiguous":
                j.verdict = "ambiguous"
                j.evidence = verdict_obj.get("evidence", "")
            else:  # absent — keep hallucination but record evidence
                j.evidence = verdict_obj.get("evidence", "")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_default_evaluator(
    *,
    sources: list[str] | None = None,
    enable_verifier: bool = True,
) -> Evaluator:
    """Build an evaluator wired up to whatever sources exist under ``MD/``.

    If ``sources`` is None, all subdirectories of ``MD/`` are picked up
    automatically. Pass ``sources=["gemini", "gpt-5.4"]`` to constrain.
    """
    s = get_settings()
    md_root = s.md_root

    if sources is None:
        discovered = [p.name for p in md_root.iterdir() if p.is_dir()] if md_root.exists() else []
        sources = sorted(discovered)
    if not sources:
        raise RuntimeError(
            f"no markdown source directories found under {md_root!s}. "
            "Create at least one folder like MD/gemini/ with *.md files."
        )

    md_readers = [MarkdownReader(name=name, root=md_root / name) for name in sources]
    di_reader = AzureDocIntelReader(name="di")
    verifier = VisionVerifierAgent() if enable_verifier and s.verify_singletons else None
    return Evaluator(
        di_reader=di_reader,
        markdown_readers=md_readers,
        verifier=verifier,
        max_cluster_distance=s.cluster_edit_distance,
    )


async def evaluate_many(
    evaluator: Evaluator,
    stems: list[str],
    *,
    concurrency: int = 1,
) -> list[ImageEvaluation]:
    """Evaluate ``stems`` (optionally in parallel) and return all results."""
    if concurrency <= 1:
        return [await evaluator.evaluate(s) for s in stems]

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(stem: str) -> ImageEvaluation:
        async with semaphore:
            return await evaluator.evaluate(stem)

    return list(await asyncio.gather(*[_one(s) for s in stems]))
