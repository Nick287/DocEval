"""Launch the Agent Framework DevUI for the doceval pipeline.

Usage::

    cd /workspaces/Agent/doceval
    python devui_app.py
    # → opens http://127.0.0.1:8088 in your browser

What you'll see in DevUI:
    * The workflow is registered with one entity named ``doceval_pipeline``.
    * The input box is auto-generated from the first executor's signature
      (``load_config`` takes ``str``), so just type a stem like ``11_mosaic``
      and press *Run*.
    * Each superstep streams in real-time — you can watch the 4 readers
      fire concurrently inside *superstep 1* and observe how the
      switch-case routes between ``verify_singletons`` and ``emit_reports``.
    * OpenTelemetry traces are enabled, so every ``ctx.send_message`` and
      LLM call appears in the trace viewer.

CLI options::

    --no-verify     Build the workflow without the LLM verifier (switch-case
                    will always take the Default branch).
    --port          Override the default port (8088).
    --host          Override the bind host (127.0.0.1).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``src/`` importable when running this file directly (no install).
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_framework.devui import serve  # noqa: E402

from doceval.agents import VisionVerifierAgent  # noqa: E402
from doceval.config import get_settings  # noqa: E402
from doceval.pipeline.batch_workflow import build_batch_workflow  # noqa: E402
from doceval.pipeline.workflow import build_pipeline_workflow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable the LLM verifier so switch-case always picks Default.",
    )
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--no-open", action="store_true", help="Do not auto-open the browser."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = get_settings()
    verifier = None if args.no_verify else VisionVerifierAgent()

    # Per-stem workflow — accepts a single stem string; useful for debugging
    # one image and seeing the full fan-out/fan-in/switch-case graph.
    per_stem_workflow = build_pipeline_workflow(
        settings=settings,
        verifier=verifier,
    )
    setattr(per_stem_workflow, "name", "doceval_per_stem")
    setattr(per_stem_workflow, "id", "doceval_per_stem")

    # Batch workflow — one run = process all stems + write summary.
    batch_workflow = build_batch_workflow(
        settings=settings,
        verifier=verifier,
    )
    setattr(batch_workflow, "name", "doceval_batch")
    setattr(batch_workflow, "id", "doceval_batch")

    md_root = settings.md_root
    available_md = (
        sorted(p.name for p in md_root.iterdir() if p.is_dir())
        if md_root.exists()
        else []
    )
    print()
    print("=" * 72)
    print(" doceval — Agent Framework DevUI")
    print("=" * 72)
    print(f"  verifier  : {'OFF (--no-verify)' if verifier is None else 'ON'}")
    print(f"  md_root   : {md_root}  (sources: {available_md or '<none>'})")
    print(f"  out_root  : {settings.out_root}")
    print(f"  url       : http://{args.host}:{args.port}")
    print()
    print("  Two entities are registered in the sidebar:")
    print("    • doceval_batch     — one run = all stems + summary.md")
    print("                              (form: stems, sources, concurrency;")
    print("                               leave fields empty to auto-discover)")
    print("    • doceval_per_stem  — type one stem string, see the full")
    print("                              fan-out/fan-in/switch-case graph")
    print("=" * 72)
    print()

    serve(
        entities=[batch_workflow, per_stem_workflow],
        host=args.host,
        port=args.port,
        auto_open=not args.no_open,
        instrumentation_enabled=True,  # show OpenTelemetry traces
        auth_enabled=False,            # local-only dev tool
    )


if __name__ == "__main__":
    main()
