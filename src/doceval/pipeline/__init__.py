from doceval.pipeline.batch_workflow import (
    BatchRequest,
    BatchReport,
    build_batch_workflow,
)
from doceval.pipeline.evaluator import (
    Evaluator,
    build_default_evaluator,
    evaluate_many,
)
from doceval.pipeline.workflow import (
    PipelineState,
    build_pipeline_workflow,
    list_available_sources,
    run_workflow_for,
    run_workflow_many,
)

__all__ = [
    # Legacy hand-written pipeline (kept for tests / direct API use)
    "Evaluator",
    "build_default_evaluator",
    "evaluate_many",
    # Per-stem Agent-Framework Workflow pipeline
    "PipelineState",
    "build_pipeline_workflow",
    "list_available_sources",
    "run_workflow_for",
    "run_workflow_many",
    # Batch wrapper (one run = all stems + summary)
    "BatchRequest",
    "BatchReport",
    "build_batch_workflow",
]
