"""Runtime guardrails for All of Us participant-level workflows."""

import os


def is_aou_workspace_run() -> bool:
    """Return whether the process appears to run against an AoU CDR workspace."""

    return bool(os.getenv("WORKSPACE_CDR"))


def validate_external_logging(enable_wandb: bool, *, synthetic_mode: bool) -> None:
    """Allow external logging only for explicitly declared synthetic/public runs."""

    if not enable_wandb:
        return

    if is_aou_workspace_run() or not synthetic_mode:
        raise RuntimeError(
            "External logging is permitted only for explicitly declared "
            "synthetic-data runs outside an AoU workspace."
        )


def default_mixed_precision(cuda_available: bool) -> str:
    """Select a safe default Accelerator mixed-precision mode."""

    return "fp16" if cuda_available else "no"
