"""Runtime guardrails for All of Us participant-level workflows."""

import os


def is_aou_workspace_run() -> bool:
    """Return whether the process appears to run against an AoU CDR workspace."""

    return bool(os.getenv("WORKSPACE_CDR"))


def validate_external_logging_disabled(enable_wandb: bool) -> None:
    """Fail closed if external logging is requested for an AoU workspace run."""

    if is_aou_workspace_run() and enable_wandb:
        raise RuntimeError(
            "External W&B logging is disabled for All of Us participant-level runs."
        )
