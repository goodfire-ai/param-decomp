"""CLI entry point for investigation SLURM launcher.

Usage:
    spd-investigate <wandb_path> "<prompt>"
    spd-investigate <wandb_path> @prompt.txt
    spd-investigate <wandb_path> "<prompt>" --max_turns 30
"""

from pathlib import Path

import fire

from spd.settings import DEFAULT_PARTITION_NAME


def _resolve_prompt(prompt: str) -> str:
    """If prompt starts with @, read from that file path. Otherwise return as-is."""
    if prompt.startswith("@"):
        path = Path(prompt[1:])
        assert path.exists(), f"Prompt file not found: {path}"
        return path.read_text().strip()
    return prompt


def main(
    wandb_path: str,
    prompt: str,
    context_length: int = 128,
    max_turns: int = 50,
    partition: str = DEFAULT_PARTITION_NAME,
    time: str = "8:00:00",
    job_suffix: str | None = None,
) -> None:
    """Launch a single investigation agent for a specific question.

    Args:
        wandb_path: WandB run path for the SPD decomposition to investigate.
        prompt: The research question, or @filepath to read from a file.
        context_length: Context length for prompts (default 128).
        max_turns: Maximum agentic turns (default 50, prevents runaway).
        partition: SLURM partition name.
        time: Job time limit (default 8 hours).
        job_suffix: Optional suffix for SLURM job names.
    """
    from spd.investigate.scripts.run_slurm import launch_investigation

    launch_investigation(
        wandb_path=wandb_path,
        prompt=_resolve_prompt(prompt),
        context_length=context_length,
        max_turns=max_turns,
        partition=partition,
        time=time,
        job_suffix=job_suffix,
    )


def cli() -> None:
    fire.Fire(main)
