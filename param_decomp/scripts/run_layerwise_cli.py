from pathlib import Path

import fire

from param_decomp.settings import DEFAULT_PARTITION_NAME, REPO_ROOT


def main(
    orchestrator: str,
    partition: str = DEFAULT_PARTITION_NAME,
    max_concurrent: int = 8,
) -> None:
    """Launch a layerwise-split PD training run.

    Args:
        orchestrator: Path to the orchestrator YAML — a fully normalised Config whose
            `module_info` enumerates every target module to decompose.
        partition: SLURM partition name.
        max_concurrent: Cap on concurrent single-GPU array tasks (cluster limit is 8).

    Example:

        pd-run-layerwise param_decomp/experiments/lm/jose_layerwise.yaml
    """
    from param_decomp.scripts.run_layerwise import launch_layerwise_run

    path = Path(orchestrator)
    if not path.is_absolute() and not path.exists():
        candidate = REPO_ROOT / orchestrator
        if candidate.exists():
            path = candidate
    assert path.exists(), f"orchestrator config not found: {orchestrator}"

    launch_layerwise_run(
        orchestrator_path=path,
        partition=partition,
        max_concurrent_tasks=max_concurrent,
    )


def cli():
    fire.Fire(main)
