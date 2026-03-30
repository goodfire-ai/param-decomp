"""Run merge iteration on a pre-harvested membership snapshot.

No GPU required — purely CPU work.

Output:
    <SPD_OUT_DIR>/clustering/runs/<run_id>/
        ├── merge_config.json
        └── history.zip
"""

import argparse
import json
import os
from pathlib import Path

from spd.clustering.consts import ComponentLabels
from spd.clustering.memberships import ProcessedMemberships
from spd.clustering.merge import LogCallback, merge_iteration_memberships
from spd.clustering.merge_config import MergeConfig
from spd.clustering.paths import clustering_run_dir, new_run_id
from spd.log import logger

os.environ["WANDB_QUIET"] = "true"


def merge(
    snapshot_path: Path,
    merge_config: MergeConfig,
    run_id: str,
    log_callback: LogCallback | None = None,
) -> Path:
    """Run merge iteration, return history path."""
    out = clustering_run_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Merge run {run_id} → {out}")

    (out / "merge_config.json").write_text(
        json.dumps(
            {
                "snapshot_path": str(snapshot_path),
                "merge_config": merge_config.model_dump(mode="json"),
            },
            indent=2,
        )
    )

    processed = ProcessedMemberships.load(snapshot_path)
    logger.info(f"Loaded: {processed.n_components_alive} components, {processed.n_samples} samples")

    history = merge_iteration_memberships(
        merge_config=merge_config,
        memberships=processed.memberships,
        n_samples=processed.n_samples,
        component_labels=ComponentLabels(list(processed.labels)),
        log_callback=log_callback,
    )

    history_path = out / "history.zip"
    history.save(history_path)
    logger.info(f"History saved to {history_path}")
    return history_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Merge from a membership snapshot.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("merge_config", type=Path)
    args = parser.parse_args()
    merge(
        snapshot_path=args.snapshot,
        merge_config=MergeConfig.from_file(args.merge_config),
        run_id=new_run_id(),
    )


if __name__ == "__main__":
    cli()
