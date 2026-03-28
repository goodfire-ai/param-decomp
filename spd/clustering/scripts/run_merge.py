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
from spd.clustering.membership_snapshot import load_membership_snapshot
from spd.clustering.merge import merge_iteration_memberships
from spd.clustering.merge_config import MergeConfig
from spd.clustering.storage import StorageBase
from spd.log import logger
from spd.utils.run_utils import ExecutionStamp

os.environ["WANDB_QUIET"] = "true"


class MergeStorage(StorageBase):
    _CONFIG = "merge_config.json"
    _HISTORY = "history.zip"

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        super().__init__(execution_stamp)
        self.config_path: Path = self.base_dir / self._CONFIG
        self.history_path: Path = self.base_dir / self._HISTORY


def merge(snapshot_path: Path, merge_config: MergeConfig) -> Path:
    execution_stamp = ExecutionStamp.create(run_type="clustering/runs", create_snapshot=False)
    storage = MergeStorage(execution_stamp)
    logger.info(f"Merge run {execution_stamp.run_id} → {storage.base_dir}")

    storage.config_path.parent.mkdir(parents=True, exist_ok=True)
    storage.config_path.write_text(
        json.dumps(
            {
                "snapshot_path": str(snapshot_path),
                "merge_config": merge_config.model_dump(mode="json"),
            },
            indent=2,
        )
    )

    snapshot = load_membership_snapshot(snapshot_path)
    logger.info(f"Loaded: {snapshot.n_components} components, {snapshot.n_samples} samples")

    history = merge_iteration_memberships(
        merge_config=merge_config,
        memberships=snapshot.to_memberships(),
        n_samples=snapshot.n_samples,
        component_labels=ComponentLabels(list(snapshot.labels)),
    )

    history.save(storage.history_path)
    logger.info(f"History saved to {storage.history_path}")
    return storage.history_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="Merge from a membership snapshot.")
    parser.add_argument("snapshot", type=Path, help="Path to harvest snapshot directory.")
    parser.add_argument("merge_config", type=Path, help="Path to MergeConfig JSON/YAML.")
    args = parser.parse_args()
    merge(snapshot_path=args.snapshot, merge_config=MergeConfig.from_file(args.merge_config))


if __name__ == "__main__":
    cli()
