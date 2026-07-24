"""The Goodfire deployment environment — the cluster facts the generic library refuses to know.

`GoodfireEnvironment.from_env` is the wrapper's single ambient read: the shared data
mount, the team's artifact namespace as the on-cluster output root, and the SLURM
partition. The generic library reads no ambient path environment at all — every
submitter passes the resolved `output_root` as an explicit argument (`--out_root` /
`out_dir` config stamp) in each command it renders.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoodfireEnvironment:
    repo_root: Path
    data_mount: Path | None
    """Cluster shared-data mount (`DATA_MOUNT`); None off-cluster (or set-but-dead)."""
    output_root: Path
    """`DATA_MOUNT/artifacts/mechanisms/param-decomp` on a cluster, else `./out` — the
    team convention the library deliberately doesn't know."""
    default_partition: str | None
    """sbatch `--partition` (`PARTITION_RESERVED`); None → the cluster's default."""

    @property
    def slurm_logs_dir(self) -> Path:
        return self.output_root / "slurm_logs"

    @property
    def sbatch_scripts_dir(self) -> Path:
        return self.output_root / "sbatch_scripts"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "GoodfireEnvironment":
        repo_root = (
            Path(env["GITHUB_WORKSPACE"])
            if ("CI" in env and "GITHUB_WORKSPACE" in env)
            else Path(__file__).parent.parent
        )
        raw_mount = env.get("DATA_MOUNT")
        data_mount = Path(raw_mount) if raw_mount and Path(raw_mount).exists() else None
        return cls(
            repo_root=repo_root,
            data_mount=data_mount,
            output_root=(
                data_mount / "artifacts/mechanisms/param-decomp" if data_mount else Path("out")
            ),
            default_partition=env.get("PARTITION_RESERVED"),
        )


GENV = GoodfireEnvironment.from_env()
