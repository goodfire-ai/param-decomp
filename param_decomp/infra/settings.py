"""The process environment, resolved once into one explicit object.

`Environment.from_env` is the ONLY place the library reads ambient environment
variables for paths / infra fit; everything else consumes the typed `ENV` singleton.
The fields are enumerated — a new environment dependency is added here deliberately,
never sniffed inline at a use site.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Environment:
    repo_root: Path
    data_mount: Path | None
    """Cluster shared-data mount (`DATA_MOUNT`); None off-cluster (or if the mount
    doesn't exist — a set-but-dead `DATA_MOUNT` is treated as off-cluster)."""
    output_root: Path
    """Where runs / logs / scripts / caches land (`PARAM_DECOMP_OUT_DIR`; defaults
    under the data mount on a cluster, `./out` elsewhere)."""
    default_partition: str | None
    """sbatch `--partition` (`PARTITION_RESERVED`); None → the cluster's default."""

    @property
    def slurm_logs_dir(self) -> Path:
        return self.output_root / "slurm_logs"

    @property
    def sbatch_scripts_dir(self) -> Path:
        return self.output_root / "sbatch_scripts"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Environment":
        repo_root = (
            Path(env["GITHUB_WORKSPACE"])
            if ("CI" in env and "GITHUB_WORKSPACE" in env)
            else Path(__file__).parent.parent.parent
        )
        raw_mount = env.get("DATA_MOUNT")
        data_mount = Path(raw_mount) if raw_mount and Path(raw_mount).exists() else None
        default_out = (
            data_mount / "artifacts/mechanisms/param-decomp" if data_mount else Path("out")
        )
        return cls(
            repo_root=repo_root,
            data_mount=data_mount,
            output_root=Path(env.get("PARAM_DECOMP_OUT_DIR", default_out)),
            default_partition=env.get("PARTITION_RESERVED"),
        )


ENV = Environment.from_env()
