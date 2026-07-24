"""The process environment, resolved once into one explicit object.

`Environment.from_env` is the ONLY place the library reads ambient environment
variables for paths; everything else consumes the typed `ENV` singleton. The library
knows NO cluster facts — no data mounts, no partitions, no team namespaces (those live
in the deployment wrapper, e.g. `param_decomp_goodfire.env`, whose submitters export
the resolved `PARAM_DECOMP_OUT_DIR` into each job).
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Environment:
    repo_root: Path
    output_root: Path
    """Where runs / logs / caches land (`PARAM_DECOMP_OUT_DIR`, default `./out`)."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Environment":
        repo_root = (
            Path(env["GITHUB_WORKSPACE"])
            if ("CI" in env and "GITHUB_WORKSPACE" in env)
            else Path(__file__).parent.parent.parent
        )
        return cls(
            repo_root=repo_root,
            output_root=Path(env.get("PARAM_DECOMP_OUT_DIR", "out")),
        )


ENV = Environment.from_env()
