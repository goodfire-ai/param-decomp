"""Bootstrap the LM process environment before importing JAX."""

import os
import sys
from pathlib import Path

import yaml

from param_decomp.experiments.lm.runtime import RuntimeConfig


def main() -> None:
    assert len(sys.argv) >= 2 and not sys.argv[1].startswith("-"), (
        "usage: python -m param_decomp.experiments.lm.run <config.yaml>"
        " --data-root <path> [--run-id ...]"
    )
    raw = yaml.safe_load(Path(sys.argv[1]).read_text())
    runtime = RuntimeConfig.model_validate(raw["runtime"])
    os.environ.update(runtime.launch_env.as_env())
    from param_decomp.experiments.lm.training import cli

    cli()


if __name__ == "__main__":
    main()
