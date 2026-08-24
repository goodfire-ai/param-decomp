"""Bootstrap the targeted-LM process environment before importing JAX."""

import os
from pathlib import Path

import fire
import yaml

from param_decomp.experiments.lm.runtime import RuntimeConfig


def main(
    config: Path,
    data_root: Path,
    local_device_count: int,
    run_id: str | None = None,
) -> None:
    """Set the process env from the config's `runtime.launch_env`, then run the trainer
    — imported only afterwards: the env must be in place before anything imports jax."""
    runtime = RuntimeConfig.model_validate(yaml.safe_load(Path(config).read_text())["runtime"])
    os.environ.update(runtime.launch_env.as_env())
    from param_decomp.experiments.lm.training_targeted import main as train

    train(config, data_root, local_device_count, run_id)


if __name__ == "__main__":
    fire.Fire(main)
