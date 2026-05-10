"""Residual MLP decomposition entrypoint."""

from pathlib import Path

import fire

from param_decomp.experiments.resid_mlp.driver import DRIVER
from param_decomp.experiments.runner import main as run_with_driver


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    run_with_driver(
        config_path=config_path,
        config_json=config_json,
        driver=DRIVER.driver_path,
        evals_id=evals_id,
        launch_id=launch_id,
        sweep_params_json=sweep_params_json,
        run_id=run_id,
    )


if __name__ == "__main__":
    fire.Fire(main)
