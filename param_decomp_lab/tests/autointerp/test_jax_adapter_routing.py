"""The PD adapter routing discriminator: a JAX single-pool run dir (orbax `ckpts/` +
a `pd-jax-lm` wrapper `config.yaml` carrying `torch_config:`) routes to `JaxPDAdapter`,
not the torch `PDAdapter` (which needs a `model_*.pth` JAX runs don't have)."""

from pathlib import Path

import pytest

from param_decomp_lab.adapters import jax_pd
from param_decomp_lab.adapters.jax_pd import is_jax_run


@pytest.fixture
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(jax_pd, "get_harvest_dir", lambda rid: tmp_path / "runs" / rid / "harvest")
    return tmp_path / "runs"


def _make_run(runs_root: Path, run_id: str, config_yaml: str | None) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    if config_yaml is not None:
        (run_dir / "config.yaml").write_text(config_yaml)


def test_jax_wrapper_is_detected(runs_root: Path):
    _make_run(
        runs_root,
        "p-jax0001",
        "torch_config: torch/foo.yaml\nrun_name: r\nrun_id: p-jax0001\n",
    )
    assert is_jax_run("p-jax0001")


def test_torch_run_without_wrapper_key_is_not_jax(runs_root: Path):
    _make_run(runs_root, "p-torch001", "pd:\n  seed: 0\nruntime:\n  device: cuda:0\n")
    assert not is_jax_run("p-torch001")


def test_missing_config_is_not_jax(runs_root: Path):
    _make_run(runs_root, "p-bare0001", config_yaml=None)
    assert not is_jax_run("p-bare0001")
