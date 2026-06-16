"""train/loss/* logged-key namespace parity (issue #647).

The torch trainer logs one `train/loss/<instance_key>` scalar per active loss metric
plus `train/loss/total` (`train_step.run_loss_step` emits `loss/<instance_key>` +
`loss/total`; `optimize.py` prefixes `train/`). The JAX step emits its own fixed
scalars, mapped to wandb keys by `run._METRIC_KEYS`, plus `loss/<term.name>` per recon
term (the sink prefixes `train/`). For a torch-vs-jax run pair to overlay on one panel
the `train/loss/*` key SETS must be identical; JAX-only extras are tolerated ONLY on the
namespaces that are JAX-only by design (`train/perf/*`, `mem/*`, `train/schedules/*`,
`train/lr/*`).

`ReconLossTerm.name` is the torch `instance_key` (`cfg.name or cfg.type`), so the recon
half is parity-by-construction; this test pins it and surfaces any non-recon drift.
"""

from pathlib import Path

import pytest
import yaml

from jax_single_pool.recon import LossSpec, build_recon_terms
from jax_single_pool.run import _METRIC_KEYS
from jax_single_pool.torch_config import load_torch_wrapper
from param_decomp_config.losses import (
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LossMetricConfig,
)

CONFIGS = Path(__file__).parent.parent / "configs"
RUN_ID = "p-0123abcd"

# Namespaces the JAX trainer emits with no torch counterpart, by design (extras here are
# allowed). Keys under any OTHER namespace must match torch's train/loss/* set.
JAX_ONLY_NAMESPACE_PREFIXES = ("train/perf/", "train/mem/", "mem/", "train/schedules/", "train/lr/")


def _stamped_wrapper(tmp_path: Path, wrapper: Path) -> Path:
    raw = yaml.safe_load(wrapper.read_text())
    raw["torch_config"] = str((wrapper.parent / raw["torch_config"]).resolve())
    raw["run_id"] = RUN_ID
    stamped = tmp_path / wrapper.name
    stamped.write_text(yaml.safe_dump(raw))
    return stamped


def _torch_instance_key(cfg: LossMetricConfig) -> str:
    """What torch's `Metric.instance_key` resolves to: `cfg.name` else the class name,
    which equals the discriminator literal `cfg.type` for every loss metric class."""
    return cfg.name if cfg.name is not None else cfg.type


def _torch_train_loss_keys(loss_metrics: tuple[LossMetricConfig, ...]) -> set[str]:
    """`train/loss/*` keys the torch trainer logs: one per loss metric instance
    (`run_loss_step` -> `loss/<instance_key>`) plus the summed `loss/total`."""
    keys = {f"train/loss/{_torch_instance_key(cfg)}" for cfg in loss_metrics}
    keys.add("train/loss/total")
    return keys


def _jax_train_loss_keys(spec: LossSpec) -> set[str]:
    """`train/loss/*` keys the JAX trainer logs: the step's fixed scalars mapped through
    `_METRIC_KEYS` (whichever land under `train/loss/`) plus `loss/<term.name>` per recon
    term, `train/`-prefixed by the sink."""
    fixed = {v for v in _METRIC_KEYS.values() if v.startswith("train/loss/")}
    recon = {f"train/loss/{t.name}" for t in spec.recon_terms}
    return fixed | recon


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN train/loss/* divergence (issue #647): the JAX step logs "
        "train/loss/ImportanceMinimalityLoss_no_beta from `imp_no_beta` (run._METRIC_KEYS), "
        "but torch emits the `_no_beta` diagnostic only via ImportanceMinimalityLoss.compute() "
        "on the eval namespace -- run_loss_step's train path logs only the single scalar from "
        "update(). All other train/loss/* keys (total, FaithfulnessLoss, "
        "ImportanceMinimalityLoss, recon-term names) match. Remove this marker once the "
        "namespaces are reconciled (drop the JAX train-side _no_beta, or add a torch "
        "train-side counterpart)."
    ),
)
def test_train_loss_key_sets_match_b128_production(tmp_path: Path):
    converted, _torch_path, _raw = load_torch_wrapper(
        _stamped_wrapper(tmp_path, CONFIGS / "llama8b_l18_b128_cmp32_from_torch.yaml")
    )
    loss_metrics = converted.loss_metrics
    spec = build_recon_terms(
        loss_metrics,
        tuple(sc.name for sc in converted.target.sites),
        converted.n_mask_samples,
        converted.sampling,
    )

    # Sanity: the production config carries the two non-recon terms + the recon terms.
    assert any(isinstance(c, FaithfulnessLossConfig) for c in loss_metrics)
    assert any(isinstance(c, ImportanceMinimalityLossConfig) for c in loss_metrics)

    torch_keys = _torch_train_loss_keys(loss_metrics)
    jax_keys = _jax_train_loss_keys(spec)

    torch_only = torch_keys - jax_keys
    jax_only = jax_keys - torch_keys

    assert not torch_only, f"torch logs train/loss keys JAX never emits: {sorted(torch_only)}"
    unexpected_jax_only = {k for k in jax_only if not k.startswith(JAX_ONLY_NAMESPACE_PREFIXES)}
    assert not unexpected_jax_only, (
        "JAX emits train/loss keys with no torch train-side counterpart "
        f"(not on a JAX-only namespace): {sorted(unexpected_jax_only)}"
    )
