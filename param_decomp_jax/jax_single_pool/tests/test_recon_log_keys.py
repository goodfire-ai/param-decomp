"""E11 logged-key parity: every `train/loss/*` key a JAX run emits must equal the
torch key for the same config, so a torch-vs-jax run pair overlays on one wandb panel.

Two emit paths feed `train/loss/*`:

  * Fixed scalar terms (`faith`, `imp`) — mapped through `run._METRIC_KEYS`.
  * Recon terms — arrive from the jitted step already shaped `loss/<term.name>`
    (`train.py`), then `train/`-prefixed by the sink.

`term.name` is set by `recon.build_recon_terms` to `cfg.name or cfg.type`. Torch's
`Metric.instance_key` is `cfg.name or type(self).__name__`. They agree byte-for-byte
because torch's `LOSS_METRIC_CLASSES` is keyed by `cls.__name__` and dispatch does
`LOSS_METRIC_CLASSES[cfg.type](cfg)` — so torch can only run a config whose `type`
literal equals the class name. This test pins the JAX half (`term.name == cfg.type`);
the torch half (`cls.__name__ == cfg.type`) was pinned against live torch dispatch
before push-1 severed the torch import — it now holds by the frozen `type` literals.
"""

from jax_single_pool.recon import build_recon_terms
from param_decomp_config.losses import (
    AdamPGDConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SCScope,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp_config.schedule import ScheduleConfig

SITE_NAMES = ("h.0.mlp.c_fc", "h.0.mlp.down_proj")


def _persistent_optimizer() -> AdamPGDConfig:
    # `_assert_supported_persistent` requires a constant, full-value lr schedule.
    return AdamPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1, fn_type="constant"))


# Every recon loss config this trainer implements, one instance each. Order is
# irrelevant here (each becomes its own term); coeffs are arbitrary-positive.
RECON_CONFIGS = (
    CIMaskedReconLossConfig(coeff=1.0),
    CIMaskedReconLayerwiseLossConfig(coeff=1.0),
    CIMaskedReconSubsetLossConfig(coeff=1.0),
    UnmaskedReconLossConfig(coeff=1.0),
    StochasticReconLossConfig(coeff=1.0),
    StochasticReconLayerwiseLossConfig(coeff=1.0),
    StochasticReconSubsetLossConfig(coeff=1.0),
    PGDReconLossConfig(coeff=1.0, init="random", step_size=0.1, n_steps=1, mask_scope="bsc"),
    PGDReconLayerwiseLossConfig(
        coeff=1.0, init="random", step_size=0.1, n_steps=1, mask_scope="bsc"
    ),
    PGDReconSubsetLossConfig(coeff=1.0, init="random", step_size=0.1, n_steps=1, mask_scope="bsc"),
    PersistentPGDReconLossConfig(coeff=1.0, optimizer=_persistent_optimizer(), scope=SCScope()),
    PersistentPGDReconSubsetLossConfig(
        coeff=1.0, optimizer=_persistent_optimizer(), scope=SCScope()
    ),
)


def _non_recon_configs() -> tuple[FaithfulnessLossConfig, ImportanceMinimalityLossConfig]:
    return (
        FaithfulnessLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(coeff=1.0, pnorm=0.9, beta=0.0, p_anneal_final_p=0.9),
    )


def _build(recon_configs: tuple[object, ...]):
    return build_recon_terms(
        (*_non_recon_configs(), *recon_configs),  # pyright: ignore[reportArgumentType]
        site_names=SITE_NAMES,
        n_mask_samples=1,
        sampling="continuous",
    )


def test_recon_term_name_is_config_type_by_default():
    """Each recon term's log-key suffix (`term.name`) is the config `type` literal when
    no `name` override is set — the JAX half of torch's `instance_key`."""
    spec = _build(RECON_CONFIGS)
    emitted = {term.name for term in spec.recon_terms}
    expected = {cfg.type for cfg in RECON_CONFIGS}
    assert emitted == expected, emitted ^ expected


def test_recon_term_name_honors_name_override():
    """A `name` override on the config flows to `term.name`, exactly as torch's
    `instance_key` returns `cfg.name` when set."""
    cfg = StochasticReconLossConfig(coeff=1.0, name="StochasticReconLoss_probe")
    (term,) = _build((cfg,)).recon_terms
    assert term.name == "StochasticReconLoss_probe"


def test_fixed_scalar_loss_keys_use_torch_class_names():
    """`run._METRIC_KEYS` maps the two non-recon scalar terms to `train/loss/<ClassName>`;
    pin those suffixes to the config `type` literals (== torch class names)."""
    from jax_single_pool.run import _METRIC_KEYS  # pyright: ignore[reportPrivateUsage]

    faith, imp = _non_recon_configs()
    assert _METRIC_KEYS["faith"] == f"train/loss/{faith.type}"
    assert _METRIC_KEYS["imp"] == f"train/loss/{imp.type}"
