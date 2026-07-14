"""Logged-key pins: every loss term logs under `train/loss/<term.name>` where
`term.name` is `cfg.name or cfg.type`, and `MetricsSink.log` rejects keys outside the
canonical namespace tree (`run._WANDB_KEY_NAMESPACES`).

The `type` literals are the torch-era class names — frozen when the JAX trainer took
over so run pairs overlay across the transition, kept now for continuity of the
existing wandb corpus.
"""

import json
from pathlib import Path

import pytest

from param_decomp.configs import (
    AdamPGDConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SCScope,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.recon import ReconLossTerm, build_loss_terms
from param_decomp.run import MetricsSink
from param_decomp.schedule import ScheduleConfig

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
)


def _non_recon_configs() -> tuple[FaithfulnessLossConfig, ImportanceMinimalityLossConfig]:
    return (
        FaithfulnessLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(
            coeff=1.0, pnorm=ScheduleConfig(start_val=0.9, fn_type="constant")
        ),
    )


def _recon_terms(recon_configs: tuple[object, ...]) -> tuple[ReconLossTerm, ...]:
    terms = build_loss_terms(
        (*_non_recon_configs(), *recon_configs),  # pyright: ignore[reportArgumentType]
        site_names=SITE_NAMES,
    )
    return terms.recon


def test_recon_term_name_is_config_type_by_default():
    """Each recon term's log-key suffix (`term.name`) is the config `type` literal when
    no `name` override is set."""
    emitted = {term.name for term in _recon_terms(RECON_CONFIGS)}
    expected = {cfg.type for cfg in RECON_CONFIGS}
    assert emitted == expected, emitted ^ expected


def test_recon_term_name_honors_name_override():
    """A `name` override on the config flows to `term.name` (and so to the
    `train/loss/<name>` key)."""
    cfg = StochasticReconLossConfig(coeff=1.0, name="StochasticReconLoss_probe")
    (term,) = _recon_terms((cfg,))
    assert term.name == "StochasticReconLoss_probe"


def test_fixed_term_names_are_the_frozen_class_literals():
    """The faith/imp singleton terms carry the frozen class-name literals — the
    `train/loss/<term.name>` keys existing panels select on."""
    faith_cfg, imp_cfg = _non_recon_configs()
    terms = build_loss_terms(
        (faith_cfg, imp_cfg, StochasticReconLossConfig(coeff=1.0)),
        site_names=SITE_NAMES,
    )
    assert terms.faith.name == "FaithfulnessLoss"
    assert terms.imp.name == "ImportanceMinimalityLoss"


def test_sink_rejects_keys_outside_the_canonical_tree(tmp_path: Path):
    """`MetricsSink.log` asserts every key lives under a declared namespace — a bare or
    typo'd key fails fast instead of drifting to the wandb top level."""
    jsonl_path = tmp_path / "metrics.jsonl"
    with jsonl_path.open("a") as jsonl:
        sink = MetricsSink(jsonl=jsonl, wandb_module=None)
        sink.log(1, {"train/loss/total": 1.0, "eval/ce_kl/kl_ci_masked": 0.5})
        with pytest.raises(AssertionError, match="canonical wandb tree"):
            sink.log(2, {"step_time_s": 0.1})
        with pytest.raises(AssertionError, match="canonical wandb tree"):
            sink.log(3, {"train/losss/total": 1.0})
    logged = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert logged == [{"step": 1, "train/loss/total": 1.0, "eval/ce_kl/kl_ci_masked": 0.5}]
