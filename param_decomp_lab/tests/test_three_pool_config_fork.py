"""Parse-time constraint tests for the Option-C 3-pool config fork.

The whole point of `ThreePoolConstrainedPDConfig` + `ThreePoolLMExperimentConfig` is to
move 3-pool misconfiguration failures from "minutes into a multi-node launch" to "YAML
parse on the login node". These tests pin that: a valid config parses, and each class of
invalid config (wrong fixed scalar, missing/extra loss, bad batch divisibility, rank-0
convention) fails at `model_validate`.
"""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp_lab.experiments.lm.three_pool_pd import (
    ThreePoolConstrainedPDConfig,
    ThreePoolLosses,
)
from param_decomp_lab.experiments.lm.three_pool_run import ThreePoolLMExperimentConfig

_VALID_YAML = (
    Path(__file__).parents[1]
    / "experiments/lm/_resumption_validation/gpt2_small_3pool_save_repro.yaml"
)


def _valid_dict() -> dict[str, Any]:
    return yaml.safe_load(_VALID_YAML.read_text())


def test_valid_config_parses() -> None:
    cfg = ThreePoolLMExperimentConfig.model_validate(_valid_dict())
    assert [m.type for m in cfg.pd.loss_metrics] == [
        "FaithfulnessLoss",
        "ImportanceMinimalityLoss",
        "StochasticReconLayerwiseLoss",
        "PersistentPGDReconLoss",
    ]
    # Fixed scalars take their frozen Literal defaults.
    assert cfg.pd.sampling == "continuous"
    assert cfg.pd.n_mask_samples == 1
    assert cfg.pd.use_delta_component is True
    assert cfg.pd.identity_decomposition_targets is None


def test_pd_dump_roundtrips_through_constrained_config() -> None:
    """The snapshot path dumps `pd` and re-validates as the constrained type on resume.

    `loss_metrics` is excluded from the dump (derived from `losses`), so the round-trip
    must reconstruct it.
    """
    cfg = ThreePoolLMExperimentConfig.model_validate(_valid_dict())
    dumped = cfg.pd.model_dump()
    assert "loss_metrics" not in dumped
    assert "losses" in dumped
    reloaded = ThreePoolConstrainedPDConfig.model_validate(dumped)
    assert [m.type for m in reloaded.loss_metrics] == [m.type for m in cfg.pd.loss_metrics]


@pytest.mark.parametrize("scalar,bad", [("n_mask_samples", 3), ("sampling", "binomial")])
def test_wrong_fixed_scalar_rejected(scalar: str, bad: Any) -> None:
    data = _valid_dict()
    data["pd"][scalar] = bad
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_use_delta_component_false_rejected() -> None:
    data = _valid_dict()
    data["pd"]["use_delta_component"] = False
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_missing_loss_rejected() -> None:
    data = _valid_dict()
    del data["pd"]["losses"]["ppgd"]
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_extra_forbidden_loss_rejected() -> None:
    """An extra key on the losses struct is forbidden (extra='forbid')."""
    data = _valid_dict()
    data["pd"]["losses"]["unmasked"] = {"coeff": 1.0}
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_authoring_loss_metrics_directly_rejected() -> None:
    data = _valid_dict()
    data["pd"]["loss_metrics"] = [{"type": "FaithfulnessLoss", "coeff": 1.0}]
    with pytest.raises(ValidationError):
        ThreePoolConstrainedPDConfig.model_validate(data["pd"])


def test_missing_coeff_rejected() -> None:
    losses = _valid_dict()["pd"]["losses"]
    del losses["faith"]["coeff"]
    with pytest.raises(ValidationError):
        ThreePoolLosses.model_validate(losses)


def test_ppgd_start_frac_nonzero_rejected() -> None:
    data = _valid_dict()
    data["pd"]["losses"]["ppgd"]["start_frac"] = 0.5
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_batch_not_divisible_by_topology_rejected() -> None:
    """Cross-field check: batch_size must be divisible by each pool arity. The topology
    stays valid (uniform N_per_block=2); only batch_size is made indivisible by it."""
    data = _valid_dict()
    data["pd"]["batch_size"] = 3  # N_per_block=2 does not divide 3
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_rank0_not_lw_leader_rejected() -> None:
    """Cross-field check: rank 0 must be the LW pool's block-0 leader."""
    data = _valid_dict()
    groups = data["runtime"]["topology"]["layerwise_block_groups"]
    # Swap rank 0 out of the first block's leader slot (move it to CI pool's place).
    groups[0]["ranks"] = [9, 1]
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_topology_under_runtime_not_three_pool_field() -> None:
    """The topology lives on `runtime.topology`; the old top-level `three_pool` key is
    no longer accepted (extra='forbid')."""
    data = _valid_dict()
    topology = copy.deepcopy(data["runtime"]["topology"])
    data["three_pool"] = topology
    with pytest.raises(ValidationError):
        ThreePoolLMExperimentConfig.model_validate(data)


def test_resume_provenance_field_defaults_none_and_roundtrips() -> None:
    """Fresh config has `resume_provenance is None`; a populated one survives a dump +
    reload (so it lands in run_meta.yaml / wandb.config)."""
    from param_decomp_lab.resumption import ResumeProvenance

    cfg = ThreePoolLMExperimentConfig.model_validate(_valid_dict())
    assert cfg.resume_provenance is None

    resumed = cfg.model_copy(
        update={
            "resume_provenance": ResumeProvenance(
                parent_run_dir=Path("/runs/p-parent"), parent_step=5000
            )
        }
    )
    reloaded = ThreePoolLMExperimentConfig.model_validate(resumed.model_dump(mode="json"))
    assert reloaded.resume_provenance is not None
    assert reloaded.resume_provenance.parent_step == 5000
    assert reloaded.resume_provenance.parent_run_dir == Path("/runs/p-parent")
