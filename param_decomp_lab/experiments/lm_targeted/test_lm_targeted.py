"""Targeted-PD LM seams: the fixed-prompt loader (end-pad + recon_positions, SPEC S38), the
non-target loss-set exclusions (both PGD variants dropped, SPEC S35), and persistent-PGD
admission in the target pass (SPEC S39)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from param_decomp.configs import (
    AdamPGDConfig,
    AnyLossMetricConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    SCScope,
    SmoothL0ImportanceMinimalityLossConfig,
    StochasticReconSubsetLossConfig,
)
from param_decomp.run import TargetPromptGeometry
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.experiments.config import build_nontarget_loss_metrics
from param_decomp_lab.experiments.lm_targeted.config import LMTargetedExperimentConfig
from param_decomp_lab.experiments.lm_targeted.data import load_prompt_tokens

_NEOX = "EleutherAI/gpt-neox-20b"


def test_load_prompt_tokens_end_pads_and_returns_recon_positions(tmp_path: Path):
    f = tmp_path / "prompts.txt"
    f.write_text("import numpy as\nimport pandas as\n")  # both 3 neox tokens
    tokens, recon_positions = load_prompt_tokens(
        str(f), _NEOX, max_seq_len=16, add_special_tokens=False
    )
    assert tokens.shape == (2, 16)
    assert recon_positions == 3
    # real tokens in the first 3 columns, END-padded with 0 after
    assert (tokens[:, 3:] == 0).all()
    assert (tokens[:, :3] != 0).all()
    assert tokens.dtype == np.int32


def test_load_prompt_tokens_rejects_mixed_lengths(tmp_path: Path):
    f = tmp_path / "prompts.txt"
    f.write_text("import numpy as\nimport pandas\n")  # different token counts
    with pytest.raises(AssertionError):
        load_prompt_tokens(str(f), _NEOX, max_seq_len=16, add_special_tokens=False)


def test_load_prompt_tokens_rejects_over_length(tmp_path: Path):
    f = tmp_path / "prompts.txt"
    f.write_text("import numpy as\n")
    with pytest.raises(AssertionError):
        load_prompt_tokens(str(f), _NEOX, max_seq_len=2, add_special_tokens=False)  # 3 tokens > 2


def test_load_prompt_tokens_prepends_special_tokens_when_requested(tmp_path: Path):
    f = tmp_path / "prompts.txt"
    f.write_text("import numpy as\nimport pandas as\n")
    without = load_prompt_tokens(str(f), _NEOX, max_seq_len=16, add_special_tokens=False)
    with_special = load_prompt_tokens(str(f), _NEOX, max_seq_len=16, add_special_tokens=True)
    # gpt-neox prepends no BOS, so the two agree; the flag threads through regardless. The
    # BOS-bearing case (Llama) is exercised end-to-end by the arithmetic config's matching
    # add_special_tokens=True vs the ArithmeticCIGrid probe.
    assert without[1] == with_special[1]


def test_nontarget_loss_set_drops_both_pgd_variants():
    target: list[AnyLossMetricConfig] = [
        FaithfulnessLossConfig(coeff=0.0),
        ImportanceMinimalityLossConfig(
            coeff=1e-3,
            pnorm=2.0,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=1.0,
            p_anneal_end_frac=1.0,
        ),  # fmt: skip
        StochasticReconSubsetLossConfig(coeff=1.0),
        PGDReconLossConfig(coeff=0.5, init="random", n_steps=1, step_size=0.1, mask_scope="c"),
    ]
    out = build_nontarget_loss_metrics(target, impmin_coeff_ratio=2.0)
    kinds = {type(m).__name__ for m in out}
    assert PGDReconLossConfig.__name__ not in kinds  # fresh-PGD excluded from non-target (S35)
    assert StochasticReconSubsetLossConfig.__name__ in kinds  # full-model recon retained
    assert FaithfulnessLossConfig.__name__ in kinds


def _persistent_pgd() -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99, lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)
        ),
        n_warmup_steps=2,
    )


def test_nontarget_loss_set_scales_smooth_l0_impmin():
    target: list[AnyLossMetricConfig] = [
        FaithfulnessLossConfig(coeff=0.0),
        SmoothL0ImportanceMinimalityLossConfig(coeff=1e-5, gamma=1.0),
        StochasticReconSubsetLossConfig(coeff=1.0),
    ]
    out = build_nontarget_loss_metrics(target, impmin_coeff_ratio=2.0)
    smooth = [m for m in out if isinstance(m, SmoothL0ImportanceMinimalityLossConfig)]
    assert len(smooth) == 1 and smooth[0].coeff == pytest.approx(2e-5)  # S37


def test_nontarget_loss_set_drops_persistent_pgd():
    target: list[AnyLossMetricConfig] = [
        FaithfulnessLossConfig(coeff=0.0),
        StochasticReconSubsetLossConfig(coeff=1.0),
        _persistent_pgd(),
    ]
    out = build_nontarget_loss_metrics(target, impmin_coeff_ratio=2.0)
    kinds = {type(m).__name__ for m in out}
    assert PersistentPGDReconLossConfig.__name__ not in kinds  # target-pass only (S35/S39)
    assert StochasticReconSubsetLossConfig.__name__ in kinds


def test_targeted_config_accepts_persistent_pgd():
    ref = (
        Path(__file__).parent / "configs" / "numpy_pandas_4L_targeted.yaml"
    )  # S39: the tPD schema admits a persistent term (it sizes off the target seq at init)
    raw = yaml.safe_load(ref.read_text())
    raw["pd"]["loss_metrics"].append(_persistent_pgd().model_dump())
    cfg = LMTargetedExperimentConfig(**raw)
    kinds = [type(m).__name__ for m in cfg.pd.loss_metrics]
    assert PersistentPGDReconLossConfig.__name__ in kinds


def test_target_prompt_geometry_bounds():
    geo = TargetPromptGeometry(recon_positions=3, seq_len=16)
    assert (geo.recon_positions, geo.seq_len) == (3, 16)
    with pytest.raises(AssertionError):
        TargetPromptGeometry(recon_positions=17, seq_len=16)
    with pytest.raises(AssertionError):
        TargetPromptGeometry(recon_positions=0, seq_len=16)
