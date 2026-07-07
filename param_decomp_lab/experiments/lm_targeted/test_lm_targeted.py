"""Targeted-PD LM seams: the fixed-prompt loader (end-pad + recon_positions, SPEC S38) and
the non-target loss-set exclusions (both PGD variants dropped, SPEC S35)."""

from pathlib import Path

import numpy as np
import pytest

from param_decomp.configs import (
    AnyLossMetricConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PGDReconLossConfig,
    StochasticReconSubsetLossConfig,
)
from param_decomp_lab.experiments.config import build_nontarget_loss_metrics
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
