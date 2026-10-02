"""The targeted (tPD) LM seat: prompt-pool construction (local stub tokenizer —
no hub dependency), the pure per-step pool batch, the config shape's refusals, and
a targeted variant of the canonical Llama recipe."""

from pathlib import Path

import numpy as np
import pytest

from param_decomp.core.components import DenseFactorization, SiteSpec
from param_decomp.core.objective import build_targeted_objective
from param_decomp.experiments.lm.arithmetic_probe import build_arithmetic_probe
from param_decomp.experiments.lm.config import LMExperimentConfig, LMTargetedExperimentConfig
from param_decomp.experiments.lm.targeted_data import (
    ArithmeticGridPromptsConfig,
    PromptsFileConfig,
    build_prompt_pool,
    pool_batch,
)
from param_decomp.tests.lm_configs import l18_mlp_raw


def _targeted_raw():
    raw = l18_mlp_raw()
    raw["target"]["attention_implementation"] = "xla"
    raw["decomposition"]["ci"]["attention"]["implementation"] = "xla"
    raw["pd"]["loss_metrics"] = [
        loss for loss in raw["pd"]["loss_metrics"] if loss["type"] != "FaithfulnessLoss"
    ]
    for field in (
        "faithfulness_warmup_lr",
        "faithfulness_warmup_steps",
        "faithfulness_warmup_weight_decay",
    ):
        del raw["pd"][field]
    raw["prompts"] = {
        "kind": "arithmetic_grid",
        "operation": "add",
        "a_range": [1, 100],
        "b_range": [1, 100],
    }
    raw["nontarget"] = {
        "batch_size": 256,
        "impmin_coeff": 1e-5,
        "recon": [
            {
                "type": "StochasticReconSubsetLoss",
                "coeff": 1.0,
                "routing": {"type": "uniform_k_subset"},
            }
        ],
    }
    return raw


class _StubTokenizer:
    """One id per character, no BOS — the Qwen-family number geometry exactly: digits
    tokenize per digit, so same-digit-count operands share one prompt length while any
    answer past 9 is multi-token."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [1 + ord(c) for c in text]


def test_arithmetic_pool_runs_at_natural_prompt_length():
    pool = build_prompt_pool(
        ArithmeticGridPromptsConfig(operation="add", a_range=(1, 4), b_range=(1, 4)),
        _StubTokenizer(),
    )
    # "<a>+<b>=" at one char per token — unpadded.
    assert pool.tokens.shape == (16, 4)
    assert pool.tokens.dtype == np.int32


def test_arithmetic_pool_admits_multi_token_answers():
    """The pool contract is prompt geometry only: `[1, 9] x [1, 9]` sums reach 18 —
    two ids under per-digit tokenization, which the eval probe refuses but a training
    pool, which never scores an answer position, must not."""
    pool = build_prompt_pool(
        ArithmeticGridPromptsConfig(operation="add", a_range=(1, 9), b_range=(1, 9)),
        _StubTokenizer(),
    )
    assert pool.tokens.shape == (81, 4)
    with pytest.raises(AssertionError, match="single answer token"):
        build_arithmetic_probe("add", (1, 9), (1, 9), _StubTokenizer())


def test_prompts_file_pool_requires_one_shared_length(tmp_path: Path):
    path = tmp_path / "prompts.txt"
    path.write_text("abcd\n\nefgh\n")
    pool = build_prompt_pool(PromptsFileConfig(path=path), _StubTokenizer())
    assert pool.tokens.shape == (2, 4)

    path.write_text("abcd\ntoolong\n")
    with pytest.raises(AssertionError, match="ONE shared length"):
        build_prompt_pool(PromptsFileConfig(path=path), _StubTokenizer())


def test_prompts_file_pool_under_per_digit_tokenization(tmp_path: Path):
    """Under a per-digit number tokenizer (the Qwen family) file prompts share a length
    exactly when their digit counts line up; mixed digit counts refuse rather than pad."""
    path = tmp_path / "prompts.txt"
    path.write_text("12+34=\n56+78=\n")
    pool = build_prompt_pool(PromptsFileConfig(path=path), _StubTokenizer())
    assert pool.tokens.shape == (2, 6)

    path.write_text("12+34=\n9+8=\n")
    with pytest.raises(AssertionError, match="ONE shared length"):
        build_prompt_pool(PromptsFileConfig(path=path), _StubTokenizer())


def test_pool_batch_is_pure_in_seed_and_step():
    pool = build_prompt_pool(
        ArithmeticGridPromptsConfig(operation="add", a_range=(1, 4), b_range=(1, 4)),
        _StubTokenizer(),
    )
    a = pool_batch(pool, seed=7, step=3, global_batch=32)
    b = pool_batch(pool, seed=7, step=3, global_batch=32)
    c = pool_batch(pool, seed=7, step=4, global_batch=32)
    assert a.shape == (32, 4)
    assert np.array_equal(a, b), "same (seed, step) must draw the same batch"
    assert not np.array_equal(a, c), "different steps must draw different batches"
    # Every drawn row is a pool row.
    pool_rows = {row.tobytes() for row in pool.tokens}
    assert all(row.tobytes() in pool_rows for row in a)


def test_targeted_variant_validates():
    raw = _targeted_raw()
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    # The objective builds (faithfulness-free target pass + the authored non-target pass)
    # against a stand-in site list — full resolution needs the HF snapshot.
    build_targeted_objective(
        cfg.pd.loss_metrics,
        cfg.nontarget,
        tuple(
            SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
            for name in (
                "layers.18.mlp.gate_proj",
                "layers.18.mlp.up_proj",
                "layers.18.mlp.down_proj",
            )
        ),
    )


def test_targeted_shape_refuses_plain_only_sections():
    raw = _targeted_raw()
    raw["resume_provenance"] = {"parent_run_dir": "/abs/run", "parent_step": 100}
    with pytest.raises(Exception, match="resume_provenance"):
        LMTargetedExperimentConfig.model_validate(raw)


def test_plain_shape_refuses_targeted_sections():
    raw = _targeted_raw()
    with pytest.raises(Exception, match="prompts|nontarget"):
        LMExperimentConfig.model_validate(raw)
