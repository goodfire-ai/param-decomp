"""The targeted (tPD, SPEC §11) LM seat: prompt-pool construction (local stub tokenizer —
no hub dependency), the pure per-step pool batch, the config shape's refusals, and the
shipped seat config."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from param_decomp.core.objective import build_targeted_objective
from param_decomp.experiments.lm.config import LMExperimentConfig, LMTargetedExperimentConfig
from param_decomp.experiments.lm.targeted_data import (
    ArithmeticGridPromptsConfig,
    PromptsFileConfig,
    build_prompt_pool,
    pool_batch,
)

_CONFIGS_DIR = Path(__file__).parents[3] / "experiments" / "lm" / "configs"


class _StubTokenizer:
    """One id per character — every `<a><op><b>=` prompt over 1-digit operands is 4 ids,
    every 1-digit answer 1 id, so the probe's single-length asserts hold without any HF
    artifact."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [1 + ord(c) for c in text]


def test_arithmetic_pool_runs_at_natural_prompt_length():
    pool = build_prompt_pool(
        ArithmeticGridPromptsConfig(operation="add", a_range=(1, 4), b_range=(1, 4)),
        _StubTokenizer(),
    )
    # "<a>+<b>=" at one char per token — unpadded (T8).
    assert pool.tokens.shape == (16, 4)
    assert pool.tokens.dtype == np.int32


def test_prompts_file_pool_requires_one_shared_length(tmp_path: Path):
    path = tmp_path / "prompts.txt"
    path.write_text("abcd\n\nefgh\n")
    pool = build_prompt_pool(PromptsFileConfig(path=path), _StubTokenizer())
    assert pool.tokens.shape == (2, 4)

    path.write_text("abcd\ntoolong\n")
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
    assert np.array_equal(a, b), "same (seed, step) must draw the same batch (S18)"
    assert not np.array_equal(a, c), "different steps must draw different batches"
    # Every drawn row is a pool row.
    pool_rows = {row.tobytes() for row in pool.tokens}
    assert all(row.tobytes() in pool_rows for row in a)


def test_shipped_targeted_config_validates():
    raw = yaml.safe_load((_CONFIGS_DIR / "llama8b_l18_arith_targeted.yaml").read_text())
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    # The objective builds (faithfulness-free target pass + the authored non-target pass)
    # against a stand-in site list — full resolution needs the HF snapshot.
    build_targeted_objective(
        cfg.pd.loss_metrics,
        cfg.nontarget,
        ("layers.18.mlp.gate_proj", "layers.18.mlp.up_proj", "layers.18.mlp.down_proj"),
    )


def test_targeted_shape_refuses_plain_only_sections():
    raw = yaml.safe_load((_CONFIGS_DIR / "llama8b_l18_arith_targeted.yaml").read_text())
    raw["resume_provenance"] = {"parent_run_dir": "/abs/run", "parent_step": 100}
    with pytest.raises(Exception, match="resume_provenance"):
        LMTargetedExperimentConfig.model_validate(raw)


def test_plain_shape_refuses_targeted_sections():
    raw = yaml.safe_load((_CONFIGS_DIR / "llama8b_l18_arith_targeted.yaml").read_text())
    with pytest.raises(Exception, match="prompts|nontarget"):
        LMExperimentConfig.model_validate(raw)
