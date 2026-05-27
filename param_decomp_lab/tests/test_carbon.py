"""Smoke test for the Carbon experiment.

Exercises `optimize(...)` on a tiny in-process `LlamaForCausalLM` configured to mirror
Carbon-500M's architecture (Llama decoder, GQA, SwiGLU, RMSNorm, tied word embeddings)
without downloading the real 500M-param weights. The synthetic random-token loader from
`carbon.run` feeds it.
"""

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM
from transformers.models.llama import LlamaConfig

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.stochastic_recon_layerwise import (
    StochasticReconLayerwiseLossConfig,
)
from param_decomp.optimize import EvalLoop, optimize
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.eval_metrics.ci_l0 import CI_L0, CI_L0Config
from param_decomp_lab.experiments.carbon.run import (
    CarbonDataConfig,
    CarbonTargetConfig,
    build_carbon_loader,
    make_run_batch,
)
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


def _tiny_carbon_like_model() -> LlamaForCausalLM:
    """Tiny LlamaForCausalLM with Carbon-500M-shaped architecture: GQA, SwiGLU, RoPE,
    tied word embeddings. Dimensions are scaled to keep the test under a second on CPU.
    """
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # mirrors Carbon's GQA 16/8 ratio
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=32,
        rope_theta=500000.0,
        rms_norm_eps=1.0e-6,
        tie_word_embeddings=True,  # mirrors Carbon
        hidden_act="silu",
    )
    return LlamaForCausalLM(cfg)


def test_carbon_decomposition_happy_path(tmp_path: Path) -> None:
    set_seed(0)
    device = "cpu"

    target_model = _tiny_carbon_like_model()
    target_model.eval()

    # Single decomposition target on a mid-stack down_proj — one matrix is enough to
    # exercise every code path Carbon-500M would hit at scale.
    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[16]),
        decomposition_targets=[
            DecompositionTargetConfig(module_pattern="model.layers.1.mlp.down_proj", C=8),
        ],
        identity_decomposition_targets=None,
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=1e-2, pnorm=0.9, beta=0.5, eps=1e-12),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=10.0),
        ],
        components_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
            ),
        ),
        ci_fn_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
            ),
        ),
        batch_size=2,
        steps=2,
    )

    target_cfg = CarbonTargetConfig(
        # build_target isn't called here — we hand-build the tiny model above so the
        # test doesn't download Carbon-500M. The CarbonTargetConfig is still validated
        # so we exercise its schema.
        model_class="transformers.LlamaForCausalLM",
        model_name="HuggingFaceBio/Carbon-500M",
        trust_remote_code=True,
        dtype="float32",  # tiny test, no need for bf16
        output_extract="logits",
    )
    data_cfg = CarbonDataConfig(
        kind="synthetic",
        vocab_size=128,  # match the tiny model's vocab, not Carbon's full 155776
        seq_len=16,
        n_train=64,
        n_eval=8,
    )

    train_loader = build_carbon_loader(
        target_cfg,
        data_cfg,
        split="train",
        device=device,
        batch_size=pd_config.batch_size,
        seed=pd_config.seed,
    )
    eval_loader = build_carbon_loader(
        target_cfg,
        data_cfg,
        split="eval",
        device=device,
        batch_size=2,
        seed=pd_config.seed,
    )

    sink = RunSink.local(tmp_path)
    cadence = Cadence(train_log_every=1, save_every=None)
    eval_loop = EvalLoop(
        loader=eval_loader,
        metrics=[CI_L0(CI_L0Config(ci_alive_threshold=0.1, groups=None))],
        n_steps=1,
        every=500,
        slow_every=500,
        slow_on_first_step=False,
    )

    optimize(
        target_model=target_model,
        train_loader=train_loader,
        run_batch=make_run_batch(target_cfg),
        reconstruction_loss=recon_loss_kl,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(device=device),
        sink=sink,
        cadence=cadence,
        eval_loop=eval_loop,
    )

    # A checkpoint should have been written for the final step.
    assert any(tmp_path.glob("model_*.pth")), "expected a final checkpoint"


@pytest.mark.slow
def test_carbon_loader_yields_expected_shapes() -> None:
    """Sanity check the synthetic loader by itself."""
    target_cfg = CarbonTargetConfig()
    data_cfg = CarbonDataConfig(seq_len=24, n_train=8, vocab_size=4096)
    loader = build_carbon_loader(
        target_cfg,
        data_cfg,
        split="train",
        device="cpu",
        batch_size=2,
        seed=0,
    )
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (2, 24)
    assert batch.dtype == torch.int64
    assert int(batch.max()) < data_cfg.vocab_size
