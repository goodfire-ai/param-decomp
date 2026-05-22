from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers import GPT2LMHeadModel

from param_decomp import PDConfig, RuntimeConfig, optimize
from param_decomp.configs import (
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    OptimizerConfig,
    ScheduleConfig,
)
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.metrics.faithfulness_loss import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality_loss import ImportanceMinimalityLossConfig
from param_decomp.metrics.stochastic_recon_layerwise_loss import (
    StochasticReconLayerwiseLossConfig,
)
from param_decomp.metrics.stochastic_recon_loss import StochasticReconLossConfig
from param_decomp_lab.eval_metrics.ci_l0 import CI_L0, CI_L0Config
from param_decomp_lab.experiments.lm.data import LMDataConfig, create_lm_data_loader
from param_decomp_lab.models.batch_and_loss_fns import make_run_batch, recon_loss_kl
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.utils.seed import set_seed


@pytest.mark.slow
def test_gpt_2_decomposition_happy_path(tmp_path: Path) -> None:
    """Test that PD works for GPT-2"""
    set_seed(0)
    device = "cpu"

    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[128]),
        module_info=[
            ModulePatternInfoConfig(module_pattern="transformer.h.2.attn.c_attn", C=10),
            ModulePatternInfoConfig(module_pattern="transformer.h.3.mlp.c_fc", C=10),
        ],
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="transformer.h.1.attn.c_attn", C=10),
        ],
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=1e-2, pnorm=0.9, beta=0.5, eps=1e-12),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
            StochasticReconLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=200),
        ],
        components_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
            ),
        ),
        ci_fn_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
            ),
        ),
        batch_size=4,
        steps=2,
    )

    model_name = "SimpleStories/test-SimpleStories-gpt2-1.25M"
    target_model = GPT2LMHeadModel.from_pretrained(model_name)
    target_model.eval()

    if pd_config.identity_module_info is not None:
        insert_identity_operations_(
            target_model, identity_module_info=pd_config.identity_module_info
        )

    data_config = LMDataConfig(
        dataset_name="SimpleStories/SimpleStories",
        tokenizer_name=model_name,
        max_seq_len=16,
        train_split="train[:100]",
        eval_split="test[100:200]",
        is_tokenized=False,
        streaming=False,
        column_name="story",
    )

    def collate_input_ids(batch: list[dict[str, Tensor]]) -> Tensor:
        return torch.stack([item["input_ids"] for item in batch])

    train_loader, _tokenizer = create_lm_data_loader(
        data_config,
        split=data_config.train_split,
        batch_size=pd_config.batch_size,
        seed=pd_config.seed,
        collate_fn=collate_input_ids,
    )
    eval_loader, _ = create_lm_data_loader(
        data_config,
        split=data_config.eval_split,
        batch_size=1,
        seed=pd_config.seed + 1,
        collate_fn=collate_input_ids,
    )

    sink = RunSink.local(
        tmp_path,
        train_log_freq=50,
        eval_freq=500,
        slow_eval_freq=500,
        slow_eval_on_first_step=False,
        n_eval_steps=1,
        save_freq=None,
    )

    optimize(
        target_model=target_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=make_run_batch("logits"),
        reconstruction_loss=recon_loss_kl,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(),
        sink=sink,
        eval_metrics=[CI_L0(CI_L0Config(ci_alive_threshold=0.1, groups=None))],
        device=device,
    )
