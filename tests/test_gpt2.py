from pathlib import Path

import pytest
from transformers import GPT2LMHeadModel

from param_decomp.configs import (
    CI_L0Config,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    PDConfig,
    ScheduleConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
)
from param_decomp.data import DatasetConfig, create_data_loader, input_ids_collate_fn
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.models.batch_and_loss_fns import (
    make_run_batch,
    move_batch_to_device,
    recon_loss_kl,
)
from param_decomp.run_param_decomp import optimize
from param_decomp.utils.general_utils import set_seed


@pytest.mark.slow
def test_gpt_2_decomposition_happy_path(tmp_path: Path) -> None:
    """Test that PD works for GPT-2"""
    set_seed(0)
    device = "cpu"

    config = PDConfig(
        wandb_project=None,
        wandb_run_name=None,
        wandb_run_name_prefix="",
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
        loss_metric_configs=[
            ImportanceMinimalityLossConfig(
                coeff=1e-2,
                pnorm=0.9,
                beta=0.5,
                eps=1e-12,
            ),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
            StochasticReconLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=200),
        ],
        lr_schedule=ScheduleConfig(
            start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
        ),
        batch_size=4,
        steps=2,
        n_eval_steps=1,
        train_log_freq=50,
        eval_freq=500,
        eval_batch_size=1,
        slow_eval_freq=500,
        slow_eval_on_first_step=False,
        save_freq=None,
        ci_alive_threshold=0.1,
        eval_metric_configs=[
            CI_L0Config(groups=None),
        ],
    )

    model_name = "SimpleStories/test-SimpleStories-gpt2-1.25M"
    target_model = GPT2LMHeadModel.from_pretrained(model_name)
    target_model.eval()

    if config.identity_module_info is not None:
        insert_identity_operations_(target_model, identity_module_info=config.identity_module_info)

    train_data_config = DatasetConfig(
        name="SimpleStories/SimpleStories",
        hf_tokenizer_path=model_name,
        split="train[:100]",
        n_ctx=16,
        is_tokenized=False,
        streaming=False,
        column_name="story",
        seed=None,
    )

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.batch_size,
        buffer_size=1000,
        global_seed=config.seed,
        collate_fn=input_ids_collate_fn,
    )

    eval_data_config = DatasetConfig(
        name="SimpleStories/SimpleStories",
        hf_tokenizer_path=model_name,
        split="test[100:200]",
        n_ctx=16,
        is_tokenized=False,
        streaming=False,
        column_name="story",
        seed=None,
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.batch_size,
        buffer_size=1000,
        global_seed=config.seed + 1,
        collate_fn=input_ids_collate_fn,
    )

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=make_run_batch("logits"),
        reconstruction_loss=recon_loss_kl,
        to_device=move_batch_to_device,
        out_dir=tmp_path,
    )

    assert True, "Test completed successfully"
