from pathlib import Path

import pytest
from transformers import PreTrainedModel

from spd.configs import (
    CI_L0Config,
    Config,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    LMTaskConfig,
    ModulePatternInfoConfig,
    ScheduleConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
)
from spd.data import DatasetConfig, create_data_loader
from spd.identity_insertion import insert_identity_operations_
from spd.run_spd import optimize
from spd.utils.general_utils import resolve_class, set_seed


@pytest.mark.slow
def test_gpt_2_decomposition_happy_path(tmp_path: Path) -> None:
    """Test that SPD decomposition works on for GPT-2"""
    set_seed(0)
    device = "cpu"

    # Create config similar to the gpt-2 config in gpt2_config.yaml
    config = Config(
        # WandB
        wandb_project=None,  # Disable wandb for testing
        wandb_run_name=None,
        wandb_run_name_prefix="",
        # General
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
        output_loss_type="kl",
        # Training
        lr_schedule=ScheduleConfig(
            start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
        ),
        batch_size=4,
        steps=2,
        n_eval_steps=1,
        # Logging & Saving
        train_log_freq=50,  # Print at step 0, 50, and 100
        eval_freq=500,
        eval_batch_size=1,
        slow_eval_freq=500,
        slow_eval_on_first_step=False,
        save_freq=None,
        ci_alive_threshold=0.1,
        eval_metric_configs=[
            CI_L0Config(groups=None),
        ],
        # Pretrained model info
        pretrained_model_class="transformers.GPT2LMHeadModel",
        pretrained_model_path=None,
        pretrained_model_name="SimpleStories/test-SimpleStories-gpt2-1.25M",
        pretrained_model_output_attr="logits",
        tokenizer_name="SimpleStories/test-SimpleStories-gpt2-1.25M",
        # Task Specific
        task_config=LMTaskConfig(
            task_name="lm",
            max_seq_len=16,
            buffer_size=1000,
            dataset_name="SimpleStories/SimpleStories",
            column_name="story",
            train_data_split="train[:100]",
            eval_data_split="test[100:200]",
        ),
    )

    assert isinstance(config.task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Create a GPT-2 model
    hf_model_class = resolve_class(config.pretrained_model_class)
    assert issubclass(hf_model_class, PreTrainedModel), (
        f"Model class {hf_model_class} should be a subclass of PreTrainedModel which "
        "defines a `from_pretrained` method"
    )
    assert config.pretrained_model_name is not None
    target_model = hf_model_class.from_pretrained(config.pretrained_model_name)
    target_model.eval()

    if config.identity_module_info is not None:
        insert_identity_operations_(target_model, identity_module_info=config.identity_module_info)

    train_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name,
        split=config.task_config.train_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        seed=None,
    )

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed,
    )

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.pretrained_model_name,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        seed=None,
    )
    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=config.batch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
    )

    # Run optimize function
    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=tmp_path,
    )

    # Basic assertion to ensure the test ran
    assert True, "Test completed successfully"
