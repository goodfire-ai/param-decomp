from pathlib import Path

from param_decomp.configs import (
    Config,
    GlobalCiConfig,
    LayerwiseCiConfig,
    LMTaskConfig,
    ModulePatternInfoConfig,
    OptimizerConfig,
    ScheduleConfig,
)
from param_decomp.experiments.lm.lm_decomposition import _load_target_model
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from param_decomp.settings import REPO_ROOT


def test_lm_can_load_random_target_from_inline_config() -> None:
    config = Config(
        wandb_project=None,
        wandb_run_name=None,
        wandb_run_name_prefix="",
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        module_info=[ModulePatternInfoConfig(module_pattern="h.*.mlp.c_fc", C=4)],
        identity_module_info=None,
        loss_metric_configs=[],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        steps=1,
        batch_size=1,
        train_log_freq=1,
        eval_freq=10,
        eval_batch_size=1,
        slow_eval_freq=10,
        n_eval_steps=1,
        save_freq=None,
        pretrained_model_class="param_decomp.pretrain.models.llama_simple_mlp.LlamaSimpleMLP",
        pretrained_model_path=None,
        pretrained_model_name=None,
        target_model_config={
            "model_type": "LlamaSimpleMLP",
            "block_size": 8,
            "vocab_size": 64,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "n_intermediate": 64,
            "rotary_dim": 8,
            "n_ctx": 8,
            "n_key_value_heads": 2,
            "flash_attention": False,
        },
        output_extract=0,
        tokenizer_name="EleutherAI/gpt-neox-20b",
        task_config=LMTaskConfig(
            task_name="lm",
            max_seq_len=8,
            dataset_name="dummy",
            column_name="input_ids",
            is_tokenized=True,
        ),
    )

    model = _load_target_model(config)

    assert isinstance(model, LlamaSimpleMLP)
    assert model.config.n_layer == 1
    assert model.config.n_embd == 16
    assert model.config.vocab_size == 64


def test_random_1b_lm_experiment_config_validates() -> None:
    config_path = Path("param_decomp/experiments/lm/pile_llama_simple_mlp-1B-random.yaml")
    config = Config.from_file(REPO_ROOT / config_path)

    assert config.target_model_config is not None
    assert isinstance(config.ci_config, GlobalCiConfig)
    model_cfg = config.target_model_config
    assert model_cfg["n_layer"] == 20
    assert model_cfg["n_embd"] == 2048
    assert model_cfg["n_intermediate"] == 8192
    assert config.include_loss_metrics_in_eval is False
    assert config.eval_metric_configs == []
    assert config.save_final_checkpoint is False
    assert config.task_config.max_seq_len == 2048
    assert config.task_config.dataset_name == "monology/pile-uncopyrighted-parquet"
    assert config.task_config.column_name == "text"
    assert config.task_config.eval_data_split == "train"
    assert config.task_config.is_tokenized is False

    d_model = model_cfg["n_embd"]
    n_layer = model_cfg["n_layer"]
    n_intermediate = model_cfg["n_intermediate"]
    n_head = model_cfg["n_head"]
    n_kv = model_cfg["n_key_value_heads"]
    head_dim = d_model // n_head

    attn_params_per_layer = 2 * d_model * d_model + 2 * d_model * (n_kv * head_dim)
    mlp_params_per_layer = 2 * d_model * n_intermediate
    norm_params_per_layer = 2 * d_model
    non_embedding_params = n_layer * (
        attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    ) + d_model

    assert 1_000_000_000 <= non_embedding_params <= 1_020_000_000

    transformer_cfg = config.ci_config.simple_transformer_ci_cfg
    assert transformer_cfg is not None
    assert transformer_cfg.attn_config.max_len == 2048
    assert model_cfg["block_size"] == 2048
    assert model_cfg["n_ctx"] == 2048
    ci_d_model = transformer_cfg.d_model
    ci_n_blocks = transformer_cfg.n_blocks
    ci_mlp_hidden_dim = transformer_cfg.mlp_hidden_dim[0]
    total_input_dim = n_layer * (d_model + n_intermediate + 4 * d_model)
    total_c = sum(info.C for info in config.module_info) * n_layer
    ci_input_head_params = total_input_dim * ci_d_model + ci_d_model
    ci_output_head_params = ci_d_model * total_c + total_c
    ci_block_params = (
        4 * ci_d_model * ci_d_model
        + 2 * ci_d_model * ci_mlp_hidden_dim
        + ci_mlp_hidden_dim
        + ci_d_model
    )
    ci_params = ci_input_head_params + ci_output_head_params + ci_n_blocks * ci_block_params

    assert 11_000_000_000 <= ci_params <= 11_100_000_000
