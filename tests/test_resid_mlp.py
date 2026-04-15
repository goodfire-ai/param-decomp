from pathlib import Path

from spd.configs import (
    Config,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    ResidMLPTaskConfig,
    ScheduleConfig,
    StochasticReconLossConfig,
)
from spd.experiments.resid_mlp.configs import ResidMLPModelConfig
from spd.experiments.resid_mlp.models import ResidMLP
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.identity_insertion import insert_identity_operations_
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import set_seed


def test_resid_mlp_decomposition_happy_path(tmp_path: Path) -> None:
    """Test that SPD decomposition works on a 2-layer ResidMLP model."""
    set_seed(0)
    device = "cpu"

    # Create a 2-layer ResidMLP config
    resid_mlp_model_config = ResidMLPModelConfig(
        n_features=5,
        d_embed=4,
        d_mlp=6,
        n_layers=2,
        act_fn_name="relu",
        in_bias=True,
        out_bias=True,
    )

    # Create config similar to the 2-layer config in resid_mlp2_config.yaml
    config = Config(
        # WandB
        wandb_project=None,  # Disable wandb for testing
        wandb_run_name=None,
        wandb_run_name_prefix="",
        # General
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        loss_metric_configs=[
            ImportanceMinimalityLossConfig(
                coeff=3e-3,
                pnorm=0.9,
                beta=0.5,
                eps=1e-12,
            ),
            StochasticReconLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=1.0),
        ],
        module_info=[
            ModulePatternInfoConfig(module_pattern="layers.*.mlp_in", C=10),
            ModulePatternInfoConfig(module_pattern="layers.*.mlp_out", C=10),
        ],
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="layers.*.mlp_in", C=10),
        ],
        output_loss_type="mse",
        # Training
        lr_schedule=ScheduleConfig(
            start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
        ),
        batch_size=4,
        steps=3,  # Run more steps to see improvement
        n_eval_steps=1,
        eval_freq=10,
        eval_batch_size=4,
        slow_eval_freq=10,
        slow_eval_on_first_step=True,
        # Logging & Saving
        train_log_freq=50,  # Print at step 0, 50, and 100
        save_freq=None,
        ci_alive_threshold=0.1,
        # Pretrained model info
        pretrained_model_class="spd.experiments.resid_mlp.models.ResidMLP",
        pretrained_model_path=None,
        pretrained_model_name=None,
        pretrained_model_output_attr=None,
        tokenizer_name=None,
        # Task Specific
        task_config=ResidMLPTaskConfig(
            task_name="resid_mlp",
            feature_probability=0.01,
            data_generation_type="at_least_zero_active",
        ),
    )

    # Create a pretrained model

    target_model = ResidMLP(config=resid_mlp_model_config).to(device)
    target_model.requires_grad_(False)

    if config.identity_module_info is not None:
        insert_identity_operations_(target_model, identity_module_info=config.identity_module_info)

    assert isinstance(config.task_config, ResidMLPTaskConfig)
    # Create dataset
    dataset = ResidMLPDataset(
        n_features=resid_mlp_model_config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,  # Our labels will be the output of the target model
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=config.task_config.data_generation_type,
        synced_inputs=None,
    )

    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
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
