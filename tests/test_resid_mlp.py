from pathlib import Path

from param_decomp import PDConfig, RunSink, RuntimeConfig, optimize
from param_decomp.configs import (
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    OptimizerConfig,
    ScheduleConfig,
)
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.metrics.builtin.faithfulness_loss import FaithfulnessLossConfig
from param_decomp.metrics.builtin.importance_minimality_loss import ImportanceMinimalityLossConfig
from param_decomp.metrics.builtin.stochastic_recon_loss import StochasticReconLossConfig
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
from param_decomp.utils.general_utils import set_seed
from param_decomp_lab.experiments.resid_mlp.models import ResidMLP, ResidMLPModelConfig
from param_decomp_lab.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset


def test_resid_mlp_decomposition_happy_path(tmp_path: Path) -> None:
    """Test that PD works on a 2-layer ResidMLP model."""
    set_seed(0)
    device = "cpu"

    resid_mlp_model_config = ResidMLPModelConfig(
        n_features=5,
        d_embed=4,
        d_mlp=6,
        n_layers=2,
        act_fn_name="relu",
        in_bias=True,
        out_bias=True,
    )

    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        loss_metrics={
            "ImportanceMinimalityLoss": ImportanceMinimalityLossConfig(
                coeff=3e-3, pnorm=0.9, beta=0.5, eps=1e-12
            ),
            "StochasticReconLoss": StochasticReconLossConfig(coeff=1.0),
            "FaithfulnessLoss": FaithfulnessLossConfig(coeff=1.0),
        },
        module_info=[
            ModulePatternInfoConfig(module_pattern="layers.*.mlp_in", C=10),
            ModulePatternInfoConfig(module_pattern="layers.*.mlp_out", C=10),
        ],
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="layers.*.mlp_in", C=10),
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
        steps=3,
    )

    target_model = ResidMLP(config=resid_mlp_model_config).to(device)
    target_model.requires_grad_(False)

    if pd_config.identity_module_info is not None:
        insert_identity_operations_(
            target_model, identity_module_info=pd_config.identity_module_info
        )

    dataset = ResidMLPDataset(
        n_features=resid_mlp_model_config.n_features,
        feature_probability=0.01,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type="at_least_zero_active",
        synced_inputs=None,
    )

    eval_batch_size = 4
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=pd_config.batch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    sink = RunSink.local(
        tmp_path,
        train_log_freq=50,
        eval_freq=10,
        slow_eval_freq=10,
        n_eval_steps=1,
        save_freq=None,
    )

    optimize(
        target_model=target_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(),
        sink=sink,
        eval_metrics=[],
        device=device,
    )
