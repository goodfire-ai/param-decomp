from pathlib import Path
from typing import cast

import torch
from torch import nn

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
from param_decomp.metrics.builtin.stochastic_recon_layerwise_loss import (
    StochasticReconLayerwiseLossConfig,
)
from param_decomp.metrics.builtin.stochastic_recon_loss import StochasticReconLossConfig
from param_decomp.models.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.general_utils import set_seed
from param_decomp_lab.experiments.tms.models import TMSModel, TMSModelConfig, TMSTrainConfig
from param_decomp_lab.experiments.tms.train_tms import get_model_and_dataloader, train


def test_tms_decomposition_happy_path(tmp_path: Path) -> None:
    """Test that PD works on a TMS model."""
    set_seed(0)
    device = "cpu"

    tms_model_config = TMSModelConfig(
        n_features=5,
        n_hidden=2,
        n_hidden_layers=1,
        tied_weights=True,
        init_bias_to_zero=False,
        device=device,
    )

    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        module_info=[
            ModulePatternInfoConfig(module_pattern="linear1", C=10),
            ModulePatternInfoConfig(module_pattern="linear2", C=10),
            ModulePatternInfoConfig(module_pattern="hidden_layers.0", C=10),
        ],
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="linear1", C=10),
        ],
        loss_metrics={
            "ImportanceMinimalityLoss": ImportanceMinimalityLossConfig(
                coeff=3e-3, pnorm=2.0, beta=0.5, eps=1e-12
            ),
            "StochasticReconLayerwiseLoss": StochasticReconLayerwiseLossConfig(coeff=1.0),
            "StochasticReconLoss": StochasticReconLossConfig(coeff=1.0),
            "FaithfulnessLoss": FaithfulnessLossConfig(coeff=1.0),
        },
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
        batch_size=4,
        steps=3,
        faithfulness_warmup_steps=2,
        faithfulness_warmup_lr=0.001,
        faithfulness_warmup_weight_decay=0.0,
        tied_weights=[("linear1", "linear2")],
    )

    target_model = TMSModel(config=tms_model_config).to(device)
    target_model.eval()

    if pd_config.identity_module_info is not None:
        insert_identity_operations_(
            target_model, identity_module_info=pd_config.identity_module_info
        )

    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=0.05,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
        synced_inputs=None,
    )

    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=pd_config.batch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=pd_config.batch_size, shuffle=False
    )

    sink = RunSink.local(
        tmp_path,
        train_log_freq=2,
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

    print("TMS PD optimization completed successfully")


def test_train_tms_happy_path():
    """Test training a TMS model from scratch."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=0,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=5,
        lr_schedule=ScheduleConfig(start_val=5e-3),
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    train(
        model,
        dataloader,
        importance=1.0,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    print("TMS training completed successfully")


def test_tms_train_fixed_identity():
    """Check that hidden layer is identity before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=2,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr_schedule=ScheduleConfig(start_val=5e-3),
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=True,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    eye = torch.eye(config.tms_model_config.n_hidden, device=device)

    assert model.hidden_layers is not None
    initial_hidden = cast(nn.Linear, model.hidden_layers[0]).weight.data.clone()
    assert torch.allclose(initial_hidden, eye), "Initial hidden layer is not identity"

    train(
        model,
        dataloader,
        importance=1.0,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    assert torch.allclose(cast(nn.Linear, model.hidden_layers[0]).weight.data, eye), (
        "Hidden layer changed"
    )


def test_tms_train_fixed_random():
    """Check that hidden layer is random before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_hidden_layers=2,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr_schedule=ScheduleConfig(start_val=5e-3),
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=True,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    assert model.hidden_layers is not None
    initial_hidden = cast(nn.Linear, model.hidden_layers[0]).weight.data.clone()

    train(
        model,
        dataloader,
        importance=1.0,
        lr_schedule=config.lr_schedule,
        steps=config.steps,
        print_freq=1000,
        log_wandb=False,
    )

    assert torch.allclose(cast(nn.Linear, model.hidden_layers[0]).weight.data, initial_hidden), (
        "Hidden layer changed"
    )
