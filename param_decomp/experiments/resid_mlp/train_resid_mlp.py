"""Trains a residual MLP model on one-hot input vectors."""

import einops
import torch
import wandb
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm

from param_decomp.configs import ScheduleConfig
from param_decomp.experiments.resid_mlp.configs import ResidMLPModelConfig, ResidMLPTrainConfig
from param_decomp.experiments.resid_mlp.models import ResidMLP
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import (
    ResidMLPDataset,
)
from param_decomp.log import logger
from param_decomp.settings import DEFAULT_PROJECT_NAME
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import (
    compute_feature_importances,
    get_scheduled_value,
    set_seed,
)
from param_decomp.utils.run_utils import ExecutionStamp, save_file
from param_decomp.utils.wandb_utils import init_wandb


def loss_function(
    out: Float[Tensor, "batch n_features"] | Float[Tensor, "batch d_embed"],
    labels: Float[Tensor, "batch n_features"],
    feature_importances: Float[Tensor, "batch n_features"],
    model: ResidMLP,
    config: ResidMLPTrainConfig,
) -> Float[Tensor, "batch n_features"] | Float[Tensor, "batch d_embed"]:
    if config.loss_type == "readoff":
        loss = ((out - labels) ** 2) * feature_importances
    elif config.loss_type == "resid":
        assert torch.allclose(feature_importances, torch.ones_like(feature_importances)), (
            "feature_importances incompatible with loss_type resid"
        )
        resid_out: Float[Tensor, "batch d_embed"] = out
        resid_labels: Float[Tensor, "batch d_embed"] = einops.einsum(
            labels,
            model.W_E,
            "batch n_features, n_features d_embed -> batch d_embed",
        )
        loss = (resid_out - resid_labels) ** 2
    else:
        raise ValueError(f"Invalid loss_type: {config.loss_type}")
    return loss


def train(
    config: ResidMLPTrainConfig,
    model: ResidMLP,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[
            Float[Tensor, "batch n_features"],
            Float[Tensor, "batch n_features"],
        ]
    ],
    feature_importances: Float[Tensor, "batch n_features"],
    device: str,
    run_name: str,
) -> Float[Tensor, ""]:
    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = [f"resid_mlp{config.resid_mlp_model_config.n_layers}-train"]
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=execution_stamp.run_id,
            name=run_name,
            tags=tags,
        )

    # Save config
    config_path = out_dir / "resid_mlp_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    logger.info(f"Saved config to {config_path}")
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")

    # Save the coefficients used to generate the labels
    assert isinstance(dataloader.dataset, ResidMLPDataset)
    assert dataloader.dataset.label_coeffs is not None
    label_coeffs = dataloader.dataset.label_coeffs.tolist()
    label_coeffs_path = out_dir / "label_coeffs.json"
    save_file(label_coeffs, label_coeffs_path)
    logger.info(f"Saved label coefficients to {label_coeffs_path}")
    if config.wandb_project:
        wandb.save(str(label_coeffs_path), base_path=out_dir, policy="now")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.lr_schedule.start_val, weight_decay=0.01
    )

    pbar = tqdm(range(config.steps), total=config.steps)
    for step, (batch, labels) in zip(pbar, dataloader, strict=False):
        if step >= config.steps:
            break

        current_lr = get_scheduled_value(step, config.steps, config.lr_schedule)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch: Float[Tensor, "batch n_features"] = batch.to(device)
        labels: Float[Tensor, "batch n_features"] = labels.to(device)
        out = model(batch, return_residual=config.loss_type == "resid")
        loss: Float[Tensor, "batch n_features"] | Float[Tensor, "batch d_embed"] = loss_function(
            out, labels, feature_importances, model, config
        )
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if step % config.print_freq == 0:
            tqdm.write(f"step {step}: loss={loss.item():.2e}, lr={current_lr:.2e}")
            if config.wandb_project:
                wandb.log({"loss": loss.item(), "lr": current_lr}, step=step)

    model_path = out_dir / "resid_mlp.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    # Calculate final losses by averaging many batches
    final_losses = []
    for _ in range(config.n_batches_final_losses):
        batch, labels = next(iter(dataloader))
        batch = batch.to(device)
        labels = labels.to(device)
        out = model(batch, return_residual=config.loss_type == "resid")
        loss = loss_function(out, labels, feature_importances, model, config)
        loss = loss.mean()
        final_losses.append(loss)
    final_losses = torch.stack(final_losses).mean().cpu().detach()
    logger.info(f"Final losses: {final_losses.numpy()}")
    return final_losses


def run_train(config: ResidMLPTrainConfig, device: str) -> Float[Tensor, ""]:
    model_cfg = config.resid_mlp_model_config
    run_name = (
        f"resid_mlp_identity_{config.label_type}_"
        f"n-features{model_cfg.n_features}_d-resid{model_cfg.d_embed}_"
        f"d-mlp{model_cfg.d_mlp}_n-layers{model_cfg.n_layers}_seed{config.seed}"
        f"_p{config.feature_probability}_random_embedding_{config.fixed_random_embedding}_"
        f"identity_embedding_{config.fixed_identity_embedding}_bias_{model_cfg.in_bias}_"
        f"{model_cfg.out_bias}_loss{config.loss_type}"
    )

    model = ResidMLP(config=model_cfg).to(device)

    if config.fixed_random_embedding or config.fixed_identity_embedding:
        # Don't train the embedding matrices
        model.W_E.requires_grad = False
        model.W_U.requires_grad = False
        if config.fixed_random_embedding:
            # Init with randn values and make unit norm
            model.W_E.data[:, :] = torch.randn(
                model_cfg.n_features, model_cfg.d_embed, device=device
            )
            model.W_E.data /= model.W_E.data.norm(dim=-1, keepdim=True)
            # Set W_U to W_E^T
            model.W_U.data = model.W_E.data.T
            assert torch.allclose(model.W_U.data, model.W_E.data.T)
        elif config.fixed_identity_embedding:
            assert model_cfg.n_features == model_cfg.d_embed, (
                "n_features must equal d_embed for W_E=id"
            )
            # Make W_E the identity matrix
            model.W_E.data[:, :] = torch.eye(model_cfg.d_embed, device=device)

    label_coeffs = None
    if config.use_trivial_label_coeffs:
        label_coeffs = torch.ones(model_cfg.n_features, device=device)

    dataset = ResidMLPDataset(
        n_features=model_cfg.n_features,
        feature_probability=config.feature_probability,
        device=device,
        calc_labels=True,
        label_type=config.label_type,
        act_fn_name=model_cfg.act_fn_name,
        label_fn_seed=config.label_fn_seed,
        label_coeffs=label_coeffs,
        data_generation_type=config.data_generation_type,
        synced_inputs=config.synced_inputs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    feature_importances = compute_feature_importances(
        batch_size=config.batch_size,
        n_features=model_cfg.n_features,
        importance_val=config.importance_val,
        device=device,
    )

    final_losses = train(
        config=config,
        model=model,
        trainable_params=[p for p in model.parameters() if p.requires_grad],
        dataloader=dataloader,
        feature_importances=feature_importances,
        device=device,
        run_name=run_name,
    )
    return final_losses


if __name__ == "__main__":
    device = get_device()
    # 1 layer
    config = ResidMLPTrainConfig(
        wandb_project=DEFAULT_PROJECT_NAME,
        seed=0,
        resid_mlp_model_config=ResidMLPModelConfig(
            n_features=100,  # 1 layer
            d_embed=1000,
            d_mlp=50,  # 1 layer
            n_layers=1,  # 1 layer
            act_fn_name="relu",
            in_bias=False,
            out_bias=False,
        ),
        label_fn_seed=0,
        label_type="act_plus_resid",
        loss_type="readoff",
        use_trivial_label_coeffs=True,
        feature_probability=0.01,
        # synced_inputs=[[0, 1], [2, 3]], # synced inputs
        importance_val=1,
        data_generation_type="at_least_zero_active",
        batch_size=2048,
        steps=1000,  # 1 layer
        print_freq=100,
        lr_schedule=ScheduleConfig(start_val=3e-3, fn_type="cosine", final_val_frac=0.0),
        fixed_random_embedding=True,
        fixed_identity_embedding=False,
        n_batches_final_losses=10,
    )
    # # 2 layers
    # config = ResidMLPTrainConfig(
    #     wandb_project=DEFAULT_PROJECT_NAME,
    #     seed=0,
    #     resid_mlp_model_config=ResidMLPModelConfig(
    #         n_features=100, # 2 layers
    #         d_embed=1000,
    #         d_mlp=25, # 2 layers
    #         n_layers=2, # 2 layers
    #         act_fn_name="relu",
    #         in_bias=False,
    #         out_bias=False,
    #     ),
    #     label_fn_seed=0,
    #     label_type="act_plus_resid",
    #     loss_type="readoff",
    #     use_trivial_label_coeffs=True,
    #     feature_probability=0.01,
    #     # synced_inputs=[[0, 1], [2, 3]], # synced inputs
    #     importance_val=1,
    #     data_generation_type="at_least_zero_active",
    #     batch_size=2048,
    #     steps=1000, # 2 layers
    #     print_freq=100,
    #     lr_schedule=ScheduleConfig(start_val=3e-3, fn_type="cosine", final_val_frac=0.0),
    #     fixed_random_embedding=True,
    #     fixed_identity_embedding=False,
    #     n_batches_final_losses=10,
    # )
    # # 3 layers
    # config = ResidMLPTrainConfig(
    #     wandb_project=DEFAULT_PROJECT_NAME,
    #     seed=0,
    #     resid_mlp_model_config=ResidMLPModelConfig(
    #         n_features=102,  # 3 layers
    #         d_embed=1000,
    #         d_mlp=17,  # 3 layers
    #         n_layers=3,  # 3 layers
    #         act_fn_name="relu",
    #         in_bias=False,
    #         out_bias=False,
    #     ),
    #     label_fn_seed=0,
    #     label_type="act_plus_resid",
    #     loss_type="readoff",
    #     use_trivial_label_coeffs=True,
    #     feature_probability=0.01,
    #     # synced_inputs=[[0, 1], [2, 3]], # synced inputs
    #     importance_val=1,
    #     data_generation_type="at_least_zero_active",
    #     batch_size=2048,
    #     steps=10_000,  # 3 layers
    #     print_freq=100,
    #     lr_schedule=ScheduleConfig(start_val=3e-3, fn_type="cosine", final_val_frac=0.0),
    #     fixed_random_embedding=True,
    #     fixed_identity_embedding=False,
    #     n_batches_final_losses=10,
    # )

    set_seed(config.seed)

    run_train(config, device)
