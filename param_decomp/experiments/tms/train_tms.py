"""TMS model, adapted from
https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb
"""

from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib import collections as mc
from torch import Tensor, nn
from tqdm import tqdm, trange

from param_decomp.configs import ScheduleConfig
from param_decomp.experiments.tms.configs import TMSModelConfig, TMSTrainConfig
from param_decomp.experiments.tms.models import TMSModel
from param_decomp.log import logger
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import get_scheduled_value, set_seed
from param_decomp.utils.run_utils import ExecutionStamp, save_file


def train(
    model: TMSModel,
    dataloader: DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    log_wandb: bool,
    importance: float,
    steps: int,
    print_freq: int,
    lr_schedule: ScheduleConfig,
) -> None:
    hooks = []

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr_schedule.start_val)

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = get_scheduled_value(step, steps, lr_schedule)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch, labels = next(data_iter)
            out = model(batch)
            error = importance * (labels.abs() - out) ** 2
            loss = error.mean()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(
                    model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item()}")
                t.set_postfix(
                    loss=loss.item(),
                    lr=step_lr,
                )
                if log_wandb:
                    wandb.log({"loss": loss.item(), "lr": step_lr}, step=step)


def plot_intro_diagram(model: TMSModel, filepath: Path) -> None:
    """2D polygon plot of the TMS model.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    WA = model.linear1.weight.T.detach()
    color = plt.cm.viridis(np.array([0.0]))  # pyright: ignore[reportAttributeAccessIssue]
    plt.rcParams["figure.dpi"] = 200
    _, ax = plt.subplots(1, 1, figsize=(2, 2))

    W = WA.cpu().detach().numpy()
    ax.scatter(W[:, 0], W[:, 1], c=color)
    ax.set_aspect("equal")
    ax.add_collection(
        mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=[color])  # pyright: ignore[reportArgumentType]
    )

    z = 1.5
    ax.set_facecolor("#FCFBF8")
    ax.set_xlim((-z, z))
    ax.set_ylim((-z, z))
    ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position("center")
    plt.savefig(filepath)


def plot_cosine_similarity_distribution(
    model: TMSModel,
    filepath: Path,
) -> None:
    """Create scatter plot of cosine similarities between feature vectors.

    Args:
        model: The trained TMS model
        filepath: Where to save the plot
    """
    # Calculate cosine similarities
    rows = model.linear1.weight.T.detach()
    rows /= rows.norm(dim=-1, keepdim=True)
    cosine_sims = einops.einsum(rows, rows, "f1 h, f2 h -> f1 f2")
    mask = ~torch.eye(rows.shape[0], device=rows.device, dtype=torch.bool)
    masked_sims = cosine_sims[mask]

    _, ax = plt.subplots(1, 1, figsize=(4, 4))

    sims = masked_sims.cpu().numpy()
    ax.scatter(sims, np.zeros_like(sims), alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cosine Similarity")
    ax.set_yticks([])  # Hide y-axis ticks

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def get_model_and_dataloader(
    config: TMSTrainConfig, device: str
) -> tuple[TMSModel, DatasetGeneratedDataLoader[tuple[Tensor, Tensor]]]:
    model = TMSModel(config=config.tms_model_config)
    model.to(device)
    if (
        config.fixed_identity_hidden_layers or config.fixed_random_hidden_layers
    ) and model.hidden_layers is not None:
        for i in range(model.config.n_hidden_layers):
            layer = model.hidden_layers[i]
            assert isinstance(layer, nn.Linear)
            if config.fixed_identity_hidden_layers:
                layer.weight.data[:, :] = torch.eye(model.config.n_hidden, device=device)
            elif config.fixed_random_hidden_layers:
                layer.weight.data[:, :] = torch.randn_like(layer.weight)
            layer.weight.requires_grad = False

    dataset = SparseFeatureDataset(
        n_features=config.tms_model_config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        data_generation_type=config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=config.synced_inputs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)
    return model, dataloader


def run_train(config: TMSTrainConfig, device: str) -> None:
    model, dataloader = get_model_and_dataloader(config, device)

    model_cfg = config.tms_model_config
    run_name = (
        f"tms_n-features{model_cfg.n_features}_n-hidden{model_cfg.n_hidden}_"
        f"n-hidden-layers{model_cfg.n_hidden_layers}_"
        f"feat_prob{config.feature_probability}_seed{config.seed}"
    )
    if config.fixed_identity_hidden_layers:
        run_name += "_fixed-identity"
    elif config.fixed_random_hidden_layers:
        run_name += "_fixed-random"

    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = [f"tms_{model_cfg.n_features}-{model_cfg.n_hidden}"]
        if model_cfg.n_hidden_layers > 0:
            tags[0] += "-id"
        wandb.init(
            id=execution_stamp.run_id,
            project=config.wandb_project,
            name=run_name,
            tags=tags,
        )

    # Save config
    config_path = out_dir / "tms_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    train(
        model,
        dataloader=dataloader,
        log_wandb=config.wandb_project is not None,
        steps=config.steps,
        importance=1.0,
        print_freq=100,
        lr_schedule=config.lr_schedule,
    )

    model_path = out_dir / "tms.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    # Analysis code from play.py
    input_size = config.tms_model_config.n_features
    test_value = 0.75
    output_values = []

    logger.info("Testing representation of each input feature...")
    logger.info(f"Input size: {input_size}, Test value: {test_value}")

    for i in range(input_size):
        # Create batch with test_value at position i, zeros elsewhere
        batch = torch.zeros(1, input_size, device=device)
        batch[0, i] = test_value

        # Run the model
        with torch.no_grad():
            out = model(batch)

        # Record the output value at the same index
        output_value = out[0, i].item()
        output_values.append(output_value)

        logger.info(f"Input index {i}: output value = {output_value:.4f}")

    # Convert to numpy for plotting
    output_values = np.array(output_values)

    # Create barplot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(input_size), output_values, alpha=0.7)

    # Color bars based on how well they preserve the input
    colors = [
        "green"
        if abs(val - test_value) < 0.1
        else "orange"
        if abs(val - test_value) < 0.3
        else "red"
        for val in output_values
    ]
    for bar, color in zip(bars, colors, strict=False):
        bar.set_color(color)

    plt.xlabel("Input Feature Index")
    plt.ylabel("Output Value at Same Index")
    plt.title(
        f"Feature Representation Quality\n(Input value: {test_value}, Green: good preservation, Orange: moderate, Red: poor)"
    )
    plt.grid(True, alpha=0.3)

    # Add horizontal line at test value for reference
    plt.axhline(
        y=test_value, color="black", linestyle="--", alpha=0.8, label=f"Target value ({test_value})"
    )
    plt.legend()

    # Add statistics
    mean_output = np.mean(output_values)
    std_output = np.std(output_values)
    min_output = np.min(output_values)
    max_output = np.max(output_values)

    plt.text(
        0.02,
        0.98,
        f"Stats:\nMean: {mean_output:.3f}\nStd: {std_output:.3f}\nMin: {min_output:.3f}\nMax: {max_output:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_dir / "feature_representation_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Summary statistics
    logger.values(
        msg="=== SUMMARY ===",
        data={
            "Mean output value": f"{mean_output:.4f}",
            "Standard deviation": f"{std_output:.4f}",
            "Min output value": f"{min_output:.4f}",
            "Max output value": f"{max_output:.4f}",
            "Target input value": f"{test_value:.4f}",
        },
    )

    # Count how many features are well-preserved
    well_preserved = np.sum(np.abs(output_values - test_value) < 0.1)
    moderately_preserved = np.sum(
        (np.abs(output_values - test_value) >= 0.1) & (np.abs(output_values - test_value) < 0.3)
    )
    poorly_preserved = np.sum(np.abs(output_values - test_value) >= 0.3)

    logger.values(
        msg="Feature preservation quality",
        data={
            f"Well preserved (|output - {test_value}| < 0.1)": f"{well_preserved}/{input_size} ({100 * well_preserved / input_size:.1f}%)",
            f"Moderately preserved (0.1 ≤ |output - {test_value}| < 0.3)": f"{moderately_preserved}/{input_size} ({100 * moderately_preserved / input_size:.1f}%)",
            f"Poorly preserved (|output - {test_value}| ≥ 0.3)": f"{poorly_preserved}/{input_size} ({100 * poorly_preserved / input_size:.1f}%)",
        },
    )

    # Show which features are poorly preserved
    if poorly_preserved > 0:
        poor_indices = np.where(np.abs(output_values - test_value) >= 0.3)[0]
        logger.info(f"Poorly preserved feature indices: {poor_indices.tolist()}")
        logger.values(
            msg="Poorly preserved feature output values",
            data={
                f"Index {idx}": f"{output_values[idx]:.4f} (diff: {output_values[idx] - test_value:.4f})"
                for idx in poor_indices
            },
        )

    if model_cfg.n_hidden == 2:
        fname_polygon: Path = out_dir / "polygon.png"
        plot_intro_diagram(model, filepath=fname_polygon)
        logger.info(f"Saved diagram to {fname_polygon}")

    fname_cos_sim: Path = out_dir / "cosine_similarity_distribution.png"
    plot_cosine_similarity_distribution(model, filepath=fname_cos_sim)
    logger.info(f"Saved cosine similarity distribution to {fname_cos_sim}")
    logger.info(f"1/sqrt(n_hidden): {1 / np.sqrt(model_cfg.n_hidden)}")


if __name__ == "__main__":
    device = get_device()
    # NOTE: Training TMS is very finnicky, you may need to adjust hyperparams to get it working
    # TMS 5-2
    config = TMSTrainConfig(
        wandb_project="param-decomp",
        tms_model_config=TMSModelConfig(
            n_features=5,
            n_hidden=2,
            n_hidden_layers=0,
            tied_weights=True,
            device=device,
            init_bias_to_zero=False,
        ),
        feature_probability=0.05,
        batch_size=1024,
        steps=10000,
        seed=0,
        lr_schedule=ScheduleConfig(start_val=5e-3, fn_type="constant"),
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )
    # # TMS 5-2 w/ identity
    # config = TMSTrainConfig(
    #     wandb_project="param-decomp",
    #     tms_model_config=TMSModelConfig(
    #         n_features=5,
    #         n_hidden=2,
    #         n_hidden_layers=1,
    #         tied_weights=True,
    #         device=device,
    #         init_bias_to_zero=False,
    #     ),
    #     feature_probability=0.05,
    #     batch_size=1024,
    #     steps=10000,
    #     seed=0,
    #     lr_schedule=ScheduleConfig(start_val=5e-3, fn_type="constant"),
    #     data_generation_type="at_least_zero_active",
    #     fixed_identity_hidden_layers=True,
    #     fixed_random_hidden_layers=False,
    # )
    # # TMS 40-10
    # config = TMSTrainConfig(
    #     wandb_project="param-decomp",
    #     tms_model_config=TMSModelConfig(
    #         n_features=40,
    #         n_hidden=10,
    #         n_hidden_layers=0,
    #         tied_weights=True,
    #         device=device,
    #         init_bias_to_zero=True,
    #     ),
    #     feature_probability=0.05,
    #     # feature_probability=0.02, # synced inputs
    #     batch_size=8192,
    #     steps=10000,
    #     seed=0,
    #     lr_schedule=ScheduleConfig(start_val=5e-3, fn_type="constant"),
    #     data_generation_type="at_least_zero_active",
    #     fixed_identity_hidden_layers=False,
    #     fixed_random_hidden_layers=False,
    #     # synced_inputs=[[5, 6], [0, 2, 3]],
    # )
    # # TMS 40-10 w/ identity
    # config = TMSTrainConfig(
    #     wandb_project="param-decomp",
    #     tms_model_config=TMSModelConfig(
    #         n_features=40,
    #         n_hidden=10,
    #         n_hidden_layers=1,
    #         tied_weights=True,
    #         device=device,
    #         init_bias_to_zero=True,
    #     ),
    #     feature_probability=0.05,
    #     # feature_probability=0.02, # synced inputs
    #     batch_size=8192,
    #     steps=10000,
    #     seed=0,
    #     lr_schedule=ScheduleConfig(start_val=5e-3, fn_type="constant"),
    #     data_generation_type="at_least_zero_active",
    #     fixed_identity_hidden_layers=True,
    #     fixed_random_hidden_layers=False,
    #     # synced_inputs=[[5, 6], [0, 2, 3]],
    # )

    set_seed(config.seed)

    run_train(config, device)
