"""Dataset loading utilities for clustering runs.

Each clustering run loads its own dataset, seeded by the run index.
"""

from typing import Any

from torch.utils.data import DataLoader

from param_decomp.configs import LMTaskConfig, ResidMLPTaskConfig
from param_decomp.data import DatasetConfig, create_data_loader
from param_decomp.experiments.resid_mlp.models import ResidMLP
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.param_decomp_types import TaskName


def create_clustering_dataloader(
    model_path: str,
    task_name: TaskName,
    batch_size: int,
    seed: int,
) -> DataLoader[Any]:
    """Create a dataloader for clustering.

    Args:
        model_path: Path to decomposed model
        task_name: Task type
        batch_size: Batch size
        seed: Random seed for dataset

    Returns:
        DataLoader yielding batches
    """
    match task_name:
        case "lm":
            return _create_lm_dataloader(
                model_path=model_path,
                batch_size=batch_size,
                seed=seed,
            )
        case "resid_mlp":
            return _create_resid_mlp_dataloader(
                model_path=model_path,
                batch_size=batch_size,
                seed=seed,
            )
        case _:
            raise ValueError(f"Unsupported task: {task_name}")


def _create_lm_dataloader(model_path: str, batch_size: int, seed: int) -> DataLoader[Any]:
    """Create a dataloader for language model task."""
    pd_run = ParamDecompRunInfo.from_path(model_path)
    cfg = pd_run.config

    assert isinstance(cfg.task_config, LMTaskConfig), (
        f"Expected task_config to be of type LMTaskConfig, but got {type(cfg.task_config) = }"
    )

    dataset_config = DatasetConfig(
        name=cfg.task_config.dataset_name,
        hf_tokenizer_path=cfg.tokenizer_name,
        split=cfg.task_config.train_data_split,
        n_ctx=cfg.task_config.max_seq_len,
        seed=seed,  # Use run-specific seed
        column_name=cfg.task_config.column_name,
        is_tokenized=cfg.task_config.is_tokenized,
        streaming=cfg.task_config.streaming,
    )

    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=cfg.task_config.buffer_size,
        global_seed=seed,  # Use run-specific seed
    )

    return dataloader


def _create_resid_mlp_dataloader(model_path: str, batch_size: int, seed: int) -> DataLoader[Any]:
    """Create a dataloader for ResidMLP task."""
    from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
    from param_decomp.utils.data_utils import DatasetGeneratedDataLoader

    pd_run = ParamDecompRunInfo.from_path(model_path)
    cfg = pd_run.config
    component_model = ComponentModel.from_run_info(pd_run)

    assert isinstance(cfg.task_config, ResidMLPTaskConfig), (
        f"Expected task_config to be of type ResidMLPTaskConfig, but got {type(cfg.task_config) = }"
    )
    assert isinstance(component_model.target_model, ResidMLP), (
        f"Expected target_model to be of type ResidMLP, but got {type(component_model.target_model) = }"
    )

    # Create dataset with run-specific seed
    dataset = ResidMLPDataset(
        n_features=component_model.target_model.config.n_features,
        feature_probability=cfg.task_config.feature_probability,
        device="cpu",
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=seed,  # Use run-specific seed
        label_coeffs=None,
        data_generation_type=cfg.task_config.data_generation_type,
    )

    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
