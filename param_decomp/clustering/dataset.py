"""Dataset loading utilities for clustering runs.

Each clustering run loads its own dataset, seeded by the run index.
"""

from typing import Any

from torch.utils.data import DataLoader

from param_decomp.data import DatasetConfig, create_data_loader
from param_decomp.experiments.lm.configs import LMExperimentConfig
from param_decomp.experiments.resid_mlp.configs import ResidMLPExperimentConfig
from param_decomp.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp.load import load_pd, load_target_from_experiment_config
from param_decomp.models.component_model import ComponentModel, PDRunInfo
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
    pd_run = PDRunInfo.from_path(model_path)
    exp = pd_run.experiment_config
    assert isinstance(exp, LMExperimentConfig), (
        f"Expected LM experiment, got {exp.kind if exp is not None else None!r}"
    )
    data = exp.data

    dataset_config = DatasetConfig(
        name=data.dataset_name,
        hf_tokenizer_path=data.tokenizer_name,
        split=data.train_split,
        n_ctx=data.max_seq_len,
        seed=seed,  # Use run-specific seed
        column_name=data.column_name,
        is_tokenized=data.is_tokenized,
        streaming=data.streaming,
    )

    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=data.buffer_size,
        global_seed=seed,
    )

    return dataloader


def _create_resid_mlp_dataloader(model_path: str, batch_size: int, seed: int) -> DataLoader[Any]:
    """Create a dataloader for ResidMLP task."""
    from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
    from param_decomp.utils.data_utils import DatasetGeneratedDataLoader

    pd_run = PDRunInfo.from_path(model_path)
    exp = pd_run.experiment_config
    assert isinstance(exp, ResidMLPExperimentConfig), (
        f"Expected ResidMLP experiment, got {exp.kind if exp is not None else None!r}"
    )
    target = load_target_from_experiment_config(exp)
    component_model: ComponentModel = load_pd(model_path, target=target)

    assert isinstance(component_model.target_model, ResidMLP), (
        f"Expected target_model to be of type ResidMLP, got {type(component_model.target_model)}"
    )
    target_run_info = ResidMLPTargetRunInfo.from_path(exp.target.run_path)

    dataset = ResidMLPDataset(
        n_features=component_model.target_model.config.n_features,
        feature_probability=exp.data.feature_probability,
        device="cpu",
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=seed,
        label_coeffs=None,
        data_generation_type=exp.data.data_generation_type,
        synced_inputs=target_run_info.config.synced_inputs,
    )

    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
