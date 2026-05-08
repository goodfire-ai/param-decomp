"""Build ResidMLP dataloaders from `ResidMLPDataConfig`."""

from torch import Tensor

from param_decomp.experiments.resid_mlp.configs import ResidMLPDataConfig
from param_decomp.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader


def build_resid_mlp_dataloaders(
    data_cfg: ResidMLPDataConfig,
    target_model: ResidMLP,
    target_run_info: ResidMLPTargetRunInfo,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    device: str,
) -> tuple[
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
]:
    dataset = ResidMLPDataset(
        n_features=target_model.config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        calc_labels=False,  # labels come from the target model output
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type=data_cfg.data_generation_type,
        synced_inputs=target_run_info.config.synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=train_batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, eval_loader
