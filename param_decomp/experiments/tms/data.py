"""Build TMS dataloaders from `TMSDataConfig`."""

from torch import Tensor

from param_decomp.experiments.tms.configs import TMSDataConfig
from param_decomp.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset


def build_tms_dataloaders(
    data_cfg: TMSDataConfig,
    target_model: TMSModel,
    target_run_info: TMSTargetRunInfo,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    device: str,
) -> tuple[
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
]:
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        data_generation_type=data_cfg.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=target_run_info.config.synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=train_batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, eval_loader
