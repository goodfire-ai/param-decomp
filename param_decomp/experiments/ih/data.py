"""Build IH dataloaders from `IHDataConfig`."""

from torch import Tensor

from param_decomp.experiments.ih.configs import IHDataConfig
from param_decomp.experiments.ih.model import InductionModelTargetRunInfo
from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset


def build_ih_dataloaders(
    data_cfg: IHDataConfig,
    target_run_info: InductionModelTargetRunInfo,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    device: str,
) -> tuple[
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
    DatasetGeneratedDataLoader[tuple[Tensor, Tensor]],
]:
    seq_len = target_run_info.config.ih_model_config.seq_len
    prefix_window = data_cfg.prefix_window or seq_len - 3
    dataset = InductionDataset(
        vocab_size=target_run_info.config.ih_model_config.vocab_size,
        seq_len=seq_len,
        prefix_window=prefix_window,
        device=device,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=train_batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, eval_loader
