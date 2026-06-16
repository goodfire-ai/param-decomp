"""TMS PD experiment: builders + the `SavedTMSRun` reload class.

The torch training driver has been retired (the JAX single-pool trainer is
production; the torch oracle lives at git tag `torch-oracle`). What remains is the
consumer bridge: the pure builders (`build_target`, `build_tms_loader`,
`make_run_batch`) and `SavedTMSRun`, which load a saved TMS decomposition off disk.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from torch.utils.data import DataLoader

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState
from param_decomp_config.base import BaseConfig, Probability
from param_decomp_config.experiment import ExperimentConfig
from param_decomp_lab.batch_and_loss_fns import run_batch_first_element
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.experiments.tms.data import SparseFeatureDataset
from param_decomp_lab.experiments.tms.models import TMSModel, TMSTargetRunInfo
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files


class TMSTargetConfig(BaseConfig):
    run_path: str = Field(..., description="Local or wandb path to a TMS pretrain run.")


class TMSDataConfig(BaseConfig):
    """Synthetic-feature dataset settings for TMS PD."""

    feature_probability: Probability
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = (
        "at_least_zero_active"
    )


class TMSExperimentConfig(ExperimentConfig[TMSTargetConfig, TMSDataConfig]):
    pass


def build_target(target_cfg: TMSTargetConfig) -> TMSModel:
    """Load the pretrained TMS target model in eval mode."""
    run_info = TMSTargetRunInfo.from_path(target_cfg.run_path)
    target_model = TMSModel.from_run_info(run_info)
    target_model.eval()
    return target_model


def build_tms_loader(
    target_cfg: TMSTargetConfig,
    data_cfg: TMSDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Synthetic `SparseFeatureDataset` loader for TMS.

    The dataset is infinite, so `split` / `dist_state` / `seed` are ignored — train and
    eval loaders are identical.
    """
    del split, dist_state, seed
    train_config = TMSTargetRunInfo.from_path(target_cfg.run_path).config
    dataset = SparseFeatureDataset(
        n_features=train_config.tms_model_config.n_features,
        feature_probability=data_cfg.feature_probability,
        device=device,
        batch_size=batch_size,
        data_generation_type=data_cfg.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=train_config.synced_inputs,
    )
    return DataLoader(dataset, batch_size=None)


def make_run_batch(target_cfg: TMSTargetConfig) -> RunBatch:
    """`RunBatch` for TMS: unwraps the `(inputs, labels)` tuple."""
    del target_cfg
    return run_batch_first_element


@dataclass(frozen=True)
class SavedTMSRun:
    """Handle to a completed TMS PD run on disk or in W&B."""

    cfg: TMSExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedTMSRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedTMSRun`."""
        files = resolve_run_files(
            path, config_filename=EXPERIMENT_CONFIG_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=TMSExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )
