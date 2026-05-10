"""Shared base for the per-experiment config variants.

Lives in its own module to avoid the import cycle between `experiment_config.py`
(which references all concrete variants to form the discriminated union) and the
variant configs themselves (which subclass `BaseExperimentConfig`).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.configs import PDConfig
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState


@dataclass(frozen=True)
class LoadedTarget:
    """Uniform return shape for an experiment's target loader.

    `target_train_config` is the config that produced the target weights. When set,
    `run_pd` writes both `target_train_config.yaml` and `target_model.pth` into the
    PD run dir, making the run self-contained. LM-via-HF leaves it None — the target
    is already addressable by HF id and bundling Llama weights would balloon the run
    dir. TMS/ResidMLP/IH targets are small enough to bundle.
    """

    target: PDTarget
    target_train_config: BaseConfig | None = None


class BaseExperimentConfig(BaseConfig, ABC):
    """Contract every experiment config satisfies.

    Concrete subclasses live in `param_decomp/experiments/<kind>/configs.py` and add
    the variant-specific `kind: Literal[...]`, `target`, and `data` fields. The
    discriminated union over all subclasses is built in `experiment_config.py`.
    """

    pd: PDConfig

    @abstractmethod
    def load_target(self) -> LoadedTarget: ...

    @abstractmethod
    def build_dataloaders(
        self,
        *,
        seed: int,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> tuple[DataLoader[Any], DataLoader[Any]]: ...

    @abstractmethod
    def display_name(self) -> str: ...
