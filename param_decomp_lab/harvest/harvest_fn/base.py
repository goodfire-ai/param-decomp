from typing import Protocol

import torch

from param_decomp_lab.harvest.schemas import HarvestBatch


class HarvestFn(Protocol):
    def __call__(self, batch_item: torch.Tensor) -> HarvestBatch: ...
