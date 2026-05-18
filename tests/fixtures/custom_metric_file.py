"""Fixture metric loaded via absolute file path in tests/test_metric_modules.py."""

from typing import Any, ClassVar

import torch
from torch import Tensor

from param_decomp.metrics.base import LossMetricConfig
from param_decomp.metrics.registry import register_metric


class FileFixtureLossConfig(LossMetricConfig):
    pass


@register_metric
class FileFixtureLoss:
    name: ClassVar[str] = "_test_metric_modules_file_loss"
    section: ClassVar[str] = "loss"
    config_type: ClassVar[type[LossMetricConfig]] = FileFixtureLossConfig
    slow: ClassVar[bool] = False
    short_name: ClassVar[str | None] = None

    def __init__(self, cfg: FileFixtureLossConfig, *, model: Any, device: str) -> None:
        self.cfg = cfg

    def reset(self) -> None:
        pass

    def update(self, _ctx: Any) -> Tensor:
        return torch.zeros(())

    def compute(self) -> Tensor:
        return torch.zeros(())
