"""Fixture metric loaded via dotted module name in tests/test_metric_modules.py."""

from typing import Any, ClassVar, override

import torch
from torch import Tensor

from param_decomp.metrics.base import LossMetricConfig, Metric, MetricConfig, MetricResult
from param_decomp.metrics.registry import register_metric


class DottedFixtureLossConfig(LossMetricConfig):
    pass


@register_metric
class DottedFixtureLoss(Metric[DottedFixtureLossConfig]):
    section: ClassVar[str] = "loss"
    config_type: ClassVar[type[MetricConfig]] = DottedFixtureLossConfig
    slow: ClassVar[bool] = False
    short_name: ClassVar[str | None] = None

    def __init__(self, cfg: DottedFixtureLossConfig, *, model: Any, device: str) -> None:
        self.cfg = cfg

    @override
    def reset(self) -> None:
        pass

    @override
    def update(self, ctx: Any) -> Tensor:
        _ = ctx
        return torch.zeros(())

    @override
    def compute(self) -> MetricResult:
        return torch.zeros(())
