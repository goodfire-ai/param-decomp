"""Metric protocol and config base classes.

Metrics are auto-registered via `@register_metric` and looked up by their `name` ClassVar from
`PDConfig.loss_metrics` / `PDConfig.eval_metrics`. Each metric file defines its pydantic config
class (subclassing `MetricConfig` for eval-only or `LossMetricConfig` for loss-capable) alongside
the `Metric` class itself.

A metric's `update(ctx)` is called once per training step (returning the live loss for
loss-capable metrics) and once per eval batch. Eval reads `compute()` after the last batch.
`reset()` is called before each eval pass; loss-capable metrics' accumulators MUST `.detach()`
before adding to avoid retaining the autograd graph across training steps.
"""

from numbers import Number
from typing import Any, ClassVar, Protocol

import wandb.plot
from PIL import Image
from torch import Tensor

from param_decomp.base_config import BaseConfig


class MetricConfig(BaseConfig):
    """Pydantic config for an eval-only metric. Subclass and add fields as needed."""


class LossMetricConfig(MetricConfig):
    """Pydantic config for a metric that can also be used as a training loss.

    `coeff` is required when this metric is listed under `loss_metrics` (asserted by PDConfig's
    field validator) and ignored when listed under `eval_metrics`.
    """

    coeff: float | None = None


MetricResult = Tensor | dict[str, Tensor | Number | str | Image.Image | wandb.plot.CustomChart]


class Metric(Protocol):
    """Structural protocol that every metric must satisfy.

    Concrete metric classes should NOT subclass `Metric` — Python structural Protocols are
    satisfied implicitly by matching the API. This avoids multi-inheritance and override-variance
    issues with concrete return types.
    """

    name: ClassVar[str]
    section: ClassVar[str]
    config_type: ClassVar[type[MetricConfig]]
    slow: ClassVar[bool]
    short_name: ClassVar[str | None]
    cfg: MetricConfig

    def __init__(self, cfg: MetricConfig, *, model: Any, device: str) -> None: ...

    def reset(self) -> None: ...

    def update(self, ctx: Any) -> Tensor | None:
        """Process one batch. Accumulates state. Returns the per-batch scalar (the live loss when
        loss-capable, used for backprop), or None for metrics without a per-batch scalar.

        Loss-capable metrics MUST .detach() before adding to accumulators; otherwise the autograd
        graph is retained across steps and leaks memory.
        """
        ...

    def compute(self) -> MetricResult: ...


# Opt-in hooks (not part of `Metric` — `run_pd` discovers them via `getattr`):
#   before_backward(live_loss: Tensor | None) -> None
#   after_backward() -> None
# Currently used only by `PersistentPGDReconLoss` to extract source gradients with
# `retain_graph=True` and step its adversarial sources around `total_loss.backward()`.
