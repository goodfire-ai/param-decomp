"""Metric protocol and config base classes.

Metrics are auto-registered via `@register_metric` and looked up by their class name from
`PDConfig.loss_metrics` / `PDConfig.eval_metrics`. Each metric file defines its pydantic config
class (subclassing `MetricConfig` for eval-only or `LossMetricConfig` for loss-capable) alongside
the `Metric` class itself.

A metric's `update(ctx)` is called once per training step (returning the live loss for
loss-capable metrics) and once per eval batch. Eval reads `compute()` after the last batch.
`reset()` is called before each eval pass; loss-capable metrics' accumulators must `.detach()`
before adding to avoid retaining the autograd graph across training steps.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Number
from typing import Any, ClassVar

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


MetricResult = (
    Tensor | Mapping[str, Tensor | float | Number | str | Image.Image | wandb.plot.CustomChart]
)


class Metric[TConfig: MetricConfig](ABC):
    """Abstract base class that every metric must subclass."""

    section: ClassVar[str]
    config_type: ClassVar[type[MetricConfig]]
    slow: ClassVar[bool] = False
    short_name: ClassVar[str | None]
    cfg: TConfig

    @abstractmethod
    def __init__(self, cfg: TConfig, *, model: Any, device: str) -> None:
        """Initialize one metric instance from validated config and shared runtime objects.

        `model` is the component model being optimized or evaluated, and `device` is the target
        torch device string used by the run.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state before an evaluation pass.

        Stateless metrics may implement this as a no-op. Stateful metrics should reset counters,
        sums, cached examples, plots, or adversarial eval state so a subsequent `compute()` only
        reflects batches processed after this call.
        """
        ...

    @abstractmethod
    def update(self, ctx: Any) -> Tensor | None:
        """Process one batch from the metric context and update metric state.

        Return the per-batch scalar when one exists. For loss-capable metrics, that scalar is the
        live loss used for backprop. Metrics that only accumulate evaluation state should return
        None.

        Loss-capable metrics must call `.detach()` before adding tensors to accumulators;
        otherwise the autograd graph is retained across steps and leaks memory.
        """
        ...

    @abstractmethod
    def compute(self) -> MetricResult:
        """Return the scalar, artifact, or keyed metric outputs accumulated by `update()`."""
        ...

    def before_backward(self, live_loss: Tensor | None) -> None:
        """Hook called for each loss metric right before `total_loss.backward()`.

        Default is a no-op. Override when a metric needs to extract gradients before the
        outer backward consumes them — e.g. `PersistentPGDReconLoss` uses this to grab
        source gradients with `retain_graph=True` before the outer step.

        `live_loss` is whatever this metric's `update()` returned for the current batch
        (or `None` if the metric was gated off this step).
        """
        del live_loss

    def after_backward(self) -> None:  # noqa: B027 — intentional no-op default
        """Hook called for each loss metric right after `total_loss.backward()`.

        Default is a no-op. Override when a metric needs to step internal state coupled
        to the outer backward — e.g. `PersistentPGDReconLoss` steps its adversarial
        sources here.
        """
