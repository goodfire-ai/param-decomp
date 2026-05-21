"""Metric protocol and config base classes.

Each metric file defines its pydantic config class alongside the `Metric` class itself. Eval
metric configs subclass `BaseConfig` directly; loss metrics subclass `LossMetricConfig` (which
carries the required `coeff` field for training). Loss metrics are referenced from
`PDConfig.loss_metrics` as a pydantic discriminated union keyed on each subclass's `type`
literal; the runtime dispatch to the matching `Metric` class lives in
`param_decomp.metrics.loss_metrics.LOSS_METRIC_CLASSES`. Eval metrics are instantiated by
the caller and passed to `optimize(eval_metrics=...)`.

Metrics are instantiated with just the validated config (`MyMetric(cfg)`). The training loop
calls `metric.bind(model=component_model, device=...)` once before any other method, then
`update(ctx)` per step / per eval batch and `compute()` per eval pass. `reset()` is called
inside `bind()` to initialise stateful tensors on the bound device, and before each eval pass.
Loss-capable metrics' accumulators must `.detach()` before adding to avoid retaining the
autograd graph across training steps.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Number
from typing import Any, ClassVar

import wandb.plot
from PIL import Image
from torch import Tensor

from param_decomp.base_config import BaseConfig


class LossMetricConfig(BaseConfig):
    """Pydantic config for a metric that can also be used as a training loss.

    `coeff` is required when this metric is listed under `loss_metrics` (asserted by PDConfig's
    field validator) and ignored when an eval-only instance is constructed directly.
    """

    coeff: float | None = None


MetricResult = (
    Tensor | Mapping[str, Tensor | float | Number | str | Image.Image | wandb.plot.CustomChart]
)


class Metric[TConfig: BaseConfig](ABC):
    """Abstract base class that every metric must subclass."""

    section: ClassVar[str]
    config_type: ClassVar[type[BaseConfig]]
    slow: ClassVar[bool] = False
    short_name: ClassVar[str | None] = None
    cfg: TConfig
    model: Any
    device: str

    def __init__(self, cfg: TConfig) -> None:
        """Initialize the metric from its validated config.

        Construction does not bind runtime resources. The training loop calls
        :meth:`bind` once with the live `ComponentModel` and device before any
        other method on the metric is invoked.
        """
        self.cfg = cfg
        self._bound = False

    def bind(self, *, model: Any, device: str) -> None:
        """Attach the component model and device, then call `reset()`.

        Called by the training loop after the `ComponentModel` is constructed. Subclasses
        that need additional bind-time setup (e.g. resolving module paths against the model)
        should override and call `super().bind(...)` first.
        """
        assert not self._bound, f"{type(self).__name__} is already bound"
        self.model = model
        self.device = device
        self._bound = True
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state before an evaluation pass.

        Stateless metrics may implement this as a no-op. Stateful metrics should reset counters,
        sums, cached examples, plots, or adversarial eval state so a subsequent `compute()` only
        reflects batches processed after this call. Called automatically inside `bind()` to
        initialise device-typed accumulators.
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
