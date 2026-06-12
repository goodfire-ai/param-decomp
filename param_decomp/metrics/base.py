"""`Metric` ABC (the `LossMetricConfig` base lives in `param_decomp_config.losses`).

Lifecycle: `MyMetric(cfg)` -> `bind(model, device)` (calls `reset()`) -> per-step
`update(ctx)` -> per-eval-pass `compute()`, with `reset()` between eval passes.
Loss-capable metrics' accumulators must `.detach()` before adding to avoid retaining the
autograd graph across training steps.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from torch import Tensor

from param_decomp.component_model import ComponentModelProtocol
from param_decomp_config.base import BaseConfig
from param_decomp_config.losses import LossMetricConfig

MetricResult = Tensor | Mapping[str, Any]


class Metric[TConfig: BaseConfig](ABC):
    """Abstract base class that every metric must subclass.

    Constructed from the validated config alone; runtime resources are attached later
    via `bind`. `log_namespace` is the namespace prefix for emitted keys; `slow` gates
    this metric behind the slow-eval cadence; `short_name` is the optional config-key
    short label consumed by lab-side logging helpers.
    """

    log_namespace: ClassVar[str]
    slow: ClassVar[bool] = False
    short_name: ClassVar[str | None] = None
    cfg: TConfig
    model: ComponentModelProtocol
    device: str

    def __init__(self, cfg: TConfig) -> None:
        """Construct from a validated config. Runtime resources are attached later by `bind`."""
        self.cfg = cfg
        self._bound = False

    @property
    def requires_clean_logits(self) -> bool:
        """Whether this metric needs model forwards (the step's clean forward and any
        forward it runs itself) to produce LOGITS.

        Metrics that never touch model outputs, or that run their forwards under the
        vendored LM-head bypass (fused KL), return False. When every loss metric does,
        the trainer may run the whole loss step under `bypass_lm_head`:
        `ctx.target_out` is then the pre-LM-head hidden state
        (`ctx.target_out_kind == "hidden"`) and no vocab-scale tensor is ever
        materialized on a train step.
        """
        return True

    @property
    def instance_key(self) -> str:
        """Identity for dict keys and log-key suffixes; defaults to the class name.

        A loss-capable config may override it via `LossMetricConfig.name` so two
        instances of the same metric class (one loss, one eval) stay distinct.
        """
        name = self.cfg.name if isinstance(self.cfg, LossMetricConfig) else None
        return name if name is not None else type(self).__name__

    def bind(self, *, model: ComponentModelProtocol, device: str) -> None:
        """Attach the live component model and device, then call `reset()`.

        Called once by the training loop before any other method. Subclasses needing
        additional bind-time setup (e.g. resolving module paths against the model)
        should override and call `super().bind(...)` first.

        Args:
            model: The component model this metric will observe — any
                `ComponentModelProtocol` (core `ComponentModel`, FSDP adapter, or the
                vendored `LMComponentModel`).
            device: Device string used for accumulators and any other allocated state.
        """
        assert not self._bound, f"{type(self).__name__} is already bound"
        self.model = model
        self.device = device
        self._bound = True
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state before an evaluation pass.

        Stateless metrics may implement this as a no-op. Stateful metrics should reset
        counters, sums, cached examples, plots, or adversarial eval state so a
        subsequent `compute()` only reflects batches processed after this call.
        Invoked automatically inside `bind()` to initialise device-typed accumulators.
        """
        ...

    @abstractmethod
    def update(self, ctx: Any) -> Tensor | None:
        """Process one batch and update accumulated state.

        Loss-capable metrics must `.detach()` before adding tensors to accumulators;
        otherwise the autograd graph is retained across steps and leaks memory.

        Args:
            ctx: The per-step `MetricContext` bundle (see `metrics/context.py`).

        Returns:
            The per-batch scalar when one exists. For loss-capable metrics that scalar
            is the live loss used for backprop. Eval-only metrics return `None`.
        """
        ...

    @abstractmethod
    def compute(self) -> MetricResult:
        """Return the scalar, artifact, or keyed outputs accumulated since the last `reset()`."""
        ...

    def after_backward(self) -> None:  # noqa: B027 — intentional no-op default
        """Hook called for each loss metric right after `total_loss.backward()`.

        Override when a metric needs to step internal state coupled to the outer
        backward — e.g. `PersistentPGDReconLoss` steps its adversarial sources here
        from the gradients that backward left on them.
        """

    def state_dict(self) -> dict[str, Any]:
        """Return persistent metric state for round-tripping across a training restart.

        Default is an empty dict: most metrics are stateless w.r.t. the optimizer
        trajectory (their accumulators reset between eval passes). Override when a
        metric carries trajectory-dependent state — e.g. `PersistentPGDReconLoss`
        round-trips its adversarial sources here. Mirrors the `nn.Module` / `Optimizer`
        convention; the parent `Trainer` composes these into its own state blob.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:  # noqa: B027 — intentional no-op default
        """Load persistent state produced by a prior :meth:`state_dict` call.

        Default is a no-op (matches the default empty `state_dict`). Override
        alongside `state_dict` for any metric carrying trajectory-dependent state.
        """
        del state
