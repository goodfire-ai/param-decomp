"""Authored metric schemas whose semantics require an LM target."""

from typing import ClassVar, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import HiddenActsReconstruction


class CEandKLLossesConfig(BaseConfig):
    """Token-level CE/KL metrics for categorical LM outputs."""

    slow: ClassVar[bool] = False
    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIMaskedAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"
    n_mask_samples: PositiveInt = 1


class ArithmeticCEKLConfig(BaseConfig):
    rounding_threshold: float


class ArithmeticCIL0Config(BaseConfig):
    ci_alive_threshold: float
    groups: dict[str, list[str]] | None


class ArithmeticFreshPGDConfig(BaseConfig):
    name: str | None = None
    n_steps: NonNegativeInt
    step_size: PositiveFloat
    hidden_acts_reconstruction: HiddenActsReconstruction | None = None


class ArithmeticProbeMetrics(BaseConfig):
    """Scalar operations evaluated on the arithmetic grid rather than corpus batches."""

    ce_kl: ArithmeticCEKLConfig
    ci_l0: ArithmeticCIL0Config
    fresh_pgd: ArithmeticFreshPGDConfig | None


class ArithmeticCIGridConfig(BaseConfig):
    """Per-component causal-importance heatmaps over an arithmetic operand grid."""

    slow: ClassVar[bool] = True
    type: Literal["ArithmeticCIGrid"] = "ArithmeticCIGrid"
    probe_metrics: ArithmeticProbeMetrics
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    thresholds: list[float] = Field(default_factory=lambda: [0.1])
    top_k: PositiveInt = 24


class TwoStreamCIMeanPerComponentConfig(BaseConfig):
    """Both streams' mean CI per component on ONE axis per site, ordered by descending
    TARGET mean and coloured by stream.

    Computes `CIMeanPerComponent`'s reduction on both streams, so authoring both pays for
    the nontarget pass twice. Refuses on a plain run, which has no target stream."""

    slow: ClassVar[bool] = True
    type: Literal["TwoStreamCIMeanPerComponent"] = "TwoStreamCIMeanPerComponent"
