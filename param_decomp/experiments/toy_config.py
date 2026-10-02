"""Shared authored and resolved configuration for positionless toy experiments."""

from typing import Annotated, Literal

from pydantic import Field, PositiveInt

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFnArch
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.components import SiteSpec
from param_decomp.core.configs import ExplicitCSpec
from param_decomp.experiments.config import ExperimentConfig
from param_decomp.experiments.eval_config import EvalConfig


class LayerwiseMlpCIFnConfig(BaseConfig):
    """One independent MLP CI function per toy site."""

    type: Literal["layerwise_mlp"] = "layerwise_mlp"
    hidden_dims: list[PositiveInt] = Field(..., min_length=1)


class GlobalMlpCIFnConfig(BaseConfig):
    """One MLP CI function over all toy sites jointly."""

    type: Literal["global_mlp"] = "global_mlp"
    hidden_dims: list[PositiveInt] = Field(..., min_length=1)


class ToyDecompositionConfig(BaseConfig):
    """Explicit toy sites and their positionless CI architecture."""

    sites: ExplicitCSpec
    ci: Annotated[LayerwiseMlpCIFnConfig | GlobalMlpCIFnConfig, Field(discriminator="type")]


class ToyExperimentConfig(ExperimentConfig):
    """A toy seat: no `runtime:` section at all.

    A toy is single-device by construction and trains in seconds on CPU, so it has no
    compute substrate to author — no world size, no placement, no remat trade, no XLA
    tuning. `ExperimentConfig` carries no `runtime`, so `extra="forbid"` refuses the
    section outright."""

    eval: EvalConfig | None = None


ToyCIFnArch = LayerwiseMLPCIFnArch | GlobalMLPCIFnArch
"""The pointwise arches a positionless toy run carries."""


def build_toy_ci_fn_arch(
    ci_fn_config: LayerwiseMlpCIFnConfig | GlobalMlpCIFnConfig,
    input_names: tuple[str, ...],
    sites: tuple[SiteSpec, ...],
) -> ToyCIFnArch:
    """`input_names[i]` is the tap feeding `sites[i]` (the toy one-tap-per-site
    alignment), so the global arm reads each tap's width off its site's `d_in`."""
    match ci_fn_config:
        case LayerwiseMlpCIFnConfig():
            return LayerwiseMLPCIFnArch(
                hidden_dims=tuple(ci_fn_config.hidden_dims),
                has_position_axis=False,
                input_names=input_names,
            )
        case GlobalMlpCIFnConfig():
            return GlobalMLPCIFnArch(
                hidden_dims=tuple(ci_fn_config.hidden_dims),
                has_position_axis=False,
                input_taps=tuple(
                    TapSpec(key=name, width=site.d_in)
                    for name, site in zip(input_names, sites, strict=True)
                ),
            )
