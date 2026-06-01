"""`LMComponentModel` — the vendored 3-pool component model.

A clean reimplementation (deliberately not a refactor of the core `ComponentModel`) that
composes a `ComponentGPT2` (frozen target + in-tree trainable components, with a pure
checkpointable masked forward) with a causal-importance network. This is the object the LM
3-pool training path treats as its `component_model`.

The factorization is `(target-with-components) + (ci-fn)`: the model owns the forward, the
components, the weight deltas and the pre-weight-acts capture; this wrapper adds the CI fn and
the CI squashing. The CI sigmoid/split logic is reimplemented here rather than shared with core
(see the team's reimplement-then-unify preference) — but the value types it produces
(`CIOutputs`, `Components`) are the same ones the metrics consume.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp.ci_fns import (
    CiConfig,
    GlobalCiFnWrapper,
    LayerwiseCiFnWrapper,
    make_ci_fn_wrapper,
)
from param_decomp.ci_sigmoids import SIGMOID_TYPES, SigmoidType
from param_decomp.component_model import CIOutputs
from param_decomp.components import Components, make_components
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import SamplingType
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import GPT2Simple
from param_decomp_lab.experiments.lm.vendored.gpt2 import (
    ComponentGPT2,
    MaskInfos,
    PreWeightActs,
    componentize_gpt2,
)


class LMComponentModel(nn.Module):
    """A `ComponentGPT2` paired with a causal-importance network for the LM 3-pool path."""

    model: ComponentGPT2
    ci_fn: GlobalCiFnWrapper | LayerwiseCiFnWrapper | None
    lower_leaky_fn: Callable[[Tensor], Tensor]
    upper_leaky_fn: Callable[[Tensor], Tensor]

    def __init__(
        self,
        model: ComponentGPT2,
        ci_fn: GlobalCiFnWrapper | LayerwiseCiFnWrapper | None,
        sigmoid_type: SigmoidType,
    ):
        super().__init__()
        self.model = model
        self.ci_fn = ci_fn
        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    @classmethod
    def build(
        cls,
        target_model: GPT2Simple,
        decomposition_targets: list[DecompositionTarget],
        ci_config: CiConfig,
        sigmoid_type: SigmoidType,
    ) -> "LMComponentModel":
        """Build from a (to-be-frozen) `GPT2Simple`: make components + CI fn, then componentize.

        The CI fn is built from the still-`nn.Linear` target (so `make_ci_fn_wrapper` sees the
        original module shapes); `componentize_gpt2` then freezes the target and swaps the
        decomposition leaves for the same `Components` instances.
        """
        module_to_c = {t.module_path: t.C for t in decomposition_targets}
        components = make_components(target_model, module_to_c)
        ci_fn = make_ci_fn_wrapper(
            target_model=target_model,
            module_to_c=module_to_c,
            components=components,
            ci_config=ci_config,
        )
        model = componentize_gpt2(target_model, components)
        return cls(model, ci_fn, sigmoid_type)

    # --- forward + capture, delegated to the model ---

    @override
    def forward(self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None) -> Tensor:
        return self.model(idx, mask_infos)

    def forward_with_pre_weight_acts(
        self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        return self.model.forward_with_pre_weight_acts(idx, mask_infos)

    def pre_weight_acts(self, idx: Int[Tensor, "batch pos"]) -> PreWeightActs:
        return self.model.pre_weight_acts(idx)

    def forward_with_output_acts(
        self, idx: Int[Tensor, "batch pos"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        return self.model.forward_with_output_acts(idx, mask_infos)

    @contextmanager
    def bypass_lm_head(self) -> Iterator[Float[Tensor, "vocab d_model"]]:
        with self.model.bypass_lm_head() as lm_head_weight:
            yield lm_head_weight

    # --- pure queries over the components, delegated to the model ---

    @property
    def components(self) -> dict[str, Components]:
        return self.model.components

    @property
    def module_to_c(self) -> dict[str, int]:
        return self.model.module_to_c

    @property
    def target_module_paths(self) -> list[str]:
        return self.model.target_module_paths

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        return self.model.target_weight(module_name)

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        return self.model.calc_weight_deltas()

    # --- causal importances (reimplemented here) ---

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool = False,
    ) -> CIOutputs:
        if detach_inputs:
            pre_weight_acts = {k: v.detach() for k, v in pre_weight_acts.items()}
        assert self.ci_fn is not None, "calc_causal_importances called after drop_ci_fn"
        match self.ci_fn:
            case GlobalCiFnWrapper():
                return self._ci_global(self.ci_fn, pre_weight_acts, sampling)
            case LayerwiseCiFnWrapper():
                return self._ci_layerwise(self.ci_fn, pre_weight_acts, sampling)

    def _lower_leaky_input(self, ci_fn_output: Tensor, sampling: SamplingType) -> Tensor:
        if sampling == "binomial":
            return 1.05 * ci_fn_output - 0.05 * torch.rand_like(ci_fn_output)
        return ci_fn_output

    def _ci_global(
        self, ci_fn: GlobalCiFnWrapper, pre_weight_acts: dict[str, Tensor], sampling: SamplingType
    ) -> CIOutputs:
        out = ci_fn(pre_weight_acts)
        lower = self.lower_leaky_fn(self._lower_leaky_input(out, sampling))
        upper = self.upper_leaky_fn(out)
        order, sizes = ci_fn.layer_order, ci_fn.split_sizes
        lower_s = torch.split(lower, sizes, dim=-1)
        upper_s = torch.split(upper, sizes, dim=-1)
        pre_s = torch.split(out, sizes, dim=-1)
        return CIOutputs(
            lower_leaky={name: lower_s[i] for i, name in enumerate(order)},
            upper_leaky={name: upper_s[i] for i, name in enumerate(order)},
            pre_sigmoid={name: pre_s[i] for i, name in enumerate(order)},
        )

    def _ci_layerwise(
        self,
        ci_fn: LayerwiseCiFnWrapper,
        pre_weight_acts: dict[str, Tensor],
        sampling: SamplingType,
    ) -> CIOutputs:
        outs = ci_fn(pre_weight_acts)
        lower: dict[str, Tensor] = {}
        upper: dict[str, Tensor] = {}
        pre: dict[str, Tensor] = {}
        for name, out in outs.items():
            lower[name] = self.lower_leaky_fn(self._lower_leaky_input(out, sampling))
            upper[name] = self.upper_leaky_fn(out)
            pre[name] = out
        return CIOutputs(lower_leaky=lower, upper_leaky=upper, pre_sigmoid=pre)

    # --- pool lifecycle ---

    def drop_ci_fn(self) -> None:
        """Free the CI fn (pools that only receive CI via NCCL). Post-init, pre-`.to(device)`."""
        if self.ci_fn is not None:
            del self.ci_fn
            self.ci_fn = None

    def drop_components(self) -> None:
        """Free the per-site V/U params (the CI pool, which holds no V/U). Delegates to the
        model; post-init, pre-`.to(device)`. See `ComponentGPT2.drop_components`."""
        self.model.drop_components()
