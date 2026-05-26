"""`ComponentModel` — wraps the target model with `Components` modules + a CI fn.

Emits gradient-aware cached forward passes consumed by the loss metrics.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.base_config import runtime_cast
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import CiConfig, make_ci_fn_wrapper
from param_decomp.ci_sigmoids import SIGMOID_TYPES, SigmoidType
from param_decomp.components import Components, make_components
from param_decomp.decomposition_targets import DecompositionTarget, Identity
from param_decomp.masks import ComponentsMaskInfo, SamplingType


class OutputWithCache(NamedTuple):
    """Forward output paired with per-module cached activations.

    Cache keys are target-module paths (or `f"{path}_{kind}"` for component-acts entries);
    contents depend on the `cache_type` requested.
    """

    output: Tensor
    cache: dict[str, Tensor]


@dataclass
class CIOutputs:
    """Triple of CI tensors keyed by target module path.

    `lower_leaky` is multiplied into component contributions (bounded above by 1);
    `upper_leaky` is used by importance-minimality losses (bounded below by 0);
    `pre_sigmoid` is the raw CI-fn output.
    """

    lower_leaky: dict[str, Float[Tensor, "... C"]]
    upper_leaky: dict[str, Float[Tensor, "... C"]]
    pre_sigmoid: dict[str, Tensor]


class ComponentModel(nn.Module):
    """Wrapper around a frozen target model that exposes parameter components.

    The underlying base model can be any `nn.Module` (e.g. `LlamaForCausalLM`,
    `AutoModelForCausalLM`) as long as the sub-module paths to decompose are
    provided in `decomposition_targets`. The wrapper registers components and the
    causal-importance function (`ci_fn`) as submodules so they participate in DDP
    parameter sync and `.to(device)` semantics.

    The target model's parameters must not require grad — the constructor asserts this.
    Forward pass supports four cache modes and optional component replacement; see
    `forward` for the matrix of behaviors.
    """

    def __init__(
        self,
        target_model: nn.Module,
        run_batch: RunBatch,
        decomposition_targets: list[DecompositionTarget],
        ci_config: CiConfig,
        sigmoid_type: SigmoidType,
    ):
        """Wrap `target_model` with parameter-component machinery.

        Args:
            target_model: Frozen model whose weights are being decomposed. Constructor
                asserts every parameter has `requires_grad=False`.
            run_batch: Callable that runs the target model on one batch and returns its
                output tensor; invoked through the wrapper for caching / DDP.
            decomposition_targets: Resolved target list — one `(module_path, C)` per
                module to decompose. Produced by `resolve_decomposition_targets`.
            ci_config: Discriminated CI-fn config selecting layerwise vs global.
            sigmoid_type: Sigmoid used to squash raw CI-fn outputs. `"leaky_hard"`
                splits into lower- and upper-leaky variants; everything else uses one
                function for both branches.
        """
        super().__init__()
        self._run_batch: RunBatch = run_batch

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.module_to_c = {target.module_path: target.C for target in decomposition_targets}
        self.target_module_paths = list(self.module_to_c.keys())

        self.components = make_components(target_model, self.module_to_c)
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
        )

        self.ci_fn = make_ci_fn_wrapper(
            target_model=target_model,
            module_to_c=self.module_to_c,
            components=self.components,
            ci_config=ci_config,
        )

        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            # For other sigmoid types, use the same function for both
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        """Weight matrix of a target module in PD's `[d_out, d_in]` row-major convention.

        Radford `Conv1D` is transposed back from its stored `[d_in, d_out]` layout so all
        targets share the same shape. For an `Identity` shim the returned tensor is the
        identity matrix of size `target_module.d` on the model's device/dtype.
        """
        target_module = self.target_model.get_submodule(module_name)

        match target_module:
            case RadfordConv1D():
                return target_module.weight.T
            case nn.Linear() | nn.Embedding():
                return target_module.weight
            case Identity():
                p = next(self.parameters())
                return torch.eye(target_module.d, device=p.device, dtype=p.dtype)
            case _:
                raise ValueError(f"Module {target_module} not supported")

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["component_acts"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["input"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["output"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
    ) -> Tensor: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "output", "none"] = "none",
    ) -> Tensor | OutputWithCache:
        """Run the target model with optional component replacement and/or caching.

        With no extra args, this is just a forward pass through the frozen target model.
        If `mask_infos` is given, those modules' outputs are replaced by their
        components' forward pass under the supplied masks. Returns an `OutputWithCache`
        when `cache_type != "none"`, else the bare output.

        Args:
            batch: Passed unchanged to the wrapped `run_batch` callable.
            mask_infos: Per-module mask payload. If set, listed modules are replaced via
                forward hooks running the corresponding `Components` instance.
            cache_type: What each hooked module records. `"input"` caches pre-weight
                activations; `"output"` caches post-weight (post-replacement) outputs;
                `"component_acts"` caches per-component activations under the keys
                `f"{module_path}_pre_detach"` / `f"{module_path}_post_detach"`;
                `"none"` disables caching.
        """
        if mask_infos is None and cache_type == "none":
            return self._run_batch(self.target_model, batch)

        cache: dict[str, Tensor] = {}
        hooks: dict[str, Callable[..., Any]] = {}

        hook_module_names = list(mask_infos.keys()) if mask_infos else self.target_module_paths

        for module_name in hook_module_names:
            mask_info = mask_infos[module_name] if mask_infos else None
            components = self.components[module_name] if mask_info else None

            hooks[module_name] = partial(
                self._components_and_cache_hook,
                module_name=module_name,
                components=components,
                mask_info=mask_info,
                cache_type=cache_type,
                cache=cache,
            )

        with self._attach_forward_hooks(hooks):
            out: Tensor = self._run_batch(self.target_model, batch)

        match cache_type:
            case "input" | "output" | "component_acts":
                return OutputWithCache(output=out, cache=cache)
            case "none":
                return out

    def _components_and_cache_hook(
        self,
        _module: nn.Module,
        args: list[Any],
        kwargs: dict[Any, Any],
        output: Any,
        module_name: str,
        components: Components | None,
        mask_info: ComponentsMaskInfo | None,
        cache_type: Literal["component_acts", "input", "output", "none"],
        cache: dict[str, Tensor],
    ) -> Any | None:
        """Forward hook handling both component replacement and caching.

        Returns the replaced output when components are applied, else `None` (telling
        PyTorch to keep the original output).
        """
        assert len(args) == 1, "Expected 1 argument"
        assert len(kwargs) == 0, "Expected no keyword arguments"
        x = args[0]
        assert isinstance(x, Tensor), "Expected input tensor"

        if cache_type == "input":
            cache[module_name] = x

        if components is not None and mask_info is not None:
            assert isinstance(output, Tensor), (
                f"Only supports single-tensor outputs, got {type(output)}"
            )

            component_acts_cache = {} if cache_type == "component_acts" else None
            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
                component_acts_cache=component_acts_cache,
            )
            if component_acts_cache is not None:
                for k, v in component_acts_cache.items():
                    cache[f"{module_name}_{k}"] = v

            final_out = (
                components_out
                if mask_info.routing_mask == "all"
                else torch.where(mask_info.routing_mask[..., None], components_out, output)
            )

            if cache_type == "output":
                cache[module_name] = final_out
            return final_out

        # No component replacement - keep original output
        if cache_type == "output":
            assert isinstance(output, Tensor)
            cache[module_name] = output
        return None

    @contextmanager
    def _attach_forward_hooks(self, hooks: dict[str, Callable[..., Any]]) -> Generator[None]:
        """Attach forward hooks to the listed target modules for the block's lifetime."""
        handles: list[RemovableHandle] = []
        for module_name, hook in hooks.items():
            target_module = self.target_model.get_submodule(module_name)
            handle = target_module.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool = False,
    ) -> CIOutputs:
        """CI values for every decomposition target.

        Runs the CI fn on `pre_weight_acts` and squashes through both lower- and
        upper-leaky sigmoids. Under `sampling="binomial"`, the lower-leaky branch has a
        small amount of uniform noise mixed in before squashing.

        Args:
            pre_weight_acts: Per-module input activations (or token-id tensors for
                embedding targets), typically the cache from a `cache_type="input"`
                forward pass.
            sampling: Selects the stochastic mask regime; gates the noise injection on
                the lower-leaky branch.
            detach_inputs: When true, gradients do not flow from CI back into
                `pre_weight_acts`. Used by metrics that want to optimise CI without
                perturbing the upstream graph.
        """
        if detach_inputs:
            pre_weight_acts = {k: v.detach() for k, v in pre_weight_acts.items()}

        ci_fn_outputs = self.ci_fn(pre_weight_acts)
        return self._apply_sigmoid_to_ci_outputs(ci_fn_outputs, sampling)

    def _apply_sigmoid_to_ci_outputs(
        self,
        ci_fn_outputs: dict[str, Float[Tensor, "... C"]],
        sampling: SamplingType,
    ) -> CIOutputs:
        """Squash raw CI-fn outputs through the lower- and upper-leaky sigmoids."""
        causal_importances_lower_leaky = {}
        causal_importances_upper_leaky = {}
        pre_sigmoid = {}

        for target_module_name, ci_fn_output in ci_fn_outputs.items():
            if sampling == "binomial":
                ci_fn_output_for_lower_leaky = 1.05 * ci_fn_output - 0.05 * torch.rand_like(
                    ci_fn_output
                )
            else:
                ci_fn_output_for_lower_leaky = ci_fn_output

            lower_leaky_output = self.lower_leaky_fn(ci_fn_output_for_lower_leaky)
            assert (lower_leaky_output <= 1.0).all()
            causal_importances_lower_leaky[target_module_name] = lower_leaky_output

            upper_leaky_output = self.upper_leaky_fn(ci_fn_output)
            assert (upper_leaky_output >= 0).all()
            causal_importances_upper_leaky[target_module_name] = upper_leaky_output

            pre_sigmoid[target_module_name] = ci_fn_output

        return CIOutputs(
            lower_leaky=causal_importances_lower_leaky,
            upper_leaky=causal_importances_upper_leaky,
            pre_sigmoid=pre_sigmoid,
        )

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Per-target `target_weight - sum_components` residuals.

        Used by the delta-component pathway and by faithfulness diagnostics.
        """
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


def component_grad_norms(
    component_model: ComponentModel, device: torch.device | str
) -> dict[str, float]:
    """Per-parameter and summary gradient norms for components and the CI fn.

    Returns a flat dict with three key families:

    - `components/<module_path>.<param>` — L2 norm of each component parameter's
      gradient. `NaN` if its grad was never populated.
    - `ci_fns/<param>` — L2 norm of each CI-fn parameter's gradient. `NaN` if its grad
      was never populated.
    - `summary/components`, `summary/ci_fns`, `summary/total` — aggregate L2 norms over
      each pool and over both pools. `NaN` if any contributing grad was missing.
    """
    out: dict[str, float] = {}

    comp_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    missing_component_grad = False
    for target_module_path, component in component_model.components.items():
        for local_param_name, local_param in component.named_parameters():
            if local_param.grad is None:
                missing_component_grad = True
                out[f"components/{target_module_path}.{local_param_name}"] = float("nan")
                continue
            param_grad = runtime_cast(Tensor, local_param.grad)
            param_grad_sum_sq = param_grad.pow(2).sum()
            key = f"components/{target_module_path}.{local_param_name}"
            out[key] = param_grad_sum_sq.sqrt().item()
            comp_grad_norm_sq_sum += param_grad_sum_sq

    ci_fn_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    missing_ci_fn_grad = False
    for local_param_name, local_param in component_model.ci_fn.named_parameters():
        if local_param.grad is None:
            missing_ci_fn_grad = True
            key = f"ci_fns/{local_param_name}"
            assert key not in out, f"Key {key} already exists in grad norms log"
            out[key] = float("nan")
            continue
        ci_fn_grad = runtime_cast(Tensor, local_param.grad)
        ci_fn_grad_sum_sq = ci_fn_grad.pow(2).sum()
        key = f"ci_fns/{local_param_name}"
        assert key not in out, f"Key {key} already exists in grad norms log"
        out[key] = ci_fn_grad_sum_sq.sqrt().item()
        ci_fn_grad_norm_sq_sum += ci_fn_grad_sum_sq

    out["summary/components"] = (
        float("nan") if missing_component_grad else comp_grad_norm_sq_sum.sqrt().item()
    )
    out["summary/ci_fns"] = (
        float("nan") if missing_ci_fn_grad else ci_fn_grad_norm_sq_sum.sqrt().item()
    )
    out["summary/total"] = (
        float("nan")
        if missing_component_grad or missing_ci_fn_grad
        else (comp_grad_norm_sq_sum + ci_fn_grad_norm_sq_sum).sqrt().item()
    )
    return out
