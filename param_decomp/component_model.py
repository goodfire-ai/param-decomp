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
from param_decomp.ci_fns import (
    CiConfig,
    GlobalCiFnWrapper,
    LayerwiseCiFnWrapper,
    make_ci_fn_wrapper,
)
from param_decomp.ci_sigmoids import SIGMOID_TYPES, SigmoidType
from param_decomp.components import Components, make_components
from param_decomp.decomposition_targets import DecompositionTarget, Identity
from param_decomp.masks import ComponentsMaskInfo, SamplingType


class OutputWithCache(NamedTuple):
    """Forward output paired with cached activations.

    Attributes:
        output: Model output tensor from the forward pass.
        cache: Per-module activations captured by forward hooks. Keys are
            target-module paths (or ``f"{path}_{kind}"`` for component-acts
            entries); the contents depend on ``cache_type`` chosen at the call
            site.
    """

    output: Tensor
    cache: dict[str, Tensor]


@dataclass
class CIOutputs:
    """Triple of CI tensors keyed by target module path.

    Attributes:
        lower_leaky: CI values squashed by the lower-leaky sigmoid. Multiplied
            into component contributions; bounded above by 1.
        upper_leaky: CI values squashed by the upper-leaky sigmoid. Used by
            importance-minimality losses; bounded below by 0.
        pre_sigmoid: Raw CI-fn outputs before any sigmoid.
    """

    lower_leaky: dict[str, Float[Tensor, "... C"]]
    upper_leaky: dict[str, Float[Tensor, "... C"]]
    pre_sigmoid: dict[str, Tensor]


class ComponentModel(nn.Module):
    """Wrapper around a frozen target model that exposes parameter components.

    The underlying *base model* can be any subclass of ``nn.Module`` (e.g.
    ``LlamaForCausalLM``, ``AutoModelForCausalLM``) as long as the sub-module
    paths to decompose are provided in ``decomposition_targets``. The wrapper
    registers components and the causal-importance function (``ci_fn``) as
    submodules so they participate in ``DistributedDataParallel`` parameter
    sync and ``.to(device)`` semantics.

    Forward pass supports four cache modes and optional component replacement.
    See :meth:`forward` for the matrix of behaviors.

    Attributes:
        target_model: The frozen base model. Its parameters must not require
            grad — the constructor asserts this.
        module_to_c: Map from decomposition-target module path to the number of
            components ``C`` for that module.
        target_module_paths: Ordered list of ``module_to_c`` keys.
        components: ``Components`` instance per decomposition-target path.
        ci_fn: The CI-fn wrapper (layerwise or global) that maps activations to
            per-component CI values.
        lower_leaky_fn: Sigmoid applied to produce ``CIOutputs.lower_leaky``.
        upper_leaky_fn: Sigmoid applied to produce ``CIOutputs.upper_leaky``.
    """

    def __init__(
        self,
        target_model: nn.Module,
        run_batch: RunBatch,
        decomposition_targets: list[DecompositionTarget],
        ci_config: CiConfig,
        sigmoid_type: SigmoidType,
    ):
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

        self.ci_fn: LayerwiseCiFnWrapper | GlobalCiFnWrapper | None = make_ci_fn_wrapper(
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

    def drop_ci_fn(self) -> None:
        """Free the CI fn — for pools that only receive CI values via NCCL.

        Call AFTER ``__init__`` (so RNG draws used to init the CI fn stay
        identical across pools) but BEFORE ``.to(device)`` (so the unused
        params never touch GPU memory). Idempotent.
        """
        if self.ci_fn is not None:
            del self.ci_fn
            self.ci_fn = None

    def drop_components(self) -> None:
        """Free the per-site V/U components — for pools that don't run V/U.

        Only safe when no component is an ``EmbeddingComponents`` (the CI fn
        wrapper consults those to convert token IDs to acts). Asserted at
        runtime. Same lifecycle as ``drop_ci_fn``: post-init, pre-device.
        """
        from param_decomp.components import EmbeddingComponents

        assert not any(isinstance(c, EmbeddingComponents) for c in self.components.values()), (
            "drop_components() called on a ComponentModel with embedding components — "
            "the CI fn wrapper needs those to convert token IDs to acts."
        )
        # Both the in-place clear (for the dict reference held by ci_fn wrapper)
        # AND ModuleDict re-init (to unregister the children from nn.Module so
        # .to(device) skips them).
        self.components.clear()
        self._components = nn.ModuleDict()

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        """Return the weight matrix of a target module in PD's row-major convention.

        For ``transformers.pytorch_utils.Conv1D`` (Radford-style) the stored
        weight is transposed relative to ``nn.Linear``; this method returns it
        transposed back so all targets share the same ``[d_out, d_in]`` shape.
        For an :class:`Identity` shim the returned tensor is the identity matrix
        of size ``target_module.d`` on the model's device/dtype.

        Args:
            module_name: Path of the target module as registered in
                ``module_to_c``.
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

        With no extra args, this is just a forward pass through the frozen
        target model. If ``mask_infos`` is given, those modules' outputs are
        replaced by their components' forward pass under the supplied masks.
        ``cache_type`` controls what each hooked module records.

        Args:
            batch: The input batch, passed unchanged to ``self._run_batch``.
            mask_infos: Per-module mask info. If provided, the listed modules
                are replaced via forward hooks; if ``None`` and ``cache_type``
                is set, hooks are attached to every target module for caching
                only.
            cache_type: What each hooked module caches. ``"input"`` caches
                pre-weight activations, ``"output"`` caches post-weight
                activations, ``"component_acts"`` caches per-component
                activations, ``"none"`` disables caching.

        Returns:
            An :class:`OutputWithCache` when ``cache_type != "none"``,
            otherwise the bare output tensor.
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
        """Forward hook that handles both component replacement and caching.

        Args:
            args: Positional args to the hooked module. Must be a single tensor.
            kwargs: Keyword args to the hooked module. Must be empty.
            output: Original module output (returned unchanged unless
                components replace it).
            module_name: Path of the module in ``self.target_model``.
            components: ``Components`` for this module, or ``None`` for
                cache-only mode.
            mask_info: Mask payload for this module, or ``None`` for cache-only
                mode.
            cache_type: ``"input"``, ``"output"``, ``"component_acts"``, or
                ``"none"``.
            cache: Dict to populate; keyed by ``module_name`` (or
                ``f"{module_name}_{kind}"`` for component-acts entries).

        Returns:
            The replaced output when components are applied, otherwise
            ``None`` (which tells PyTorch to keep the original output).
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
        """Compute causal-importance values for every decomposition target.

        Runs the CI fn on the pre-weight activations, then squashes the outputs
        through the lower-leaky and upper-leaky sigmoids. Under
        ``sampling="binomial"`` the lower-leaky branch additionally has a small
        amount of uniform noise mixed in before squashing.

        Args:
            pre_weight_acts: Per-module activations entering each target layer.
                For embedding targets these are integer token indices.
            sampling: Sampling regime for stochastic masks. Controls the noise
                injection on the lower-leaky branch.
            detach_inputs: If ``True``, gradients do not flow from the CI fn
                back into ``pre_weight_acts``.

        Returns:
            :class:`CIOutputs` containing the lower-leaky, upper-leaky, and
            pre-sigmoid CI values keyed by target-module path.
        """
        assert self.ci_fn is not None, (
            "calc_causal_importances called on a pool whose CI fn was dropped "
            "(see ComponentModel.drop_ci_fn)"
        )
        if detach_inputs:
            pre_weight_acts = {k: v.detach() for k, v in pre_weight_acts.items()}

        if isinstance(self.ci_fn, GlobalCiFnWrapper):
            return self._sigmoid_and_split_global(self.ci_fn, pre_weight_acts, sampling)
        return self._sigmoid_per_site_layerwise(self.ci_fn, pre_weight_acts, sampling)

    def _sigmoid_and_split_global(
        self,
        ci_fn: GlobalCiFnWrapper,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
    ) -> CIOutputs:
        """Global CI fns produce a single concatenated ``[..., total_c]`` output. We
        apply the sigmoids once on the unsplit tensor (one elementwise op apiece +
        one ``rand_like``/assert) and then split into per-site dicts — vs. the old
        path which looped per-site and incurred ~96× the autograd / dispatch cost.
        """
        ci_fn_output = ci_fn(pre_weight_acts)
        if sampling == "binomial":
            ci_fn_output_for_lower_leaky = 1.05 * ci_fn_output - 0.05 * torch.rand_like(
                ci_fn_output
            )
        else:
            ci_fn_output_for_lower_leaky = ci_fn_output

        lower_leaky_output = self.lower_leaky_fn(ci_fn_output_for_lower_leaky)
        upper_leaky_output = self.upper_leaky_fn(ci_fn_output)

        layer_order = ci_fn.layer_order
        split_sizes = ci_fn.split_sizes
        lower_splits = torch.split(lower_leaky_output, split_sizes, dim=-1)
        upper_splits = torch.split(upper_leaky_output, split_sizes, dim=-1)
        pre_splits = torch.split(ci_fn_output, split_sizes, dim=-1)
        return CIOutputs(
            lower_leaky={name: lower_splits[i] for i, name in enumerate(layer_order)},
            upper_leaky={name: upper_splits[i] for i, name in enumerate(layer_order)},
            pre_sigmoid={name: pre_splits[i] for i, name in enumerate(layer_order)},
        )

    def _sigmoid_per_site_layerwise(
        self,
        ci_fn: "LayerwiseCiFnWrapper",
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
    ) -> CIOutputs:
        """Layerwise CI fns produce a dict of per-site tensors that don't share
        storage; sigmoid them per-site (one autograd node per site is unavoidable
        here, unlike the global case)."""
        ci_fn_outputs = ci_fn(pre_weight_acts)
        causal_importances_lower_leaky: dict[str, Tensor] = {}
        causal_importances_upper_leaky: dict[str, Tensor] = {}
        pre_sigmoid: dict[str, Tensor] = {}
        for target_module_name, ci_fn_output in ci_fn_outputs.items():
            if sampling == "binomial":
                ci_fn_output_for_lower_leaky = 1.05 * ci_fn_output - 0.05 * torch.rand_like(
                    ci_fn_output
                )
            else:
                ci_fn_output_for_lower_leaky = ci_fn_output
            lower_leaky_output = self.lower_leaky_fn(ci_fn_output_for_lower_leaky)
            causal_importances_lower_leaky[target_module_name] = lower_leaky_output
            upper_leaky_output = self.upper_leaky_fn(ci_fn_output)
            causal_importances_upper_leaky[target_module_name] = upper_leaky_output
            pre_sigmoid[target_module_name] = ci_fn_output
        return CIOutputs(
            lower_leaky=causal_importances_lower_leaky,
            upper_leaky=causal_importances_upper_leaky,
            pre_sigmoid=pre_sigmoid,
        )

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Per-target ``target_weight - sum_components`` deltas.

        Returns:
            For each decomposition target, the difference between its target
            weight and the summed component weights (``V @ U``). Used by the
            delta-component pathway and by faithfulness diagnostics.
        """
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


def component_grad_norms(
    component_model: ComponentModel, device: torch.device | str
) -> dict[str, float]:
    """Per-parameter and summary gradient norms for components and the CI fn.

    Args:
        component_model: The unwrapped model whose ``.components`` and
            ``.ci_fn`` are inspected.
        device: Device the running sums are accumulated on.

    Returns:
        A flat dict with three families of keys:

        - ``components/<module_path>.<param>`` — L2 norm of each component
          parameter's gradient. ``NaN`` if its grad was never populated.
        - ``ci_fns/<param>`` — L2 norm of each CI-fn parameter's gradient.
          ``NaN`` if its grad was never populated.
        - ``summary/components``, ``summary/ci_fns``, ``summary/total`` —
          aggregate L2 norms over each pool and over both pools. ``NaN`` if
          any contributing grad was missing.
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
    assert component_model.ci_fn is not None, (
        "compute_grad_norms called on a ComponentModel whose CI fn was dropped"
    )
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
