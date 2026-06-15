"""`FsdpComponentAdapter` — presents the core `ComponentModel` surface over a vendored
`LMComponentModel` so the shared step helpers + loss/eval metrics consume it unchanged.

`fully_shard` is applied to the inner blocks (`lm.model._layers` / CI-fn blocks), so the
adapter holds the wrapped `LMComponentModel` as its only submodule and owns no parameters
of its own. It exists purely to map the core `forward(batch, mask_infos, cache_type)`
overloads + `forward_with_output_acts` onto the vendored model's named methods, and to
re-expose the pure queries the metrics read directly.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, overload, override

import einops
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from param_decomp.ci_fns import GlobalCiFnWrapper, LayerwiseCiFnWrapper
from param_decomp.component_model import CIOutputs, OutputWithCache
from param_decomp.components import Components
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_config.routing import SamplingType
from param_decomp_lab.experiments.lm.vendored.component_model import (
    ComponentTarget,
    LMComponentModel,
)


def _replicate(t: Tensor) -> Tensor:
    """Gather a sharded `DTensor` to a `Replicate` DTensor; pass a plain tensor through.

    Crucially this is NOT `full_tensor()`/`.to_local()`: it keeps the result a DTensor, so
    its backward redistributes the gradient back to the source `Shard` placement. Gathering
    the V/U *inputs* this way (then einsum-ing the replicated copies) makes the faithfulness
    path's grad to V/U land in the params' native `Shard(0)` placement — the same placement
    the recon forward produces — so FSDP2's gradient reduce-scatter sees consistent grads
    when both paths accumulate into the same param.
    """
    return t.redistribute(placements=[Replicate()]) if isinstance(t, DTensor) else t


class FsdpComponentAdapter(nn.Module):
    def __init__(self, lm: LMComponentModel):
        super().__init__()
        self.lm = lm

    @overload
    def __call__(
        self,
        batch: Int[Tensor, "batch pos"],
        cache_type: Literal["input"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Int[Tensor, "batch pos"],
        cache_type: Literal["output"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Int[Tensor, "batch pos"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
    ) -> Tensor: ...

    @override
    def __call__(self, *args: object, **kwargs: object) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        batch: Int[Tensor, "batch pos"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["input", "output", "none"] = "none",
    ) -> Tensor | OutputWithCache:
        match cache_type:
            case "input":
                logits, pre_weight_acts = self.lm.forward_with_pre_weight_acts(batch, mask_infos)
                return OutputWithCache(output=logits, cache=pre_weight_acts)
            case "output":
                logits, output_acts = self.lm.forward_with_output_acts(batch, mask_infos)
                return OutputWithCache(output=logits, cache=output_acts)
            case "none":
                return self.lm.forward(batch, mask_infos)

    def forward_with_output_acts(
        self,
        batch: Int[Tensor, "batch pos"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.lm.forward_with_output_acts(batch, mask_infos)

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool = False,
    ) -> CIOutputs:
        return self.lm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, sampling=sampling, detach_inputs=detach_inputs
        )

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Per-site `target_weight - V@U`, computed so V/U grads stay `Shard(0)` under FSDP2.

        FSDP2 shards the components' V/U (they live inside the sharded transformer blocks),
        so accessed outside a forward they are `Shard(0)` DTensors; the frozen `target_weight`
        is a replicated buffer (plain tensor) or — under `shard_frozen_target` — a sharded
        DTensor. We `redistribute` V and U to `Replicate` (NOT `full_tensor()`/`.to_local()`),
        einsum the replicated copies, and subtract a `Replicate` target. Keeping everything a
        DTensor means the faithfulness backward redistributes the grad back to V/U's native
        `Shard(0)` — matching the recon forward's grad placement, so FSDP2's reduce-scatter
        sees consistent grads when both paths accumulate into the same param. (Naive
        `target - einsum(V,U)` on the raw `Shard(0)` params instead reshards internally and
        yields `Shard(1)`/`Replicate` grads, which collides with the recon path.) Must be
        called with V/U in their sharded (DTensor) state — i.e. before a forward gathers them.
        """
        deltas: dict[str, Tensor] = {}
        for path in self.lm.target_module_paths:
            comps = self.lm.components[path]
            weight = einops.einsum(
                _replicate(comps.V), _replicate(comps.U), "d_in C, C d_out -> d_out d_in"
            )
            target = self.lm.target_weight(path)
            if isinstance(weight, DTensor) and not isinstance(target, DTensor):
                target = distribute_tensor(target, weight.device_mesh, [Replicate()])
            elif isinstance(target, DTensor):
                target = target.redistribute(placements=[Replicate()])
            delta = target - weight
            # `.to_local()` the (Replicate) delta so downstream consumers (the faithfulness
            # accumulator, the delta-component recon path, ctx invariants) see a PLAIN tensor,
            # as they did before FSDP. Backward is unaffected: the grad still flows through the
            # `_replicate`'d V/U inputs, landing in their native Shard(0) — value plain, grad
            # sharded. (Only replicating the *inputs* fixes the placement; this final to_local
            # is value-only, unlike the old full_tensor() on the raw-input einsum.)
            deltas[path] = delta.to_local() if isinstance(delta, DTensor) else delta
        return deltas

    @contextmanager
    def use_cached_residual(self, batch: Int[Tensor, "batch pos"]) -> Iterator[None]:
        with self.lm.use_cached_residual(batch):
            yield

    @property
    def module_to_c(self) -> dict[str, int]:
        return self.lm.module_to_c

    @property
    def target_module_paths(self) -> list[str]:
        return self.lm.target_module_paths

    @property
    def components(self) -> dict[str, Components]:
        return self.lm.components

    @property
    def ci_fn(self) -> GlobalCiFnWrapper | LayerwiseCiFnWrapper | None:
        return self.lm.ci_fn

    @property
    def model(self) -> ComponentTarget:
        return self.lm.model
