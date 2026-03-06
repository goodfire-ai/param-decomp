"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Three metrics are accumulated:
- attr:         E[∂y/∂x · x]           (signed mean attribution)
- attr_abs:     E[∂|y|/∂x · x]         (attribution to absolute value of target)

Output (pseudo-) component attributions are handled differently: We accumulate attributions
to the output residual stream, then later project this into token space.

All layer keys are concrete module paths (e.g. "wte", "h.0.attn.q_proj", "lm_head").
Translation to canonical names happens at the storage boundary in harvest.py.
"""

from typing import Any

import torch
from jaxtyping import Bool, Int
from torch import Tensor, nn

from spd.configs import SamplingType
from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.topology import TransformerTopology
from spd.utils.general_utils import bf16_autocast


class AttributionHarvester:
    """Accumulates attribution strengths across batches using concrete module paths.

    The attribution formula is:
        attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]

    Key optimizations:
    1. Sum outputs over positions BEFORE computing gradients, reducing backward
       passes from O(positions × components) to O(components).
    2. For output targets, store attributions to the pre-unembed residual
       (d_model dimensions) instead of vocab tokens. This eliminates the expensive
       O((V+C) × d_model × V) matmul during harvesting and reduces storage.
    """

    sampling: SamplingType

    def __init__(
        self,
        model: ComponentModel,
        topology: TransformerTopology,
        sources_by_target: dict[str, list[str]],
        component_alive: dict[str, Bool[Tensor, " n_components"]],
        sampling: SamplingType,
    ):
        self.model = model
        self.topology = topology
        self.sources_by_target = sources_by_target
        self.component_alive = component_alive
        self.sampling = sampling
        self.embed_path = topology.path_schema.embedding_path
        self.embedding_module = topology.embedding_module
        self.unembed_path = topology.path_schema.unembed_path
        self.unembed_module = topology.unembed_module
        self.output_d_model = self.unembed_module.in_features
        self.device = next(model.parameters()).device

        # attribution accumulators
        self._straight_through_attr_acc = torch.zeros(
            (self.output_d_model, self.embedding_module.num_embeddings), device=self.device
        )
        self._embed_tgts_acc = self._get_embed_targets_attr_accumulator(sources_by_target)
        self._embed_tgts_acc_abs = self._get_embed_targets_attr_accumulator(sources_by_target)
        self._unembed_srcs_acc = self._get_unembed_sources_attr_accumulator(sources_by_target)
        self._regular_layers_acc = self._get_regular_layer_attr_accumulator(sources_by_target)
        self._regular_layers_acc_abs = self._get_regular_layer_attr_accumulator(sources_by_target)

        # embed token occurrence counts for normalization (analogous to ci_sum for components)
        self._embed_token_count = torch.zeros(
            (self.embedding_module.num_embeddings,), dtype=torch.long, device=self.device
        )

        # rms normalization accumulators
        self.n_tokens = 0
        self._ci_sum_accumulator = {
            layer: torch.zeros((c,), device=self.device)
            for layer, c in self.model.module_to_c.items()
        }
        self._square_component_act_accumulator = {
            layer: torch.zeros((c,), device=self.device)
            for layer, c in self.model.module_to_c.items()
        }
        self._logit_sq_sum = torch.zeros((self.unembed_module.out_features,), device=self.device)

    def _get_embed_targets_attr_accumulator(
        self, sources_by_target: dict[str, list[str]]
    ) -> dict[str, Tensor]:
        # extract targets who's sources include the embedding
        embed_targets_attr_accumulators: dict[str, Tensor] = {}
        for target, sources in sources_by_target.items():
            if target == self.unembed_path:
                # ignore straight-through edge
                continue
            if self.embed_path in sources:
                embed_targets_attr_accumulators[target] = torch.zeros(
                    (self.model.module_to_c[target], self.embedding_module.num_embeddings),
                    device=self.device,
                )
        return embed_targets_attr_accumulators

    def _get_unembed_sources_attr_accumulator(
        self, sources_by_target: dict[str, list[str]]
    ) -> dict[str, Tensor]:
        # extract the unembed's sources
        unembed_sources_attr_accumulators: dict[str, Tensor] = {}
        for source in sources_by_target[self.unembed_path]:
            if source == self.embed_path:
                # ignore straight-through edge
                continue
            unembed_sources_attr_accumulators[source] = torch.zeros(
                (self.output_d_model, self.model.module_to_c[source]), device=self.device
            )
        return unembed_sources_attr_accumulators

    def _get_regular_layer_attr_accumulator(
        self, sources_by_target: dict[str, list[str]]
    ) -> dict[str, dict[str, Tensor]]:
        regular_layers_shapes: dict[str, dict[str, Tensor]] = {}
        for target, sources in sources_by_target.items():
            if target == self.unembed_path:
                continue
            regular_layers_shapes[target] = {}
            for source in sources:
                if source == self.embed_path:
                    continue
                regular_layers_shapes[target][source] = torch.zeros(
                    (self.model.module_to_c[target], self.model.module_to_c[source]),
                    device=self.device,
                )
        return regular_layers_shapes

    def process_batch(self, tokens: Int[Tensor, "batch seq"]) -> None:
        """Accumulate attributions from one batch."""
        self.n_tokens += tokens.numel()
        self._embed_token_count.add_(
            torch.bincount(tokens.flatten(), minlength=self.embedding_module.num_embeddings)
        )

        # Setup hooks to capture embedding output and pre-unembed residual
        embed_out: list[Tensor] = []
        pre_unembed: list[Tensor] = []

        def embed_hook(_mod: nn.Module, _args: Any, _kwargs: Any, out: Tensor) -> Tensor:
            out.requires_grad_(True)
            embed_out.clear()
            embed_out.append(out)
            return out

        def pre_unembed_hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
            args[0].requires_grad_(True)
            pre_unembed.clear()
            pre_unembed.append(args[0])

        h1 = self.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)
        h2 = self.unembed_module.register_forward_pre_hook(pre_unembed_hook, with_kwargs=True)

        # Get masks with all components active
        with torch.no_grad(), bf16_autocast():
            out = self.model(tokens, cache_type="input")
            ci = self.model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling=self.sampling, detach_inputs=False
            )

        mask_infos = make_mask_infos(
            component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
            routing_masks="all",
        )

        # Forward pass with gradients
        with torch.enable_grad(), bf16_autocast():
            model_output: OutputWithCache = self.model(
                tokens, mask_infos=mask_infos, cache_type="component_acts"
            )

        h1.remove()
        h2.remove()

        cache = model_output.cache
        cache[f"{self.embed_path}_post_detach"] = embed_out[0]
        cache[f"{self.unembed_path}_pre_detach"] = pre_unembed[0]

        with torch.no_grad():
            for real_layer, ci_vals in ci.lower_leaky.items():
                self._ci_sum_accumulator[real_layer].add_(ci_vals.sum(dim=(0, 1)))
            self._logit_sq_sum.add_(model_output.output.detach().square().sum(dim=(0, 1)))

        for target_layer in self.sources_by_target:
            if target_layer == self.unembed_path:
                self._process_output_targets(cache, tokens, ci.lower_leaky)
            else:
                with torch.no_grad():
                    sum_sq_acts = cache[f"{target_layer}_post_detach"].square().sum(dim=(0, 1))
                    self._square_component_act_accumulator[target_layer].add_(sum_sq_acts)
                self._process_component_targets(cache, tokens, ci.lower_leaky, target_layer)

    def _process_output_targets(
        self,
        cache: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
        ci: dict[str, Tensor],
    ) -> None:
        """Process output attributions via output-residual-space storage."""
        out_residual = cache[f"{self.unembed_path}_pre_detach"]

        out_residual_sum = out_residual.sum(dim=(0, 1))

        source_layers = self.sources_by_target[self.unembed_path]
        assert self.embed_path in source_layers

        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(self.output_d_model):
            grads = torch.autograd.grad(out_residual_sum[d_idx], source_acts, retain_graph=True)
            with torch.no_grad():
                for source_layer, act, grad in zip(source_layers, source_acts, grads, strict=True):
                    if source_layer == self.embed_path:
                        token_attr = (grad * act).sum(dim=-1)  # (B S)
                        self._straight_through_attr_acc[d_idx].scatter_add_(
                            0, tokens.flatten(), token_attr.flatten()
                        )
                    else:
                        ci_weighted_attr = (grad * act * ci[source_layer]).sum(dim=(0, 1))
                        self._unembed_srcs_acc[source_layer][d_idx].add_(ci_weighted_attr)

    def _process_component_targets(
        self,
        cache: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
        ci: dict[str, Tensor],
        target_layer: str,
    ) -> None:
        """Process attributions to a component layer."""
        alive_targets = self.component_alive[target_layer]
        if not alive_targets.any():
            return

        target_acts_raw = cache[f"{target_layer}_pre_detach"]

        target_acts = target_acts_raw.sum(dim=(0, 1))
        # abs() before sum — needs its own backward pass because each element has a different
        # sign, so sign·grad can't be factored out of the sum. (In the app backend's per-prompt
        # computation the target is a single scalar, so sign·grad works as an analytical shortcut
        # and avoids a second backward. See app/backend/compute.py::_compute_edges_for_target.)
        target_acts_abs = target_acts_raw.abs().sum(dim=(0, 1))

        source_layers = self.sources_by_target[target_layer]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        def _accumulate_grads(
            grads: tuple[Tensor, ...],
            t_idx: int,
            embed_acc: dict[str, Tensor],
            regular_acc: dict[str, dict[str, Tensor]],
        ) -> None:
            with torch.no_grad():
                for source_layer, act, grad in zip(source_layers, source_acts, grads, strict=True):
                    if source_layer == self.embed_path:
                        token_attr = (grad * act).sum(dim=-1)  # (B S)
                        embed_acc[target_layer][t_idx].scatter_add_(
                            0, tokens.flatten(), token_attr.flatten()
                        )
                    else:
                        ci_weighted = (grad * act * ci[source_layer]).sum(dim=(0, 1))  # (C,)
                        regular_acc[target_layer][source_layer][t_idx].add_(ci_weighted)

        for t_idx in torch.where(alive_targets)[0].tolist():
            grads = torch.autograd.grad(target_acts[t_idx], source_acts, retain_graph=True)
            _accumulate_grads(
                grads=grads,
                t_idx=t_idx,
                embed_acc=self._embed_tgts_acc,
                regular_acc=self._regular_layers_acc,
            )

            grads_abs = torch.autograd.grad(target_acts_abs[t_idx], source_acts, retain_graph=True)
            _accumulate_grads(
                grads=grads_abs,
                t_idx=t_idx,
                embed_acc=self._embed_tgts_acc_abs,
                regular_acc=self._regular_layers_acc_abs,
            )

    def finalize(self, ci_threshold: float) -> DatasetAttributionStorage:
        """Package raw accumulators into storage. No normalization — that happens at query time."""
        assert self.n_tokens > 0, "No batches processed"

        to_canon = self.topology.target_to_canon

        def _canon_nested(acc: dict[str, dict[str, Tensor]]) -> dict[str, dict[str, Tensor]]:
            return {
                to_canon(t): {to_canon(s): v for s, v in srcs.items()} for t, srcs in acc.items()
            }

        def _canon(acc: dict[str, Tensor]) -> dict[str, Tensor]:
            return {to_canon(k): v for k, v in acc.items()}

        return DatasetAttributionStorage(
            regular_attr=_canon_nested(self._regular_layers_acc),
            regular_attr_abs=_canon_nested(self._regular_layers_acc_abs),
            embed_attr=_canon(self._embed_tgts_acc),
            embed_attr_abs=_canon(self._embed_tgts_acc_abs),
            unembed_attr=_canon(self._unembed_srcs_acc),
            embed_unembed_attr=self._straight_through_attr_acc,
            w_unembed=self.topology.get_unembed_weight(),
            ci_sum=_canon(self._ci_sum_accumulator),
            component_act_sq_sum=_canon(self._square_component_act_accumulator),
            logit_sq_sum=self._logit_sq_sum,
            embed_token_count=self._embed_token_count,
            ci_threshold=ci_threshold,
            n_tokens_processed=self.n_tokens,
        )
