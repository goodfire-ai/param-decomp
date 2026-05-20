from collections import defaultdict
from fnmatch import fnmatch
from typing import override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.configs import SamplingType
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.registry import register_metric
from param_decomp.models.component_model import ComponentModel
from param_decomp.models.components import ComponentsMaskInfo, make_mask_infos
from param_decomp.routing import AllLayersRouter
from param_decomp.utils.component_utils import calc_stochastic_component_mask_info
from param_decomp.utils.distributed_utils import all_reduce
from param_decomp.utils.general_utils import calc_kl_divergence_lm


class StochasticReconSubsetCEAndKLConfig(LossMetricConfig):
    include_patterns: dict[str, list[str]] | None
    exclude_patterns: dict[str, list[str]] | None


@register_metric
class StochasticReconSubsetCEAndKL(Metric[StochasticReconSubsetCEAndKLConfig]):
    """Compute reconstruction loss for specific subsets of components."""

    section = "subset_worst"
    config_type = StochasticReconSubsetCEAndKLConfig
    short_name = "StochReconSubCEKL"

    def __init__(
        self,
        cfg: StochasticReconSubsetCEAndKLConfig,
        *,
        model: ComponentModel,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        include_patterns = cfg.include_patterns or {}
        exclude_patterns = cfg.exclude_patterns or {}
        if not include_patterns and not exclude_patterns:
            raise ValueError(
                "At least one of include_patterns or exclude_patterns must be provided"
            )

        all_modules: list[str] = model.target_module_paths
        self.subset_modules: dict[str, list[str]] = {}
        for subset_name, patterns in include_patterns.items():
            matched = [m for m in all_modules if any(fnmatch(m, p) for p in patterns)]
            if not matched:
                raise ValueError(
                    f"Include subset '{subset_name}' with patterns {patterns} matched no modules. "
                    f"Available modules: {all_modules}"
                )
            self.subset_modules[subset_name] = matched
        for subset_name, patterns in exclude_patterns.items():
            remaining = [m for m in all_modules if not any(fnmatch(m, p) for p in patterns)]
            if not remaining:
                raise ValueError(
                    f"Exclude subset '{subset_name}' with patterns {patterns} excluded all modules. "
                    f"Available modules: {all_modules}"
                )
            self.subset_modules[subset_name] = remaining
        self.reset()

    @override
    def reset(self) -> None:
        self.metric_values: defaultdict[str, list[float]] = defaultdict(list)

    @override
    def update(self, ctx: MetricContext) -> None:
        losses = self._calc_subset_losses(
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=ctx.weight_deltas,
            sampling=ctx.config.sampling,
            use_delta_component=ctx.config.use_delta_component,
            n_mask_samples=ctx.config.n_mask_samples,
        )
        for key, value in losses.items():
            self.metric_values[key].append(value)
        return None

    @override
    def compute(self) -> MetricResult:
        results: MetricResult = {}
        for key, vals in self.metric_values.items():
            local_sum = torch.tensor(sum(vals), device=self.device)
            local_count = torch.tensor(len(vals), device=self.device)
            global_sum = all_reduce(local_sum, op=ReduceOp.SUM).item()
            global_count = all_reduce(local_count, op=ReduceOp.SUM).item()
            results[key] = global_sum / global_count
        for metric_type in ["kl", "ce", "ce_unrec"]:
            results_by_type = {
                k: v for k, v in results.items() if isinstance(v, float) and k.endswith(metric_type)
            }
            worst_subset = max(results_by_type, key=lambda k: results_by_type[k])
            worst_value = results_by_type[worst_subset]
            results[f"{metric_type}"] = worst_value
            results[f"{metric_type}_subset"] = worst_subset
        return results

    def _calc_subset_losses(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
    ) -> dict[str, float]:
        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"
        masked_batch = batch.clone()
        masked_batch[:, 0] = -100
        flat_masked_batch = masked_batch.flatten()

        def ce_vs_labels(logits: Tensor) -> float:
            flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
            return F.cross_entropy(
                flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100
            ).item()

        def kl_vs_target(logits: Tensor) -> float:
            return calc_kl_divergence_lm(pred=logits, target=target_out).item()

        target_ce = ce_vs_labels(target_out)
        zero_mask_infos = make_mask_infos({k: torch.zeros_like(v) for k, v in ci.items()})
        zero_out = self.model(batch, mask_infos=zero_mask_infos)
        zero_ce = ce_vs_labels(zero_out)

        masks_list: list[dict[str, ComponentsMaskInfo]] = [
            calc_stochastic_component_mask_info(
                causal_importances=ci,
                component_mask_sampling=sampling,
                weight_deltas=weight_deltas if use_delta_component else None,
                router=AllLayersRouter(),
            )
            for _ in range(n_mask_samples)
        ]
        results: dict[str, float] = {}
        for subset_name, active_modules in self.subset_modules.items():
            outputs: list[Float[Tensor, "... vocab"]] = []
            for layers_masks in masks_list:
                mask_infos: dict[str, ComponentsMaskInfo] = {
                    module: layers_masks[module] for module in active_modules
                }
                outputs.append(self.model(batch, mask_infos=mask_infos))
            kl_losses = [kl_vs_target(out) for out in outputs]
            ce_losses = [ce_vs_labels(out) for out in outputs]
            mean_kl = sum(kl_losses) / len(kl_losses)
            mean_ce = sum(ce_losses) / len(ce_losses)
            ce_unrec = (mean_ce - target_ce) / (zero_ce - target_ce) if zero_ce != target_ce else 0
            results[f"{subset_name}_kl"] = mean_kl
            results[f"{subset_name}_ce"] = mean_ce
            results[f"{subset_name}_ce_unrec"] = ce_unrec
        return results
