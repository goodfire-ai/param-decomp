from typing import ClassVar, Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce
from param_decomp.masks import (
    AllLayersRouter,
    SamplingType,
    calc_stochastic_component_mask_info,
    make_mask_infos,
)
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.batch_and_loss_fns import calc_kl_divergence_lm


class CEandKLLossesConfig(BaseConfig):
    """`rounding_threshold` binarises CI for the `*_rounded_masked` variant (`ci > threshold`)."""

    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CEandKLLosses(Metric[CEandKLLossesConfig]):
    """Cross-entropy and KL losses under six CI masking strategies.

    Each batch runs through the component model with six mask variants and is compared
    against next-token labels (CE) and the target model's logits (KL):

    - `ci_masked`: components multiplied by CI lower-leaky values.
    - `unmasked`: all components on (mask of ones).
    - `stoch_masked`: stochastic mask derived from CI via the configured sampler.
    - `random_masked`: uniform random mask in `[0, 1)`.
    - `rounded_masked`: CI binarised at `cfg.rounding_threshold`.
    - `zero_masked`: all components off — CE ceiling.

    Result keys: `kl_<variant>` (mean per-position KL vs target), `ce_difference_<variant>`
    (CE minus target's CE), `ce_unrecovered_<variant>` (fraction of CE gap left, scaled
    so target=0 and zero-masked=1). Assumes uniform batch + sequence size.
    """

    log_namespace = "ce_kl"
    short_name = "CEandKL"

    loss_keys: ClassVar[list[str]] = [
        "kl_ci_masked",
        "kl_unmasked",
        "kl_stoch_masked",
        "kl_random_masked",
        "kl_rounded_masked",
        "kl_zero_masked",
        "ce_difference_ci_masked",
        "ce_difference_unmasked",
        "ce_difference_stoch_masked",
        "ce_difference_random_masked",
        "ce_difference_rounded_masked",
        "ce_unrecovered_ci_masked",
        "ce_unrecovered_unmasked",
        "ce_unrecovered_stoch_masked",
        "ce_unrecovered_random_masked",
        "ce_unrecovered_rounded_masked",
    ]

    @override
    def reset(self) -> None:
        self.loss_sums: dict[str, Tensor] = {
            key: torch.zeros((), device=self.device) for key in self.loss_keys
        }
        self.n_positions: Int[Tensor, ""] = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> None:
        assert ctx.batch.ndim == 2, "Batch must be 2D (batch, seq_len)"
        ce_losses = self._calc_ce_and_kl_losses(
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=ctx.weight_deltas,
            sampling_type=ctx.sampling,
        )
        n_positions_in_batch = ctx.batch.shape[0] * ctx.batch.shape[1]
        for key in self.loss_keys:
            self.loss_sums[key] += ce_losses[key] * n_positions_in_batch
        self.n_positions += n_positions_in_batch
        return None

    @override
    def compute(self) -> MetricResult:
        losses = {}
        n_positions_reduced = all_reduce(self.n_positions, op=ReduceOp.SUM).item()
        for key in self.loss_keys:
            summed_loss = all_reduce(self.loss_sums[key], op=ReduceOp.SUM).item()
            losses[key] = summed_loss / n_positions_reduced
        return losses

    def _calc_ce_and_kl_losses(
        self,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Tensor],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        sampling_type: SamplingType,
    ) -> dict[str, float]:
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

        ci_mask_infos = make_mask_infos(ci)
        ci_masked_logits = self.model(batch, mask_infos=ci_mask_infos)
        ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
        ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

        mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling_type,
            router=AllLayersRouter(),
            weight_deltas=weight_deltas,
        )
        stoch_masked_logits = self.model(batch, mask_infos=mask_infos)
        stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
        stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

        nonmask_infos = make_mask_infos({k: torch.ones_like(v) for k, v in ci.items()})
        unmasked_logits = self.model(batch, mask_infos=nonmask_infos)
        unmasked_ce_loss = ce_vs_labels(unmasked_logits)
        unmasked_kl_loss = kl_vs_target(unmasked_logits)

        rand_mask_infos = make_mask_infos({k: torch.rand_like(v) for k, v in ci.items()})
        random_masked_logits = self.model(batch, mask_infos=rand_mask_infos)
        random_masked_ce_loss = ce_vs_labels(random_masked_logits)
        random_masked_kl_loss = kl_vs_target(random_masked_logits)

        rounded_mask_infos = make_mask_infos(
            {k: (v > self.cfg.rounding_threshold).float() for k, v in ci.items()}
        )
        rounded_masked_logits = self.model(batch, mask_infos=rounded_mask_infos)
        rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
        rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

        zero_mask_infos = make_mask_infos({k: torch.zeros_like(v) for k, v in ci.items()})
        zero_masked_logits = self.model(batch, mask_infos=zero_mask_infos)
        zero_masked_ce_loss = ce_vs_labels(zero_masked_logits)
        zero_masked_kl_loss = kl_vs_target(zero_masked_logits)

        target_model_ce_loss = ce_vs_labels(target_out)

        def pct_ce_unrecovered(ce: float) -> float:
            return (ce - target_model_ce_loss) / (zero_masked_ce_loss - target_model_ce_loss)

        def ce_difference(ce: float) -> float:
            return ce - target_model_ce_loss

        out: dict[str, float] = {
            "kl_ci_masked": ci_masked_kl_loss,
            "kl_unmasked": unmasked_kl_loss,
            "kl_stoch_masked": stoch_masked_kl_loss,
            "kl_random_masked": random_masked_kl_loss,
            "kl_rounded_masked": rounded_masked_kl_loss,
            "kl_zero_masked": zero_masked_kl_loss,
            "ce_difference_ci_masked": ce_difference(ci_masked_ce_loss),
            "ce_difference_unmasked": ce_difference(unmasked_ce_loss),
            "ce_difference_stoch_masked": ce_difference(stoch_masked_ce_loss),
            "ce_difference_random_masked": ce_difference(random_masked_ce_loss),
            "ce_difference_rounded_masked": ce_difference(rounded_masked_ce_loss),
            "ce_unrecovered_ci_masked": pct_ce_unrecovered(ci_masked_ce_loss),
            "ce_unrecovered_unmasked": pct_ce_unrecovered(unmasked_ce_loss),
            "ce_unrecovered_stoch_masked": pct_ce_unrecovered(stoch_masked_ce_loss),
            "ce_unrecovered_random_masked": pct_ce_unrecovered(random_masked_ce_loss),
            "ce_unrecovered_rounded_masked": pct_ce_unrecovered(rounded_masked_ce_loss),
        }
        assert list(out.keys()) == self.loss_keys
        return out
