"""`ChunkwiseSubsetReconLoss` — the flat (single-pool) twin of the 3-pool / 2-pool
chunkwise subset recon.

Standardises the flat FSDP trainer's stochastic recon on the SAME recon as the torch
2-pool's chunkwise pool, so a flat-vs-2-pool throughput comparison is watertight. It
reuses the exact 2-pool building blocks:

  - `SubsetReconPlan.generate` (`three_pool/recon_plan.py`) — one forward per generated
    routing, over all of a chunk's sites, with a freshly-drawn per-position routing.
  - `recon_one_forward` (`three_pool/step_chunkwise.py`) — the per-forward masked-suffix
    forward + recon body (identical mask sampling, identical RNG consumption).
  - `ReconLossStrategy` (`three_pool/recon_loss_strategy.py`) — the fused-linear-KL /
    LM-head-bypass pairing.

Where the 3-pool distributes the chunks one-per-rank and streams a per-forward backward
(to bound peak memory across pools), the flat path runs ALL chunks serially on every
rank and returns a single loss for the trainer's one `total_loss.backward()`. At
`chunk_dp = n_ci = 1` the 3-pool's per-forward `stoch_grad_denom` collapses to the
textbook `sum_loss / n_examples` (`n_examples = n_forwards * n_positions`) — proven on
real grads in `tests/test_three_pool_recon_plan.py` — which is exactly the value this
metric returns. So the gradient this metric feeds into `coeff * loss` matches the
2-pool's per-step recon V/U + CI grad.
"""

from typing import cast, override

import torch
from torch import Tensor
from torch.distributed import ReduceOp
from torch.utils.checkpoint import checkpoint

from param_decomp.component_model import ComponentModelProtocol
from param_decomp.distributed import all_reduce
from param_decomp.masks import RoutingMasks
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.chunkwise_subset_recon import ChunkwiseSubsetReconLossConfig
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.recon_plan import SubsetReconPlan
from param_decomp_lab.three_pool.step_chunkwise import (
    WeightDeltasFn,
    make_weight_deltas_fn,
    recon_masked_forward,
)


def _as_lm_component_model(model: ComponentModelProtocol) -> LMComponentModel:
    """The underlying vendored `LMComponentModel` (the adapter wraps it as `.lm`)."""
    if isinstance(model, LMComponentModel):
        return model
    lm = getattr(model, "lm", None)
    assert isinstance(lm, LMComponentModel), (
        f"ChunkwiseSubsetReconLoss requires a vendored LMComponentModel (the fused-KL "
        f"LM-head bypass), got {type(model).__name__}"
    )
    return lm


def chunk_sites(ordered_sites: list[str], sites_per_chunk: int) -> list[tuple[str, ...]]:
    """Group the ordered decomposed sites into sequential chunks of `sites_per_chunk`.

    Mirrors `ThreePoolTopology.resolve`'s split, so the flat path's chunk grouping is
    identical to the 2-pool's for the same `sites_per_chunk` and site ordering.
    """
    return [
        tuple(ordered_sites[i : i + sites_per_chunk])
        for i in range(0, len(ordered_sites), sites_per_chunk)
    ]


def chunkwise_subset_recon(
    lm: LMComponentModel,
    batch: object,
    target_local: Tensor,
    ci: dict[str, Tensor],
    chunks: list[tuple[str, ...]],
    plan: SubsetReconPlan,
    strategy: ReconLossStrategy,
    weight_deltas_fn: WeightDeltasFn,
    device: torch.device,
) -> tuple[Tensor, int]:
    """The flat chunkwise subset recon: mean over all chunk forwards of `loss / n_pos`.

    For each chunk, `plan.generate` produces the recon forwards (same routing draws as
    the 2-pool); each runs through the SAME `recon_one_forward` body the 2-pool's
    `_run_routing_forwards` uses, with the SAME per-forward fresh-delta fn. Returns
    `(loss, n_forwards)` where `loss` is `Σ_forwards (loss_f / n_positions) / n_forwards`
    — the textbook `sum / n_examples` the 2-pool's per-forward `stoch_grad_denom`
    collapses to at `chunk_dp = n_ci = 1`. Builds the full graph forward-only and returns
    ONE loss; the caller multiplies by `coeff` and runs ONE backward (vs the 2-pool's
    per-forward backward, which seeds the identical V/U + CI grad — each forward's fresh
    delta subgraph makes the two normalisations coincide bit-for-bit)."""
    sum_over_forwards = torch.zeros((), device=device)
    n_forwards = 0
    for chunk in chunks:
        mask_shape = ci[chunk[0]].shape[:-1]
        routings = plan.generate(chunk, mask_shape, device)
        for sites, routing in routings:
            # Compute the fresh per-forward deltas EAGER, OUTSIDE the checkpoint, so the
            # backward recompute restores a saved plain tensor rather than re-running the
            # DTensor `redistribute`/`to_local` (which, during the FSDP backward, re-derives
            # the delta as a `Shard(0)` DTensor and collides with the plain activation inside
            # the compiled masked forward). Grad still reaches V/U through the eager delta
            # subgraph; the delta draws no RNG, so the per-site `u`/`delta_mask` draws inside
            # the recomputed region replay identically (equivalence preserved bit-for-bit).
            weight_deltas = weight_deltas_fn(sites)
            loss_f, n_positions = _checkpointed_recon_one_forward(
                lm, batch, target_local, ci, sites, routing, strategy, weight_deltas
            )
            assert loss_f.dim() == 0, f"recon loss for sites {sites!r} must be scalar"
            sum_over_forwards = sum_over_forwards + loss_f / n_positions
            n_forwards += 1
    assert n_forwards > 0, "no recon forwards generated — empty decomposition?"
    return sum_over_forwards / n_forwards, n_forwards


def _checkpointed_recon_one_forward(
    lm: LMComponentModel,
    batch: object,
    target_local: Tensor,
    ci: dict[str, Tensor],
    sites: tuple[str, ...],
    routing: RoutingMasks,
    strategy: ReconLossStrategy,
    weight_deltas: dict[str, Tensor],
) -> tuple[Tensor, int]:
    """`recon_one_forward` with its activations recomputed in backward, not retained.

    The flat path accumulates every chunk's recon graph before the trainer's single
    `total_loss.backward()` (the `Metric` contract: `update` returns a loss). Without
    checkpointing that holds all forwards' activations at once — the 3-pool instead
    streams a per-forward backward and frees each graph, holding one forward's
    activations at a time. `checkpoint(use_reentrant=False)` matches that profile: each
    forward keeps only its inputs, recomputing activations in backward. The recompute is
    numerically exact — `use_reentrant=False` saves and restores the RNG, so the
    `torch.rand_like` / `torch.rand` mask draws inside `recon_one_forward` replay
    identically (and the saved/restored RNG leaves the global stream untouched, so the
    forward-pass draw order matches the non-checkpointed path bit-for-bit).

    The CI leaves AND the precomputed weight-deltas are passed positionally so
    non-reentrant checkpoint registers them as backward inputs: the CI leaves' `.grad`
    feeds the CI grad, and the deltas' grad redistributes back to V/U through the eager
    `target − VU` subgraph the caller built OUTSIDE this region. Deltas are precomputed
    (not recomputed via a closure) so the FSDP backward recompute never re-derives them
    as `Shard(0)` DTensors — which would collide with the plain activation in the compiled
    masked forward (`got mixed torch.Tensor and DTensor`).

    `strategy.context()` (the fused-KL LM-head bypass) is entered INSIDE the checkpointed
    region, not around it, so the bypass is active on BOTH the forward pass and the backward
    recompute. The recompute runs during `backward()` — after any outer bypass context has
    exited — so without re-entering it here the recompute would run the full LM head and the
    recomputed forward output would be the vocab projection `[d_model, vocab]` instead of the
    saved pre-LM-head hidden state `[pos, d_model]` (`Recomputed values ... different
    metadata`)."""
    ci_sites = tuple(ci[s] for s in sites)
    delta_sites = tuple(weight_deltas[s] for s in sites)
    n_ci = len(ci_sites)

    def run(*tensors: Tensor) -> tuple[Tensor, int]:
        ci_local = dict(zip(sites, tensors[:n_ci], strict=True))
        deltas_local = dict(zip(sites, tensors[n_ci:], strict=True))
        with strategy.context():
            return recon_one_forward(
                lm, batch, target_local, ci_local, sites, routing, strategy, lambda _s: deltas_local
            )

    return cast("tuple[Tensor, int]", checkpoint(run, *ci_sites, *delta_sites, use_reentrant=False))


class ChunkwiseSubsetReconLoss(Metric[ChunkwiseSubsetReconLossConfig]):
    log_namespace = "loss"
    short_name = "ChunkSubsetRecon"

    @override
    def bind(self, *, model: ComponentModelProtocol, device: str) -> None:
        super().bind(model=model, device=device)
        self._lm = _as_lm_component_model(model)
        # Deltas flow through the bound model (the FSDP adapter), whose `calc_weight_deltas`
        # `.to_local()`s the `target − VU` to a PLAIN tensor (grad still redistributes to V/U's
        # native Shard(0)). The raw `self._lm.calc_weight_deltas` returns a `Shard(0)` DTensor,
        # which the checkpoint recompute restores as a DTensor and then mixes with the plain
        # activation in the compiled masked forward (`got mixed torch.Tensor and DTensor`).
        self._delta_model = model
        self._chunks = chunk_sites(self._lm.target_module_paths, self.cfg.sites_per_chunk)
        self._plan = SubsetReconPlan(routing=self.cfg.routing, n_samples=self.cfg.n_samples)
        self._strategy = ReconLossStrategy.from_cfg(
            self._lm, use_fused_kl=self.cfg.use_fused_kl, unfused_recon=recon_loss_kl
        )

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_forwards = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        assert ctx.use_delta_component, (
            "ChunkwiseSubsetReconLoss mirrors the chunkwise pool, which always uses the "
            "weight-delta component; set pd.use_delta_component=true"
        )
        ci = ctx.ci.lower_leaky
        device = torch.device(self.device)

        # Clean target under the SAME strategy context as the recon forwards: with
        # fused-KL this is the pre-LM-head hidden state (bypass), matching what
        # `recon_one_forward`'s masked forward returns. `ctx.target_out` is full logits
        # (the unbypassed metric-context forward), so it can't serve as the fused target.
        with self._strategy.context(), torch.no_grad():
            target_local = self._lm(ctx.batch).detach()

        loss, _ = chunkwise_subset_recon(
            lm=self._lm,
            batch=ctx.batch,
            target_local=target_local,
            ci=ci,
            chunks=self._chunks,
            plan=self._plan,
            strategy=self._strategy,
            weight_deltas_fn=make_weight_deltas_fn(self._delta_model),
            device=device,
        )
        self.sum_loss += loss.detach()
        self.n_forwards += 1
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_forwards = all_reduce(self.n_forwards, op=ReduceOp.SUM)
        return sum_loss / n_forwards
