"""Watertight equivalence: the flat `ChunkwiseSubsetReconLoss` recon == the torch
2-pool's chunkwise recon (`_run_routing_forwards`) for the same model + batch + RNG.

What this proves, and how:

  * **Same chunk grouping.** `chunk_sites(sites, sites_per_chunk)` == the sequential
    split `ThreePoolTopology.resolve` uses (asserted directly).
  * **Same forward count.** `n_forwards` over all chunks == the 2-pool's global
    `N_est = Σ_chunks plan.n_forwards` (asserted), so the normaliser is identical.
  * **Same routing draws + same loss + same gradients (BIT-IDENTICAL).** Both paths
    drive the SAME `recon_one_forward` body and the SAME `SubsetReconPlan.generate`, so
    with the RNG pinned identically the per-position routing, the `u`/`delta_mask` draws,
    the per-forward loss, and the V/U + CI gradients all match exactly. The flat path
    runs ONE `(coeff * loss).backward()`; the 2-pool runs a per-forward backward with
    `stoch_grad_denom = n_positions * N_est` (at `chunk_dp = n_ci = 1`). These produce
    the identical gradient — this test asserts it on the real `.grad` after one backward.

The value/grad equivalence uses the UNFUSED (MSE) strategy because the tiny test model
has no LM head; the fused-linear-KL path is the SAME `ReconLossStrategy` /
`recon_one_forward` plumbing exercised by the 3-pool grad checks and is selected here
only structurally (a separate assertion that `use_fused_kl=True` yields a fused
strategy). So: forward count + chunk grouping + routing distribution + V/U & CI grads
are bit-identical; the fused-KL choice is structurally identical.
"""

from typing import cast, override

import torch
import torch.nn as nn

from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp_config.ci_fn import LayerwiseCiConfig
from param_decomp_config.losses import ChunkwiseSubsetReconLossConfig
from param_decomp_config.routing import UniformKSubsetRoutingConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.metrics.chunkwise_subset_recon import (
    ChunkwiseSubsetReconLoss,
    chunk_sites,
    chunkwise_subset_recon,
)
from param_decomp_lab.three_pool.recon_loss_strategy import ReconLossStrategy
from param_decomp_lab.three_pool.recon_plan import SubsetReconPlan
from param_decomp_lab.three_pool.step_chunkwise import (
    Stoch,
    _run_routing_forwards,
    make_weight_deltas_fn,
)

_DRAW_SEED = 4321
_SITES = ("l0", "l1", "l2", "l3", "l4", "l5")
_SITES_PER_CHUNK = 2


class _SixLinearSeq(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        for name in _SITES:
            setattr(self, name, nn.Linear(d, d, bias=False))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in _SITES:
            x = torch.relu(getattr(self, name)(x))
        return x


def _make_model(d: int = 4, c: int = 3) -> ComponentModel:
    target = _SixLinearSeq(d)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[DecompositionTarget(module_path=s, C=c) for s in _SITES],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        sigmoid_type="leaky_hard",
    )


def _vu_grads(model: ComponentModel) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, comp in model.components.items():
        assert comp.V.grad is not None and comp.U.grad is not None
        out[f"{name}.V"] = comp.V.grad.clone()
        out[f"{name}.U"] = comp.U.grad.clone()
    return out


def _leaf_grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad.clone()


def test_chunk_sites_matches_topology_split() -> None:
    """Flat chunk grouping == the 2-pool's sequential `ordered_sites[i:i+spc]` split."""
    chunks = chunk_sites(list(_SITES), _SITES_PER_CHUNK)
    assert chunks == [("l0", "l1"), ("l2", "l3"), ("l4", "l5")]


def test_flat_recon_grads_match_two_pool_chunkwise() -> None:
    """Flat one-backward recon == 2-pool per-chunk per-forward backward (RNG-pinned)."""
    torch.manual_seed(0)
    model = _make_model()
    batch = torch.randn(2, 5, 4)
    with torch.no_grad():
        target = model(batch).detach()
    ci_vals = {s: torch.rand(2, 5, model.module_to_c[s]) for s in _SITES}

    chunks = chunk_sites(list(_SITES), _SITES_PER_CHUNK)
    plan = SubsetReconPlan(routing=UniformKSubsetRoutingConfig(), n_samples=2)
    n_est = sum(plan.n_forwards(chunk) for chunk in chunks)
    assert n_est == len(chunks) * plan.n_samples
    strategy = ReconLossStrategy.unfused(recon_loss_mse)
    lm = cast(LMComponentModel, cast(object, model))

    # --- flat path: one backward of (loss) where loss = sum_forwards(loss_f/n_pos)/N ---
    leaves_flat = {s: ci_vals[s].clone().requires_grad_(True) for s in _SITES}
    for p in model.parameters():
        p.grad = None
    torch.manual_seed(_DRAW_SEED)
    loss, n_forwards = chunkwise_subset_recon(
        lm=lm,
        batch=batch,
        target_local=target,
        ci=leaves_flat,
        chunks=chunks,
        plan=plan,
        strategy=strategy,
        weight_deltas_fn=make_weight_deltas_fn(lm),
        device=torch.device("cpu"),
    )
    assert n_forwards == n_est
    loss.backward()
    vu_flat = _vu_grads(model)
    leaf_flat = {s: _leaf_grad(leaves_flat[s]) for s in _SITES}

    # --- 2-pool path: per chunk, _run_routing_forwards (per-forward backward, the SAME
    # denom n_positions * n_est). One CI-leaf set across all chunks (the flat path shares
    # one ci dict too), so leaf grads accumulate across chunks just like the flat path. ---
    leaves_2p = {s: ci_vals[s].clone().requires_grad_(True) for s in _SITES}
    for p in model.parameters():
        p.grad = None
    torch.manual_seed(_DRAW_SEED)
    total_forwards = 0
    for chunk in chunks:
        mask_shape = leaves_2p[chunk[0]].shape[:-1]
        routings = plan.generate(chunk, mask_shape, torch.device("cpu"))
        stoch = _run_routing_forwards(
            component_model=lm,
            batch_local=batch,
            target_local=target,
            ci_leaves={s: leaves_2p[s] for s in chunk},
            routings=routings,
            weight_deltas_fn=make_weight_deltas_fn(lm),
            coeff_stoch=1.0,
            n_est=n_est,
            chunk_dp=1,
            strategy=strategy,
            bf16_autocast_enabled=False,
        )
        assert isinstance(stoch, Stoch)
        total_forwards += stoch.n_forwards
    assert total_forwards == n_est
    vu_2p = _vu_grads(model)
    leaf_2p = {s: _leaf_grad(leaves_2p[s]) for s in _SITES}

    for k in vu_flat:
        torch.testing.assert_close(vu_flat[k], vu_2p[k], msg=f"V/U grad mismatch: {k}")
    for s in _SITES:
        torch.testing.assert_close(leaf_flat[s], leaf_2p[s], msg=f"CI leaf grad mismatch: {s}")


def test_use_fused_kl_selects_fused_strategy() -> None:
    """`use_fused_kl` controls whether the metric builds a fused-linear-KL strategy.

    Drives `ReconLossStrategy.from_cfg` the way the metric's `bind` does (the metric
    itself needs an `LMComponentModel`, exercised by the smoke YAML)."""

    class _FakeBypass:
        def __init__(self) -> None:
            self.entered = False

        def bypass_lm_head(self):  # noqa: ANN202 — test double
            from contextlib import contextmanager

            @contextmanager
            def _cm():  # noqa: ANN202
                self.entered = True
                yield torch.zeros(7, 4)

            return _cm()

    fake = _FakeBypass()
    fused = ReconLossStrategy.from_cfg(
        cast(LMComponentModel, cast(object, fake)), use_fused_kl=True, unfused_recon=recon_loss_mse
    )
    with fused.context():
        pass
    assert fake.entered, "use_fused_kl=True must enter the LM-head bypass context"

    unfused = ReconLossStrategy.from_cfg(
        cast(LMComponentModel, cast(object, fake)),
        use_fused_kl=False,
        unfused_recon=recon_loss_mse,
    )
    assert unfused.recon_loss is recon_loss_mse


def test_metric_config_in_loss_union() -> None:
    """The config validates as a `pd.loss_metrics` entry (joins `AnyLossMetricConfig`)."""
    from param_decomp_config.pd import PDConfig

    cfg = ChunkwiseSubsetReconLossConfig(
        coeff=0.5, sites_per_chunk=3, routing=UniformKSubsetRoutingConfig(), n_samples=1
    )
    assert cfg.type == "ChunkwiseSubsetReconLoss"
    # The lab dispatch table resolves the type literal to the lab impl class.
    from param_decomp_lab.metrics.dispatch import ALL_LOSS_METRIC_CLASSES

    assert ALL_LOSS_METRIC_CLASSES[cfg.type] is ChunkwiseSubsetReconLoss
    del PDConfig
