"""Gradient-scaling derisk for the LW routing-plan generalisation.

The claim under test (see ``step_layerwise._run_routing_forwards``): replacing the
old ``n_sites_total`` factor with ``N_est`` (total LW recon forwards / step) keeps
the stoch gradient at the single-pool scale for ANY routing plan.

The decisive check is single-GPU (``n_per_block = n_ci = 1``), where the denom
collapses to ``n_positions * N_est = n_positions * n_forwards = n_examples`` — the
textbook single-pool normalisation ``sum_loss / n_examples``. So per-forward backward
with the new denom must equal one backward of ``sum_loss / n_examples`` over the SAME
forwards. Verified on the real ``.grad`` after one backward (RNG pinned), not on loss
curves.
"""

from typing import cast, override

import pytest
import torch
import torch.nn as nn

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import (
    AllRoutingConfig,
    StaticProbabilityRoutingConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.three_pool.loss_strategy import LayerwiseLossStrategy
from param_decomp_lab.three_pool.routing_plan import (
    PerSitePlan,
    RoutingPlan,
    SubsetRoutingPlan,
)
from param_decomp_lab.three_pool.step_layerwise import (
    _recon_one_forward,
    _run_routing_forwards,
)

_DRAW_SEED = 1234


class _TwoLayerSeq(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.l0 = nn.Linear(d, d, bias=False)
        self.l1 = nn.Linear(d, d, bias=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1(torch.relu(self.l0(x)))


def _make_model(d: int = 4, c: int = 3) -> ComponentModel:
    target = _TwoLayerSeq(d)
    target.requires_grad_(False)
    return ComponentModel(
        target_model=target,
        run_batch=run_batch_passthrough,
        decomposition_targets=[
            DecompositionTarget(module_path="l0", C=c),
            DecompositionTarget(module_path="l1", C=c),
        ],
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


def _grad(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad.clone()


@pytest.mark.parametrize(
    "plan",
    [
        PerSitePlan(),
        SubsetRoutingPlan(routing=AllRoutingConfig(), n_samples=1),
        SubsetRoutingPlan(routing=AllRoutingConfig(), n_samples=3),
        SubsetRoutingPlan(routing=UniformKSubsetRoutingConfig(), n_samples=4),
        SubsetRoutingPlan(routing=StaticProbabilityRoutingConfig(p=0.5), n_samples=2),
    ],
)
def test_denom_matches_single_pool_normalization(plan: RoutingPlan) -> None:
    """New per-forward denom (at n_per_block=n_ci=1) == textbook sum/n_examples."""
    torch.manual_seed(0)
    model = _make_model()
    owned = ("l0", "l1")
    batch = torch.randn(2, 5, 4)
    with torch.no_grad():
        target = model(batch).detach()
    mask_shape = (2, 5)
    ci_vals = {s: torch.rand(2, 5, model.module_to_c[s]) for s in owned}

    # Shared routing list so both runs see identical per-position routing.
    routings = plan.generate(owned, mask_shape, torch.device("cpu"))
    n_forwards = len(routings)
    assert n_forwards == plan.n_forwards(owned)
    strategy = LayerwiseLossStrategy.unfused(recon_loss_mse)

    # --- path under test: per-forward backward with the new denom ---
    leaves_path = {s: ci_vals[s].clone().requires_grad_(True) for s in owned}
    for p in model.parameters():
        p.grad = None
    torch.manual_seed(_DRAW_SEED)
    with strategy.context():
        _run_routing_forwards(
            component_model=cast(LMComponentModel, cast(object, model)),
            batch_local=batch,
            target_local=target,
            ci_leaves=leaves_path,
            routings=routings,
            coeff_stoch=1.0,
            n_est=n_forwards,
            n_per_block=1,
            n_ci=1,
            strategy=strategy,
            bf16_autocast_enabled=False,
        )
    vu_path = _vu_grads(model)
    leaf_path = {s: _grad(leaves_path[s]) for s in owned}

    # --- reference: same forwards, single backward of sum_loss / n_examples ---
    leaves_ref = {s: ci_vals[s].clone().requires_grad_(True) for s in owned}
    for p in model.parameters():
        p.grad = None
    torch.manual_seed(_DRAW_SEED)
    with strategy.context():
        total = torch.zeros(())
        n_positions = -1
        for sites, routing in routings:
            loss_f, n_positions = _recon_one_forward(
                cast(LMComponentModel, cast(object, model)),
                batch,
                target,
                leaves_ref,
                sites,
                routing,
                strategy,
            )
            total = total + loss_f
        n_examples = n_forwards * n_positions
        (total / n_examples).backward()
    vu_ref = _vu_grads(model)
    leaf_ref = {s: _grad(leaves_ref[s]) for s in owned}

    for k in vu_path:
        torch.testing.assert_close(vu_path[k], vu_ref[k], msg=f"V/U grad mismatch: {k}")
    for s in owned:
        torch.testing.assert_close(leaf_path[s], leaf_ref[s], msg=f"CI leaf grad mismatch: {s}")


def test_per_site_plan_n_est_equals_total_sites() -> None:
    """Back-compat: with PerSitePlan, n_est == total sites across blocks, so the
    denom reduces to the old ``n_positions * n_sites_total * n_per_block / n_ci``."""
    plan = PerSitePlan()
    blocks = [("a", "b"), ("c",), ("d", "e", "f")]
    n_est = sum(plan.n_forwards(owned) for owned in blocks)
    assert n_est == sum(len(b) for b in blocks) == 6
