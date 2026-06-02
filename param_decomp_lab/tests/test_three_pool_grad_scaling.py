"""Regression test for PPGD -> CI gradient scaling under a multi-rank CI pool.

The CI pool ends each step with an AVG all-reduce over its ``n_ci`` ranks
(``all_reduce_ci_fn_grads``). PPGD's CI grad is injected per-position on a single
CI rank, so that AVG divides it by ``n_ci``; ``_scale_grads`` must pre-multiply
the CI grad by ``n_ci`` to compensate (the LW stoch path does the same via its
``/ n_ci`` denom). V/U never hits that AVG, so it keeps the plain scale.

At ``n_ci=1`` the factor is a no-op — which is why the bug was invisible to the
8-GPU (``n_ci=1``) configs and only bit production (``n_ci`` = 16 / 24).
"""

from types import SimpleNamespace
from typing import cast

import torch

from param_decomp_lab.three_pool.context import PPGDContext
from param_decomp_lab.three_pool.runtime import _ThreePoolRuntime
from param_decomp_lab.three_pool.step_ppgd import RawGrads, _scale_grads


def _fake_ctx(*, n_ci: int, n_ppgd: int) -> PPGDContext:
    # _scale_grads only reads ctx.world.{n_ci,n_ppgd}; cast through object.
    fake = SimpleNamespace(world=SimpleNamespace(n_ci=n_ci, n_ppgd=n_ppgd))
    return cast(PPGDContext, cast(object, fake))


def _fake_cfg(*, coeff_ppgd: float) -> _ThreePoolRuntime:
    # _scale_grads only reads cfg.coeff_ppgd; cast through object.
    return cast(_ThreePoolRuntime, cast(object, SimpleNamespace(coeff_ppgd=coeff_ppgd)))


def _ones() -> dict[str, torch.Tensor]:
    return {"site": torch.ones(2, 2)}


def test_ppgd_ci_grad_carries_extra_n_ci_factor() -> None:
    n_ci, n_ppgd, n_examples_local, coeff = 4, 2, 8, 3.0
    raw = RawGrads(v=_ones(), u=_ones(), ci=_ones(), sources=_ones())

    _scale_grads(
        raw, n_examples_local, _fake_ctx(n_ci=n_ci, n_ppgd=n_ppgd), _fake_cfg(coeff_ppgd=coeff)
    )

    vu_scale = coeff / (n_examples_local * n_ppgd)
    torch.testing.assert_close(raw.v["site"], torch.full((2, 2), vu_scale))
    torch.testing.assert_close(raw.u["site"], torch.full((2, 2), vu_scale))
    # CI must carry the extra * n_ci to survive the CI-pool AVG all-reduce.
    torch.testing.assert_close(raw.ci["site"], torch.full((2, 2), vu_scale * n_ci))
    # Sources: 1 / n_examples_local only — no coeff, no 1/n_ppgd, no n_ci.
    torch.testing.assert_close(raw.sources["site"], torch.full((2, 2), 1.0 / n_examples_local))
