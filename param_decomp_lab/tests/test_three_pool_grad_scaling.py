"""Regression test for PPGD gradient scaling under the SUM-grad convention.

Under the SUM-grad convention (``three_pool/SUM_GRAD_CONVENTION.md``) every
data-parallel gradient reduction is SUM, so a producer's grad is a partial sum
normalized only by the honest global count — it carries NO pool-size transport
factor. For PPGD this means V/U and CI now share ONE scale
``coeff_ppgd / n_examples_global``: the old ``* n_ci`` on the CI grad (which
compensated for the CI-pool AVG-reduce, PR #545) is gone, because the CI-pool
reduce is now a SUM.

Sources stay per-rank-local (``1 / n_examples_local``).
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


def test_ppgd_vu_and_ci_share_one_scale() -> None:
    n_ci, n_ppgd, n_examples_local, coeff = 4, 2, 8, 3.0
    raw = RawGrads(v=_ones(), u=_ones(), ci=_ones(), sources=_ones())

    _scale_grads(
        raw, n_examples_local, _fake_ctx(n_ci=n_ci, n_ppgd=n_ppgd), _fake_cfg(coeff_ppgd=coeff)
    )

    # V/U and CI are both partial sums under the SUM convention: one scale.
    shared_scale = coeff / (n_examples_local * n_ppgd)
    torch.testing.assert_close(raw.v["site"], torch.full((2, 2), shared_scale))
    torch.testing.assert_close(raw.u["site"], torch.full((2, 2), shared_scale))
    torch.testing.assert_close(raw.ci["site"], torch.full((2, 2), shared_scale))
    # Sources: 1 / n_examples_local only — no coeff, no 1/n_ppgd, no n_ci.
    torch.testing.assert_close(raw.sources["site"], torch.full((2, 2), 1.0 / n_examples_local))


def test_ppgd_ci_scale_is_independent_of_n_ci() -> None:
    """The defining property of the SUM convention: the CI grad scale does not
    depend on ``n_ci`` (the old patch multiplied it by ``n_ci``)."""
    n_ppgd, n_examples_local, coeff = 2, 8, 3.0
    scales: list[float] = []
    for n_ci in (1, 4, 16):
        raw = RawGrads(v=_ones(), u=_ones(), ci=_ones(), sources=_ones())
        _scale_grads(
            raw, n_examples_local, _fake_ctx(n_ci=n_ci, n_ppgd=n_ppgd), _fake_cfg(coeff_ppgd=coeff)
        )
        scales.append(raw.ci["site"][0, 0].item())
    assert scales[0] == scales[1] == scales[2], (
        f"CI scale must be independent of n_ci under SUM convention; got {scales}"
    )
