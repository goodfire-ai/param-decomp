"""The 3-pool CI step's `_sparsity_losses` must match a single-process reference.

`step_ci._sparsity_losses` builds one set of per-component `(ci + eps)^p` sums and
finalizes both the imp (bare mean) and freq (`Σ f_c log2(1 + a' f_c)`) terms from it.
With `n_ci_pool=1` the cross-pool all-reduce is skipped, so the result must equal the
core `finalize_imp_min` / `finalize_freq_min` reference computed on the same sums.
"""

from dataclasses import dataclass

import torch

from param_decomp.metrics.importance_minimality import (
    finalize_freq_min,
    finalize_imp_min,
    per_component_lp_sums,
)
from param_decomp_lab.three_pool.step_ci import _sparsity_losses


@dataclass
class _StubRuntime:
    imp_min_pnorm: float
    imp_min_eps: float
    imp_min_p_anneal_start_frac: float
    imp_min_p_anneal_final_p: float | None
    imp_min_p_anneal_end_frac: float
    freq_min_reference_token_count: int


def test_sparsity_losses_match_single_process_reference() -> None:
    torch.manual_seed(0)
    ci_upper = {
        "site_a": torch.rand(2, 5, 7),  # [B, S, C]
        "site_b": torch.rand(2, 5, 3),
    }
    pnorm, eps, a_prime = 2.0, 1e-12, 4096
    cfg = _StubRuntime(
        imp_min_pnorm=pnorm,
        imp_min_eps=eps,
        imp_min_p_anneal_start_frac=1.0,
        imp_min_p_anneal_final_p=None,
        imp_min_p_anneal_end_frac=1.0,
        freq_min_reference_token_count=a_prime,
    )

    out = _sparsity_losses(
        ci_upper,
        current_frac_of_training=0.0,
        cfg=cfg,  # pyright: ignore[reportArgumentType]
        ci_pool_group=None,  # pyright: ignore[reportArgumentType]  # unused at n_ci_pool=1
        n_ci_pool=1,
    )

    sums, n = per_component_lp_sums(ci_upper_leaky=ci_upper, pnorm=pnorm, eps=eps)
    ref_imp = finalize_imp_min(per_component_sums=sums, n_examples=n)
    ref_freq = finalize_freq_min(
        per_component_sums=sums, n_examples=n, reference_token_count=a_prime
    )

    assert torch.allclose(out.imp, ref_imp)
    assert torch.allclose(out.freq, ref_freq)
