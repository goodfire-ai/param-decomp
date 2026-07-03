"""Per-site faithfulness observability — the per-site Frobenius of the weight deltas.

The global `FaithfulnessLoss` (`losses.faithfulness_loss`) is `Σ_s ‖Δ_s‖² / Σ_s numel`
over the per-site deltas `Δ_s = W_s − V_s@U_s` — an aggregate that HIDES catastrophically
broken small-norm sites: a handful of sites can carry nearly all of the unmasked-KL damage
while the mean sits at ~1e-8 (lore `2026-07-03--unmasked-kl-vs-faith-no-eval-bug-4-culprit-sites`).
This exposes the per-site breakdown so those sites are visible at eval cadence.

Scope: relative Frobenius `‖Δ_s‖_F / ‖W_s‖_F` measures per-site WEIGHT-space error, which
is NOT the same as behavioral sensitivity — the worst rel-Frobenius sites need not be the
ones that dominate KL (in the run above they were behaviorally inert). This is pure
observability, not a culprit detector; the KL-under-delta-ablation probe is what ranks
behavioral sensitivity.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from param_decomp.components import DecompVU
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel


def make_per_site_faith_step(
    lm: DecomposedModel, compiler_options: dict[str, bool | int | str] | None = None
):
    """Build the `filter_jit`'d `per_site_faith(model, components) -> {site: ‖Δ_s‖²_F}`
    (fp32 squared Frobenius per site). Reuses `model.weight_deltas` — the SAME per-site
    `Δ_s = W_s − V_s@U_s` the global `FaithfulnessLoss` reduces — so
    `Σ_s result[s] / Σ_s numel == FaithfulnessLoss` exactly.

    `model` (the frozen-weight-bearing `DecomposedModel`) is the jit ARG — array leaves
    traced, never baked (the HLO-baking rule). Each `Δ_s` is materialized then reduced to a
    scalar, so XLA frees it after its reduction: peak memory is ~one delta, not the whole
    model. Passing a zeroed `components` yields `‖W_s‖²_F` (`Δ_s = W_s − 0`) — the constant
    denominator for relative Frobenius.
    """
    site_names = lm.site_names  # static, read off the closed-over model (HLO-baking rule)

    def per_site_faith(model: DecomposedModel, components: DecompVU) -> dict[str, Float[Array, ""]]:
        deltas = model.weight_deltas(components)
        return {site: (deltas[site].astype(jnp.float32) ** 2).sum() for site in site_names}

    return filter_jit(per_site_faith, compiler_options=compiler_options)


PER_SITE_FAITH_TOP_K = 8


def per_site_faith_scalars(
    rel_frob: dict[str, float], abs_frob: dict[str, float], top_k: int
) -> dict[str, float]:
    """Stable-keyed scalar summaries of the per-site Frobenius breakdown: the `top_k`
    LARGEST relative-Frobenius values (rank-indexed `1..k`, so the keys stay fixed as the
    ranking reshuffles over training), plus the largest absolute `‖Δ_s‖_F`. Which SITE holds
    each rank rides the bar chart, not these keys."""
    ranked_rel = sorted(rel_frob.values(), reverse=True)
    out = {
        f"eval/faith/rel_frob_top{rank}": ranked_rel[rank - 1]
        for rank in range(1, min(top_k, len(ranked_rel)) + 1)
    }
    out["eval/faith/abs_frob_max"] = max(abs_frob.values())
    return out
