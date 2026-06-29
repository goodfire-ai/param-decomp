"""In-loop eval pass: scalar parity with the torch eval metrics.

Implements the scalar core of the torch reference `eval:` block — `CEandKLLosses`
(six masking variants) and `CI_L0` — inside one jitted function, logged under the
exact torch wandb keys (`eval/ce_kl/<variant>`, `eval/l0/<threshold>_<site>`).
Plot-type metrics (CI histograms, activation density, per-component means, the
permutation/UV figures) ride the in-loop SLOW tier instead — natively in JAX
(`slow_eval.py`, SPEC S28; in-loop only, no offline CLI).

Variant semantics mirror `param_decomp_lab/eval_metrics/ce_and_kl_losses.py`: each
variant is a masked forward with ALL sites live and no routing; only `stoch_masked`
carries a weight-delta mask (torch `make_mask_infos` without weight deltas drops the
delta term — delta mask 0 here). CE is next-token cross-entropy with the first label
ignored; KL is per-position vs the clean (frozen) logits.

Cross-batch aggregation (the multi-`n_steps` eval pass in `run.py`): every key this
function returns is a per-BATCH scalar that the caller averages uniformly over the
eval batches. This is mean-safe against the torch reference — i.e. it matches torch's
accumulate-then-`compute()` to within float reassociation — only because every emitted
key is itself a per-batch reduction that torch *also* averages across batches, and the
eval batches are uniform `(B, T)`. The S8/D2 Jensen trap (a nonlinearity applied AFTER
the cross-batch reduction, so mean-of-batch-results ≠ result-of-global-batch) does NOT
arise here, because no emitted key wraps the cross-batch axis in a nonlinearity:

- `ce_kl/kl_<variant>`: torch `CEandKLLosses` accumulates `kl * n_positions` and divides
  by total positions (token-weighted mean of a per-batch mean). Uniform `(B, T)` makes
  token-weighting equal to the uniform `1/n_steps` average here.
- `ce_kl/ce_difference_<variant>` = `ce_v - ce_target`: torch averages this per-batch
  DIFFERENCE (computed inside `_calc_ce_and_kl_losses`), not a difference of grand means.
  Linear, so uniform-average parity holds.
- `ce_kl/ce_unrecovered_<variant>` = `(ce_v - ce_target) / (ce_zero - ce_target)`: the
  potential Jensen term. But torch forms this RATIO per-batch too, then averages the
  per-batch ratios — it never divides grand-mean numerator by grand-mean denominator. So
  averaging JAX's per-batch ratios matches torch exactly; no global-sum-before-divide is
  needed.
- `l0/<threshold>_<site|group>`: torch `CI_L0` collects per-batch L0 and averages them
  uniformly (`sum / count`); group L0 is a per-batch sum of member L0s. Linear.
- `loss/PGDReconLoss`: torch `PGDReconLoss` accumulates `kl * n` over batches and divides
  by total `n` (example-weighted mean of a per-batch mean KL); equals the uniform average
  under uniform `(B, T)`.

Both production yamls run `eval.n_steps: 1`, so today the cross-batch average is a no-op;
the parity argument above is what keeps it correct if `n_steps` is raised.
"""

from fnmatch import fnmatch
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.built_run import EvalPGDConfig
from param_decomp.components import DecompVU
from param_decomp.lm import DecomposedModel
from param_decomp.losses import kl_per_position
from param_decomp.sharding import batch_shard_leading
from param_decomp.train import COMPUTE_DT, cast_floating


def next_token_cross_entropy(
    logits: Float[Array, "B T vocab"], token_ids: Int[Array, "B T"]
) -> Array:
    """Mean fp32 CE of positions 0..T-2 predicting tokens 1..T-1 (torch: labels with
    the first position set to ignore_index)."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    label_log_probs = jnp.take_along_axis(log_probs[:, :-1], token_ids[:, 1:, None], axis=-1)[
        ..., 0
    ]
    return -label_log_probs.mean()


def make_eval_step(
    lm: DecomposedModel,
    rounding_threshold: float,
    ci_alive_threshold: float,
    l0_group_patterns: dict[str, tuple[str, ...]] | None,
    pgd: EvalPGDConfig | None,
    mesh: Mesh | None,
):
    """Build the `eqx.filter_jit`'d `eval_step(model, components, ci_fn, token_ids,
    residual, key) -> {metric_key: scalar}` with torch-parity keys (un-prefixed: the caller
    adds `eval/`). `model` (the frozen-weight-bearing `DecomposedModel`) is the jit ARG —
    its array leaves traced, never baked.

    `pgd` (an `EvalPGDConfig`) enables the fresh sign-PGD recon probe (torch
    `PGDReconLoss` with `init: random, mask_scope: c`): per site one
    `(1, 1, C+1)` source shared across batch AND positions, `n_steps` ascents of
    `source += step_size * sign(∂KL/∂source)` clamped to `[0, 1]`, KL evaluated at the
    final source. The global-mean KL makes the source gradient the global-batch
    gradient under GSPMD (torch all-reduce-AVG parity).

    CEandKLLosses / CI_L0 read tokens + vocab logits (next-token CE, KL over the vocab
    axis) and the fresh-PGD source is `(1, 1, C+1)` over a `(batch, sequence)` waist, so
    this metric is LM-only — asserted on `leading_axes` at construction."""
    assert lm.leading_axes == ("sequence",), (
        f"CEandKLLosses/CI_L0 eval is LM-only (tokens + vocab logits over a sequence "
        f"axis); model has leading_axes={lm.leading_axes}"
    )
    site_names = lm.site_names
    site_component_counts = {s.name: s.C for s in lm.sites}
    l0_groups: dict[str, tuple[str, ...]] = {}
    if l0_group_patterns is not None:
        for group_name, patterns in l0_group_patterns.items():
            members = tuple(
                site for site in site_names if any(fnmatch(site, pat) for pat in patterns)
            )
            assert members, f"CI_L0 group {group_name!r} matches no sites: {patterns}"
            l0_groups[group_name] = members

    def batch_sharded(x: Array) -> Array:
        return batch_shard_leading(x, mesh)

    def masked_forward(
        model: DecomposedModel, components_bf16: DecompVU, tokens: Array, masks: dict[str, Array],
        delta_masks: dict[str, Array],
    ) -> Array:  # fmt: skip
        return batch_sharded(
            model.masked_output(
                components_bf16, tokens, masks, delta_masks, None, site_names, True, remat=False
            )
        )

    @eqx.filter_jit
    def eval_step(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: Any,
        token_ids: Int[Array, "B T"],
        key: PRNGKeyArray,
    ) -> dict[str, Array]:
        token_ids = batch_sharded(token_ids)
        clean_output = batch_sharded(model.clean_output(token_ids))
        taps = model.read_activations(token_ids, ci_fn.input_names)

        components_bf16 = cast_floating(components, COMPUTE_DT)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        # one explicit C-shard -> batch-shard reshard; see train.py batch_sharded_ci
        ci_lower = {site: batch_sharded(v) for site, v in ci_fn_bf16(taps).lower.items()}

        leading = token_ids.shape
        zeros_delta = {site: jnp.zeros(leading, COMPUTE_DT) for site in site_names}

        stoch_key, random_key, pgd_key = random.split(key, 3)
        variant_masks: dict[str, tuple[dict[str, Array], dict[str, Array]]] = {}
        variant_masks["ci_masked"] = (ci_lower, zeros_delta)
        variant_masks["unmasked"] = (
            {site: jnp.ones_like(ci_lower[site]) for site in site_names},
            zeros_delta,
        )
        stoch_masks = {}
        stoch_deltas = {}
        for site_idx, site in enumerate(site_names):
            ci_site = ci_lower[site]
            stochastic_source = random.uniform(
                random.fold_in(stoch_key, site_idx), ci_site.shape, COMPUTE_DT
            )
            stoch_masks[site] = ci_site + (1.0 - ci_site) * stochastic_source
            stoch_deltas[site] = random.uniform(
                random.fold_in(stoch_key, len(site_names) + site_idx), leading, COMPUTE_DT
            )
        variant_masks["stoch_masked"] = (stoch_masks, stoch_deltas)
        variant_masks["random_masked"] = (
            {
                site: random.uniform(
                    random.fold_in(random_key, site_idx), ci_lower[site].shape, COMPUTE_DT
                )
                for site_idx, site in enumerate(site_names)
            },
            zeros_delta,
        )
        variant_masks["rounded_masked"] = (
            {site: (ci_lower[site] > rounding_threshold).astype(COMPUTE_DT) for site in site_names},
            zeros_delta,
        )
        variant_masks["zero_masked"] = (
            {site: jnp.zeros_like(ci_lower[site]) for site in site_names},
            zeros_delta,
        )

        kl: dict[str, Array] = {}
        ce: dict[str, Array] = {}
        for variant, (masks, delta_masks) in variant_masks.items():
            variant_logits = masked_forward(model, components_bf16, token_ids, masks, delta_masks)
            kl[variant] = kl_per_position(variant_logits, clean_output)
            ce[variant] = next_token_cross_entropy(variant_logits, token_ids)
        target_ce = next_token_cross_entropy(clean_output, token_ids)

        out: dict[str, Array] = {}
        for variant in variant_masks:
            out[f"ce_kl/kl_{variant}"] = kl[variant]
        difference_variants = tuple(v for v in variant_masks if v != "zero_masked")
        for variant in difference_variants:
            out[f"ce_kl/ce_difference_{variant}"] = ce[variant] - target_ce
        for variant in difference_variants:
            out[f"ce_kl/ce_unrecovered_{variant}"] = (ce[variant] - target_ce) / (
                ce["zero_masked"] - target_ce
            )
        site_l0 = {
            site: (ci_lower[site] > ci_alive_threshold).astype(jnp.float32).sum(-1).mean()
            for site in site_names
        }
        for site, value in site_l0.items():
            out[f"l0/{ci_alive_threshold}_{site}"] = value
        # torch CI_L0 groups: the group's L0 is the SUM of its member sites' L0s
        for group_name, members in l0_groups.items():
            out[f"l0/{ci_alive_threshold}_{group_name}"] = sum(
                (site_l0[site] for site in members), start=jnp.zeros((), jnp.float32)
            )

        if pgd is not None:
            pgd_n_steps = pgd.n_steps
            pgd_step_size = pgd.step_size

            def kl_at_sources(adversarial_sources: dict[str, Array]) -> Array:
                masks = {}
                delta_masks = {}
                for site in site_names:
                    source = adversarial_sources[site].astype(COMPUTE_DT)
                    ci_site = ci_lower[site]
                    masks[site] = ci_site + (1.0 - ci_site) * source[..., :-1]
                    delta_masks[site] = source[..., -1]
                masked = masked_forward(model, components_bf16, token_ids, masks, delta_masks)
                return kl_per_position(masked, clean_output)

            def ascend(
                adversarial_sources: dict[str, Array], _: None
            ) -> tuple[dict[str, Array], None]:
                source_grads = jax.grad(kl_at_sources)(adversarial_sources)
                return {
                    site: jnp.clip(
                        adversarial_sources[site] + pgd_step_size * jnp.sign(source_grads[site]),
                        0.0,
                        1.0,
                    )
                    for site in site_names
                }, None

            initial_sources = {
                site: random.uniform(
                    random.fold_in(pgd_key, site_idx),
                    (1, 1, site_component_counts[site] + 1),
                    jnp.float32,
                )
                for site_idx, site in enumerate(site_names)
            }
            final_sources, _ = jax.lax.scan(ascend, initial_sources, None, length=pgd_n_steps)
            out["loss/PGDReconLoss"] = kl_at_sources(final_sources)
        return out

    return eval_step
