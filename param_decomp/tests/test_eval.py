"""CPU tests for the in-loop eval step at a tiny config.

Checks the torch-parity key set, the variant identities (rounded-at-impossible-threshold
== unmasked; CI-L0 saturates at C / 0 for out-of-range thresholds), CE correctness
against a hand-rolled computation, and determinism in the key.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from param_decomp.built_run import EvalPGDConfig
from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    CIFn,
    build_ci_fn,
)
from param_decomp.components import SiteSpec
from param_decomp.eval import make_eval_step, next_token_cross_entropy
from param_decomp.lm import DecomposedModel
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.tests.test_llama8b import (
    _tiny_cfg,
    _tiny_decomposed_lm,
)


def _build_ci_fn(lm: DecomposedModel, n_embd: int, key: jax.Array) -> CIFn:
    """One transformer chunk over all sites, reading the residual entering the first
    decomposed block. The old `CIArch(16, 1, 2, 32)` dims map onto the chunk arch."""
    site_names = lm.site_names
    first_block = min(int(name.split(".")[1]) for name in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        n_heads=2,
        mlp_hidden=32,
    )
    return build_ci_fn(arch, lm.sites, key)


class _PositionlessStub(eqx.Module):
    """A minimal `leading_axes=()` model whose methods are never called — used only to
    exercise the LM-only `leading_axes` guards (which fire at construction)."""

    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: Any) -> "_PositionlessStub":
        del mesh
        raise AssertionError("positionless stub fn must not be called")

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> jax.Array:
        del masked_output, clean_output
        raise AssertionError("positionless stub fn must not be called")

    def clean_output(self, resid: Any) -> Any:
        del resid
        raise AssertionError("positionless stub fn must not be called")

    def read_activations(self, resid: Any, wanted: tuple[str, ...]) -> dict[str, jax.Array]:
        del resid, wanted
        raise AssertionError("positionless stub fn must not be called")

    def masked_output(
        self,
        vu: Any,
        resid: Any,
        masks: Any,
        delta_masks: Any,
        routes: Any,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> Any:
        del vu, resid, masks, delta_masks, routes, live, has_delta
        raise AssertionError("positionless stub fn must not be called")

    def masked_site_outputs(
        self,
        vu: Any,
        resid: Any,
        masks: Any,
        delta_masks: Any,
        routes: Any,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, jax.Array]:
        del vu, resid, masks, delta_masks, routes, live, has_delta
        raise AssertionError("positionless stub fn must not be called")

    def weight_deltas(self, vu: Any) -> dict[str, jax.Array]:
        del vu
        raise AssertionError("positionless stub fn must not be called")


def _positionless_model() -> DecomposedModel:
    return _PositionlessStub(
        sites=(SiteSpec("linear1", 5, 2, 8), SiteSpec("linear2", 2, 5, 6)),
        leading_axes=(),
    )


def test_next_token_cross_entropy_matches_manual():
    b, t, v = 2, 5, 7
    logits = jax.random.normal(jax.random.PRNGKey(0), (b, t, v))
    token_ids = jax.random.randint(jax.random.PRNGKey(1), (b, t), 0, v)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    manual = -jnp.mean(
        jnp.stack([log_probs[i, j, token_ids[i, j + 1]] for i in range(b) for j in range(t - 1)])
    )
    assert jnp.allclose(next_token_cross_entropy(logits, token_ids), manual, rtol=1e-6)


def test_eval_step_keys_identities_and_determinism():
    cfg = _tiny_cfg()
    C = 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

    from param_decomp.components import init_decomp_vu

    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))

    b, t = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    # rounding_threshold=-1 makes the rounded mask all-ones == the unmasked variant;
    # an L0 threshold of -1 makes every component alive -> L0 == C exactly.
    eval_step = make_eval_step(
        lm,
        rounding_threshold=-1.0,
        ci_alive_thresholds=(-1.0,),
        l0_group_patterns=None,
        pgd=None,
        mesh=None,
    )
    out = eval_step(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))

    variants = ("ci_masked", "unmasked", "stoch_masked", "random_masked", "rounded_masked")
    expected_keys = (
        {f"ce_kl/kl_{v}" for v in (*variants, "zero_masked")}
        | {f"ce_kl/ce_difference_{v}" for v in variants}
        | {f"ce_kl/ce_unrecovered_{v}" for v in variants}
        | {f"l0/-1.0_{site}" for site in lm.site_names}
    )
    assert set(out) == expected_keys

    for key, value in out.items():
        assert jnp.isfinite(value), (key, value)
    for variant in (*variants, "zero_masked"):
        assert out[f"ce_kl/kl_{variant}"] >= 0, variant

    assert jnp.allclose(out["ce_kl/kl_rounded_masked"], out["ce_kl/kl_unmasked"], rtol=1e-3)
    assert jnp.allclose(
        out["ce_kl/ce_difference_rounded_masked"], out["ce_kl/ce_difference_unmasked"], rtol=1e-3
    )
    for site in lm.site_names:
        assert float(out[f"l0/-1.0_{site}"]) == C

    # deterministic in the key; key-independent variants unchanged under a new key
    out_same = eval_step(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))
    assert all(jnp.array_equal(out[k], out_same[k]) for k in out)
    out_other = eval_step(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(6))
    for variant in ("ci_masked", "unmasked", "rounded_masked", "zero_masked"):
        assert jnp.array_equal(out[f"ce_kl/kl_{variant}"], out_other[f"ce_kl/kl_{variant}"])
    assert not jnp.array_equal(out["ce_kl/kl_stoch_masked"], out_other["ce_kl/kl_stoch_masked"])

    eval_step_dead = make_eval_step(
        lm,
        rounding_threshold=-1.0,
        ci_alive_thresholds=(1.5,),
        l0_group_patterns=None,
        pgd=None,
        mesh=None,
    )
    out_dead = eval_step_dead(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))
    for site in lm.site_names:
        assert float(out_dead[f"l0/1.5_{site}"]) == 0


def test_eval_step_fresh_pgd_probe():
    """The fresh-PGD probe must come out at least as adversarial as the unascended
    random source it starts from (ascent on a fixed objective), and be deterministic."""
    cfg = _tiny_cfg()
    C = 8
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 4, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

    from param_decomp.components import init_decomp_vu

    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    b, t = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    ascended = make_eval_step(
        lm,
        rounding_threshold=0.0,
        ci_alive_thresholds=(0.0,),
        l0_group_patterns=None,
        pgd=EvalPGDConfig(n_steps=8, step_size=0.1),
        mesh=None,
    )
    unascended = make_eval_step(
        lm,
        rounding_threshold=0.0,
        ci_alive_thresholds=(0.0,),
        l0_group_patterns=None,
        pgd=EvalPGDConfig(n_steps=0, step_size=0.1),
        mesh=None,
    )
    out = ascended(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))
    out0 = unascended(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))

    assert "loss/PGDReconLoss" in out
    assert jnp.isfinite(out["loss/PGDReconLoss"])
    assert float(out["loss/PGDReconLoss"]) >= float(out0["loss/PGDReconLoss"]), (
        "8 sign-ascent steps must not be less adversarial than the raw random source"
    )
    out_same = ascended(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))
    assert jnp.array_equal(out["loss/PGDReconLoss"], out_same["loss/PGDReconLoss"])


def test_eval_step_fresh_pgd_probe_device_count_invariant():
    """R-7 (eval facet): the fresh c-scope PGD probe's KL must be invariant to device
    count up to float reassociation.

    The probe ascends `source += step * sign(dKL/dsource)` on a `(1,1,C+1)` source
    REPLICATED across the dp mesh. Each ascent's sign is taken AFTER the cotangent
    folds into the replicated leaf, so the gradient must be the GLOBAL-batch mean grad
    (torch all-reduce-AVG parity, S15/E19) — NOT a per-shard partial. A per-shard
    partial would flip signs on some shards, send the ascent down a different
    trajectory, and yield a different final KL. Comparing the single-layout run
    (mesh=None, whole batch on one device) against the GSPMD batch-sharded run pins
    that the JAX cotangent into the replicated source is the global mean. At 1 device
    the two paths are identical; the test bites under
    `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
    """
    from param_decomp.components import init_decomp_vu
    from param_decomp.sharding import dp_mesh, shard_batch

    mesh = dp_mesh()
    n_dev = mesh.devices.size

    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 4, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))

    b, t = 4 * n_dev, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)
    residual = jax.random.normal(jax.random.PRNGKey(4), (b, t, cfg.n_embd)) * 0.5

    single_step = make_eval_step(
        lm, rounding_threshold=0.0, ci_alive_thresholds=(0.0,),
        l0_group_patterns=None, pgd=EvalPGDConfig(n_steps=8, step_size=0.1), mesh=None,
    )  # fmt: skip
    sharded_step = make_eval_step(
        lm, rounding_threshold=0.0, ci_alive_thresholds=(0.0,),
        l0_group_patterns=None, pgd=EvalPGDConfig(n_steps=8, step_size=0.1), mesh=mesh,
    )  # fmt: skip

    out_single = single_step(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))
    out_sharded = sharded_step(
        lm, vu, ci_fn, token_ids, shard_batch(residual, mesh, batch_axis=0), jax.random.PRNGKey(5)
    )

    single_kl = float(out_single["loss/PGDReconLoss"])
    sharded_kl = float(out_sharded["loss/PGDReconLoss"])
    assert jnp.isfinite(single_kl) and jnp.isfinite(sharded_kl)
    # reassociation-only tolerance: cross-shard reduction order differs, so bit-exactness
    # is not achievable, but a per-shard-partial grad (the R-7 bug) would change the
    # ascent sign on some shards and blow this far past tolerance.
    assert abs(single_kl - sharded_kl) <= 1e-4 * abs(single_kl) + 1e-6, (
        f"fresh-PGD eval probe KL diverged across shardings: single {single_kl!r} vs "
        f"sharded({n_dev}) {sharded_kl!r} — c-scope source grad is not the global mean (R-7)"
    )


def test_eval_step_l0_groups_sum_member_sites():
    """torch CI_L0 `groups` parity: a group's L0 is the SUM of its fnmatch-member
    sites' L0s; an unmatched pattern refuses at build time."""
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    from param_decomp.components import init_decomp_vu

    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, cfg.vocab_size)
    residual = jax.random.normal(jax.random.PRNGKey(4), (2, 16, cfg.n_embd)) * 0.5

    groups = {"layer_4": ("layers.4.*",), "total": ("*",)}
    eval_step = make_eval_step(
        lm, rounding_threshold=0.0, ci_alive_thresholds=(0.0,),
        l0_group_patterns=groups, pgd=None, mesh=None,
    )  # fmt: skip
    out = eval_step(lm, vu, ci_fn, token_ids, residual, jax.random.PRNGKey(5))
    layer4_sites = [s for s in lm.site_names if s.startswith("layers.4.")]
    expected_layer4 = sum(float(out[f"l0/0.0_{s}"]) for s in layer4_sites)
    expected_total = sum(float(out[f"l0/0.0_{s}"]) for s in lm.site_names)
    assert abs(float(out["l0/0.0_layer_4"]) - expected_layer4) < 1e-4
    assert abs(float(out["l0/0.0_total"]) - expected_total) < 1e-4

    with pytest.raises(AssertionError, match="matches no sites"):
        make_eval_step(
            lm, rounding_threshold=0.0, ci_alive_thresholds=(0.0,),
            l0_group_patterns={"ghost": ("layers.99.*",)}, pgd=None, mesh=None,
        )  # fmt: skip


def test_make_eval_step_rejects_positionless_target():
    """CEandKLLosses/CI_L0 is LM-only (tokens + vocab logits over a sequence axis);
    constructing it against a positionless (`leading_axes=()`) target must fail loud."""
    lm = _positionless_model()
    assert lm.leading_axes == ()
    with pytest.raises(AssertionError, match="LM-only"):
        make_eval_step(
            lm, rounding_threshold=0.0, ci_alive_thresholds=(0.0,),
            l0_group_patterns=None, pgd=None, mesh=None,
        )  # fmt: skip
