"""CPU tests for the in-loop eval step at a tiny config.

Checks the torch-parity key set, the variant identities (rounded-at-impossible-threshold
== unmasked; CI-L0 saturates at C / 0 for out-of-range thresholds), CE correctness
against a hand-rolled computation, and determinism in the key.
"""

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import SiteSpec
from param_decomp.core.losses import relative_squared_error
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    PlacedModel,
)
from param_decomp.core.placement import PlacementRules
from param_decomp.core.recon import OutputAndHiddenActsReconstruction
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.experiments.lm.eval import (
    make_eval_step,
    next_token_cross_entropy,
)
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
)


def test_row_masked_relative_squared_error_excludes_padding_from_both_sums():
    clean = jnp.array([[[1.0, 2.0]], [[100.0, 200.0]]])
    masked = jnp.array([[[2.0, 0.0]], [[-300.0, 400.0]]])
    row_mask = jnp.array([1.0, 0.0])
    # Valid row only: numerator = 1 + 4; denominator = 1 + 4.
    assert float(relative_squared_error(masked, clean, valid_row_mask=row_mask)) == 1.0


def _build_ci_fn(model: PlacedModel, n_embd: int, key: jax.Array) -> PlacedCIFn:
    """One transformer chunk over all sites, reading the residual entering the first
    decomposed block. The old `CIArch(16, 1, 2, 32)` dims map onto the chunk arch."""
    site_names = model.site_names
    first_block = min(int(name.split(".")[1]) for name in site_names)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    return PlacedCIFn(
        fn=build_ci_fn(arch, model.sites, key),
        placement=resolve_ci_placement(arch, model.placement),
    )


class _PositionlessStub(eqx.Module):
    """A minimal positionless model whose methods are never called — used only to
    exercise the LM-only `has_position_axis` guards (which fire at construction)."""

    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, placement: PlacementRules) -> "_PositionlessStub":
        del placement
        raise AssertionError("positionless stub fn must not be called")

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> jax.Array:
        del masked_output, clean_output
        raise AssertionError("positionless stub fn must not be called")

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        del sites
        raise AssertionError("positionless stub fn must not be called")

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        del keys

    def clean_forward(
        self,
        resid: Any,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult:
        del resid, capture_keys, placement
        raise AssertionError("positionless stub fn must not be called")

    def prepare_compute_weights(self, vu: Any, placement: object | None) -> Any:
        del placement
        return vu

    def component_activation_forward(
        self,
        prepared_weights: Any,
        inputs: Any,
        /,
        *,
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult, dict[str, jax.Array]]:
        del prepared_weights, inputs, capture_keys, placement
        raise NotImplementedError

    def stack_ci(self, ci_lower: dict[str, Any]) -> dict[str, Any]:
        return ci_lower

    def masked_forward(
        self,
        prepared_weights: Any,
        inputs: Any,
        /,
        *,
        masking: Masking,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        del prepared_weights, inputs, masking, placement, capture_keys, remat
        raise AssertionError("positionless stub fn must not be called")

    def target_weight_sq_norms(self) -> dict[str, jax.Array]:
        raise AssertionError("positionless stub fn must not be called")

    def weight_deltas(self, vu: Any) -> dict[str, jax.Array]:
        del vu
        raise AssertionError("positionless stub fn must not be called")


def _positionless_model() -> PlacedModel:
    stub = _PositionlessStub(
        sites=(SiteSpec("linear1", 5, 2, 8, "linear1"), SiteSpec("linear2", 2, 5, 6, "linear2")),
        has_position_axis=False,
    )
    return PlacedModel(model=stub, placement=None)


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
    cfg = tiny_glu_cfg()
    C = 8
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 5, C))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )

    from param_decomp.core.components import init_component_stacks

    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))

    b, t = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)

    # rounding_threshold=-1 makes the rounded mask all-ones == the unmasked variant;
    # ci_alive_threshold=-1 makes every component alive -> L0 == C exactly.
    eval_step = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=-1.0,
        ci_alive_threshold=-1.0,
        l0_group_patterns=None,
        fresh_pgd=None,
        n_valid_rows=None,
    )
    out = eval_step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))

    variants = ("ci_masked", "unmasked", "stoch_masked", "random_masked", "rounded_masked")
    expected_keys = (
        {f"ce_kl/kl_{v}" for v in (*variants, "zero_masked")}
        | {f"ce_kl/ce_difference_{v}" for v in variants}
        | {f"l0/-1.0_{site}" for site in model.site_names}
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
    for site in model.site_names:
        assert float(out[f"l0/-1.0_{site}"]) == C

    # deterministic in the key; key-independent variants unchanged under a new key
    out_same = eval_step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))
    assert all(jnp.array_equal(out[k], out_same[k]) for k in out)
    out_other = eval_step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(6))
    for variant in ("ci_masked", "unmasked", "rounded_masked", "zero_masked"):
        assert jnp.array_equal(out[f"ce_kl/kl_{variant}"], out_other[f"ce_kl/kl_{variant}"])
    assert not jnp.array_equal(out["ce_kl/kl_stoch_masked"], out_other["ce_kl/kl_stoch_masked"])

    eval_step_dead = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=-1.0,
        ci_alive_threshold=1.5,
        l0_group_patterns=None,
        fresh_pgd=None,
        n_valid_rows=None,
    )
    out_dead = eval_step_dead(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))
    for site in model.site_names:
        assert float(out_dead[f"l0/1.5_{site}"]) == 0


def test_eval_step_fresh_pgd_probe():
    """The fresh-PGD probe must come out at least as adversarial as the unascended
    random source it starts from (ascent on a fixed objective), and be deterministic."""
    cfg = tiny_glu_cfg()
    C = 8
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 4, C))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )

    from param_decomp.core.components import init_component_stacks

    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    b, t = 2, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)

    ascended = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=FreshPGDReconEval(name="fresh_probe", n_steps=8, step_size=0.1),
        n_valid_rows=None,
    )
    unascended = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=FreshPGDReconEval(name="fresh_probe", n_steps=0, step_size=0.1),
        n_valid_rows=None,
    )
    out = ascended(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))
    out0 = unascended(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))

    assert "loss/fresh_probe" in out
    assert jnp.isfinite(out["loss/fresh_probe"])
    assert float(out["loss/fresh_probe"]) >= float(out0["loss/fresh_probe"]), (
        "8 sign-ascent steps must not be less adversarial than the raw random source"
    )
    out_same = ascended(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))
    assert jnp.array_equal(out["loss/fresh_probe"], out_same["loss/fresh_probe"])


def test_eval_step_fresh_pgd_hidden_acts_reconstruction_uses_and_logs_combined_objective():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 4, 8))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )

    from param_decomp.core.components import init_component_stacks

    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, cfg.vocab_size)
    hidden_acts_reconstruction = OutputAndHiddenActsReconstruction(
        coeff=3.0, points=("resid.5", "resid.8")
    )
    eval_step = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=FreshPGDReconEval(
            n_steps=2, step_size=0.1, reconstruction=hidden_acts_reconstruction
        ),
        n_valid_rows=None,
    )
    out = eval_step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))

    name = "loss/PGDReconLoss"
    per_point = [
        float(out[f"{name}/hidden_acts_reconstruction/{point}"])
        for point in hidden_acts_reconstruction.points
    ]
    aggregate = float(out[f"{name}/hidden_acts_reconstruction"])
    assert aggregate == pytest.approx(sum(per_point) / len(per_point), rel=1e-6)
    assert float(out[name]) == pytest.approx(
        float(out[f"{name}/e2e"]) + hidden_acts_reconstruction.coeff * aggregate, rel=1e-6
    )


def test_eval_step_fresh_pgd_ascends_hidden_acts_reconstruction_objective():
    """Changing only the auxiliary coefficient changes a one-step PGD trajectory."""
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 4, 8))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )

    from param_decomp.core.components import init_component_stacks

    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, cfg.vocab_size)
    points = ("resid.5", "resid.8")

    def run(coeff: float) -> Mapping[str, jax.Array]:
        step = make_eval_step(
            model,
            ci_fn.fn.capture_keys,
            rounding_threshold=0.0,
            ci_alive_threshold=0.0,
            l0_group_patterns=None,
            fresh_pgd=FreshPGDReconEval(
                n_steps=1,
                step_size=0.2,
                reconstruction=OutputAndHiddenActsReconstruction(coeff=coeff, points=points),
            ),
            n_valid_rows=None,
        )
        return step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))

    low = run(1e-12)
    high = run(100.0)
    keys = ("loss/PGDReconLoss/e2e", "loss/PGDReconLoss/hidden_acts_reconstruction")
    assert any(not jnp.array_equal(low[key], high[key]) for key in keys), (
        "hidden-activation reconstruction coefficient did not change the one-step PGD trajectory"
    )


@pytest.mark.slow
def test_eval_step_fresh_pgd_probe_device_count_invariant():
    """R-7 (eval facet): the fresh `c` PGD probe's KL must be invariant to device
    count up to float reassociation.

    The probe ascends `source += step * sign(dKL/dsource)` on a `(1,1,C+1)` source
    REPLICATED across the dp mesh. Each ascent's sign is taken AFTER the cotangent
    folds into the replicated leaf, so the gradient must be the GLOBAL-batch mean grad
    (torch all-reduce-AVG parity, S15/E19) — NOT a per-shard partial. A per-shard
    partial would flip signs on some shards, send the ascent down a different
    trajectory, and yield a different final KL. Comparing the single-layout run
    (mesh=None, whole batch on one device) against the batch-sharded run under the
    activated mesh (`jax.set_mesh`, as production's run boundary does) pins that the
    JAX cotangent into the replicated source is the global mean. At 1 device the two
    paths are identical; the test bites under
    `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
    """
    import numpy as np
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.components import init_component_stacks
    from param_decomp.core.placement import from_config
    from param_decomp.core.sharding import HSDP_MESH_AXES

    # The dp extent caps at 4: the guard is one scalar (the batch-mean KL), and the mean
    # over shards cancels the bugged per-shard trajectories' divergence as shards grow —
    # at 8 simulated devices the R-7 signal (~8e-6 here) sits within ~2.5x of benign
    # reassociation (~3e-6), while at <=4 shards it clears the tolerance by >=40x.
    n_dev = min(4, jax.device_count())
    mesh = Mesh(
        np.asarray(jax.devices()[:n_dev]).reshape(1, n_dev, 1),
        HSDP_MESH_AXES,
        axis_types=(AxisType.Explicit,) * 3,
    )

    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 4, 8))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))

    b, t = 4 * n_dev, 16
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)

    single_step = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=FreshPGDReconEval(n_steps=8, step_size=0.1),
        n_valid_rows=None,
    )
    sharded_model = PlacedModel(model=model.model, placement=from_config("ddp", mesh, sites))
    # Same weights (same key), paired with the resolved ddp CI rows — the two arms differ
    # only in placement, exactly the invariant under test.
    sharded_ci_fn = _build_ci_fn(sharded_model, cfg.n_embd, jax.random.PRNGKey(2))
    sharded_step = make_eval_step(
        sharded_model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=FreshPGDReconEval(n_steps=8, step_size=0.1),
        mesh=mesh,
        n_valid_rows=None,
    )

    out_single = single_step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))
    with jax.set_mesh(mesh):
        out_sharded = sharded_step(
            sharded_model, vu, sharded_ci_fn, token_ids, jax.random.PRNGKey(5)
        )

    single_kl = float(out_single["loss/PGDReconLoss"])
    sharded_kl = float(out_sharded["loss/PGDReconLoss"])
    assert jnp.isfinite(single_kl) and jnp.isfinite(sharded_kl)
    # reassociation-only tolerance: cross-shard reduction order differs, so bit-exactness
    # is not achievable, but a per-shard-partial grad (the R-7 bug) would change the
    # ascent sign on some shards and blow this far past tolerance.
    assert abs(single_kl - sharded_kl) <= 1e-4 * abs(single_kl) + 1e-6, (
        f"fresh-PGD eval probe KL diverged across shardings: single {single_kl!r} vs "
        f"sharded({n_dev}) {sharded_kl!r} — `c` source grad is not the global mean (R-7)"
    )


def test_eval_step_l0_groups_sum_member_sites():
    """torch CI_L0 `groups` parity: a group's L0 is the SUM of its fnmatch-member
    sites' L0s; an unmatched pattern refuses at build time."""
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    from param_decomp.core.components import init_component_stacks

    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    token_ids = jax.random.randint(jax.random.PRNGKey(3), (2, 16), 0, cfg.vocab_size)

    groups = {"layer_4": ("layers.4.*",), "total": ("*",)}
    eval_step = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns=groups,
        fresh_pgd=None,
        n_valid_rows=None,
    )
    out = eval_step(model, vu, ci_fn, token_ids, jax.random.PRNGKey(5))
    layer4_sites = [s for s in model.site_names if s.startswith("layers.4.")]
    expected_layer4 = sum(float(out[f"l0/0.0_{s}"]) for s in layer4_sites)
    expected_total = sum(float(out[f"l0/0.0_{s}"]) for s in model.site_names)
    assert abs(float(out["l0/0.0_layer_4"]) - expected_layer4) < 1e-4
    assert abs(float(out["l0/0.0_total"]) - expected_total) < 1e-4

    with pytest.raises(AssertionError, match="matches no sites"):
        make_eval_step(
            model,
            ci_fn.fn.capture_keys,
            rounding_threshold=0.0,
            ci_alive_threshold=0.0,
            l0_group_patterns={"ghost": ("layers.99.*",)},
            fresh_pgd=None,
            n_valid_rows=None,
        )


def test_make_eval_step_rejects_positionless_target():
    """CEandKLLosses/CI_L0 is LM-only (tokens + vocab logits over a sequence axis);
    constructing it against a positionless target must fail loud."""
    model = _positionless_model()
    assert not model.has_position_axis
    with pytest.raises(AssertionError, match="LM-only"):
        make_eval_step(
            model,
            frozenset(("linear1",)),
            rounding_threshold=0.0,
            ci_alive_threshold=0.0,
            l0_group_patterns=None,
            fresh_pgd=None,
            n_valid_rows=None,
        )


@pytest.mark.slow
def test_eval_step_n_valid_rows_masks_pad_tail():
    """A batch with garbage tail rows + `n_valid_rows` reproduces the unpadded scalars for
    every key-independent metric (pad rows carry zero weight, including inside the PGD
    objective). The stochastic variants draw shape-dependent randomness, so they only agree
    in expectation and are excluded."""
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )

    from param_decomp.core.components import init_component_stacks

    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))

    b, pad, t = 3, 2, 16
    tokens = jax.random.randint(jax.random.PRNGKey(3), (b, t), 0, cfg.vocab_size)
    padded = jnp.concatenate([tokens, jnp.zeros((pad, t), tokens.dtype)], axis=0)

    fresh_pgd = FreshPGDReconEval(
        n_steps=4,
        step_size=0.1,
        reconstruction=OutputAndHiddenActsReconstruction(coeff=2.0, points=("resid.5", "resid.8")),
    )
    reference_step = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.5,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=fresh_pgd,
        n_valid_rows=None,
    )
    masked_step = make_eval_step(
        model,
        ci_fn.fn.capture_keys,
        rounding_threshold=0.5,
        ci_alive_threshold=0.0,
        l0_group_patterns=None,
        fresh_pgd=fresh_pgd,
        n_valid_rows=b,
    )
    reference = reference_step(model, vu, ci_fn, tokens, jax.random.PRNGKey(5))
    masked = masked_step(model, vu, ci_fn, padded, jax.random.PRNGKey(5))

    assert set(masked) == set(reference)
    deterministic = [k for k in reference if "stoch" not in k and "random" not in k]
    assert any("PGDReconLoss" in k for k in deterministic)
    for k in deterministic:
        assert jnp.allclose(masked[k], reference[k], rtol=1e-3, atol=1e-5), (
            k,
            float(masked[k]),
            float(reference[k]),
        )
