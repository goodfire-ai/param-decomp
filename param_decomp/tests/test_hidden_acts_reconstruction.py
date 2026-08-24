"""Target-selected residual relative-MSE auxiliary recon loss (SPEC S35)."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from pydantic import ValidationError

from param_decomp.core import train as train_module
from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
)
from param_decomp.core.components import init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    FaithfulnessLossConfig,
    HiddenActsReconstruction,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.losses import OutputOnlyReconstructionLoss, relative_squared_error
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.objective import build_objective
from param_decomp.core.recon import (
    ForwardObservations,
    OutputOnlyReconstruction,
    ReconstructionSpec,
)
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.glu_transformer import parse_site_name as parse_glu_site_name
from param_decomp.targets.llama_simple_mlp import site_specs
from param_decomp.targets.testing import (
    SIMPLE_MLP_MIXED_SITE_CS,
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
    tiny_simple_mlp_cfg,
    tiny_simple_mlp_chunkwise_ci_fn,
    tiny_simple_mlp_decomposed_model,
)

SIMPLE_MLP_POINTS = ("resid.3", "resid.4", "resid.5", "resid.6")
"""Meaningful S35 points for blocks 2-3 of 6: downstream residual boundaries only."""


def _hidden_acts_reconstruction(coeff: float | None) -> HiddenActsReconstruction | None:
    return (
        None if coeff is None else HiddenActsReconstruction(coeff=coeff, points=SIMPLE_MLP_POINTS)
    )


def test_relative_squared_error_hand_computed():
    """Values chosen so SQUARING is load-bearing: with all-unit magnitudes an absolute-error
    implementation would score identically, so the fixture would not distinguish them."""
    clean = jnp.array([[[3.0, 4.0]]])  # batch=1, seq=1, d=2
    masked = jnp.array([[[3.0, 0.0]]])
    got = float(relative_squared_error(masked, clean))
    # sq_diff = 0 + 16 = 16; sq_clean = 9 + 16 = 25 -> 16/25. (Not a dyadic rational, so
    # compare approximately — fp32 cannot represent it exactly.)
    assert got == pytest.approx(16 / 25)
    # And it is the SQUARE that is being measured: a relative-absolute-error implementation
    # would score 4/7 here, which the fixture separates by a wide margin.
    assert got != pytest.approx(4 / 7, rel=1e-3)


def test_relative_squared_error_zero_when_identical():
    acts = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 16))
    assert float(relative_squared_error(acts, acts)) == 0.0


def test_relative_squared_error_is_scale_free_across_widths():
    """Each point divides by its OWN clean scale, so the same proportional drift scores equally
    at any width or magnitude — which is what lets the mean mix points of differing width."""
    # Powers of two and an exactly-representable ratio, so fp32 makes this exact, not approximate.
    narrow = jnp.ones((2, 4, 8))
    wide = jnp.full((2, 4, 64), 4.0)
    assert float(relative_squared_error(narrow * 1.25, narrow)) == 0.0625
    assert float(relative_squared_error(wide * 1.25, wide)) == 0.0625


def test_relative_squared_error_pools_energy_over_batch_and_sequence():
    # Per-position ratios are [1, 9/9] -> 1, but pooled energy is (1+9)/(1+9)=1 here;
    # choose asymmetric clean energies so the two reductions cleanly differ.
    clean = jnp.array([[[1.0], [3.0]], [[2.0], [4.0]]])
    masked = jnp.array([[[2.0], [3.0]], [[2.0], [0.0]]])
    pooled = (1.0 + 0.0 + 0.0 + 16.0) / (1.0 + 9.0 + 4.0 + 16.0)
    per_position = (1.0 / 1.0 + 0.0 / 9.0 + 0.0 / 4.0 + 16.0 / 16.0) / 4
    got = float(relative_squared_error(masked, clean))
    assert got == pytest.approx(pooled)
    assert got != pytest.approx(per_position)


def test_target_resolves_explicit_points_and_refuses_unknown_ones():
    """The loss chooses points; the target validates its own closed vocabulary before JIT."""
    cfg = tiny_simple_mlp_cfg()
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))

    tokens = jax.random.randint(jax.random.PRNGKey(1), (1, 4), 0, cfg.vocab_size)
    assert (
        tuple(model.clean_forward(tokens, frozenset(SIMPLE_MLP_POINTS), placement=None).captures)
        == SIMPLE_MLP_POINTS
    )
    model.assert_hidden_acts_reconstruction_points(SIMPLE_MLP_POINTS)
    # Capture may inspect an invariant point, but a mean loss refuses it: resid.2 enters the
    # first decomposed block and therefore cannot change under masking.
    assert tuple(model.clean_forward(tokens, frozenset(("resid.2",)), placement=None).captures) == (
        "resid.2",
    )
    with pytest.raises(AssertionError, match="guaranteed zeros"):
        model.assert_hidden_acts_reconstruction_points(("resid.2",))
    model.assert_hidden_acts_reconstruction_points(("h.2.attn.q_proj.out", "post_attn.2"))
    with pytest.raises(AssertionError, match="guaranteed zeros"):
        model.assert_hidden_acts_reconstruction_points(("h.2.attn.k_proj.out",))
    with pytest.raises(AssertionError, match="guaranteed zeros"):
        model.assert_hidden_acts_reconstruction_points(("attn_in.2",))
    model.assert_hidden_acts_reconstruction_points(
        ("attn_out.2", "mlp_in.2", "mlp_hidden.2", "attn_in.3")
    )
    with pytest.raises(AssertionError, match="out of range"):
        model.clean_forward(tokens, frozenset(("resid.7",)), placement=None)
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        model.clean_forward(tokens, frozenset(("python_local",)), placement=None)


def test_masked_residuals_reassemble_correctly_across_the_three_scan_segments():
    """The masked forward runs as [frozen prefix][live block][frozen suffix] with static
    boundaries, and stitches each requested residual out of whichever segment produced it.
    That index arithmetic is the only new arithmetic in this feature and an off-by-one would
    yield finite, plausible garbage (block k's residual reported as block k+1's).

    Pin it by exploiting a property the segmentation guarantees: with only a MIDDLE block
    DECOMPOSED (the segmentation is structural, derived from the model's own site set),
    every boundary at or upstream of it runs the untouched frozen target, so it must
    equal the clean forward EXACTLY — while boundaries downstream of it must differ. A
    misaligned stitch breaks one side or the other."""
    cfg = tiny_glu_cfg()
    C, n_layer = 4, cfg.n_layer
    live_block = n_layer // 2
    assert 0 < live_block < n_layer - 1, "need a genuine prefix AND suffix segment"

    sites = glu_site_specs(cfg, mlp_family_site_cs(live_block, live_block, C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    prepared = model.prepare_compute_weights(vu, None)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 8), 0, cfg.vocab_size)

    taps = tuple(f"resid.{b}" for b in range(1, n_layer + 1))
    assert all(parse_glu_site_name(n)[0] == live_block for n in model.site_names)
    c_by_site = {s.name: s.C for s in model.sites}
    masks = {n: jnp.full((2, 8, c_by_site[n]), 0.25) for n in model.site_names}
    delta_masks = {n: jnp.full((2, 8), 0.25) for n in model.site_names}

    masked_result = model.masked_forward(
        prepared,
        tokens,
        masking=MaterializedMasking(
            component_masks=masks,
            weight_delta_masks=delta_masks,
        ),
        placement=None,
        capture_keys=frozenset(taps),
        remat=False,
    )
    masked_taps = masked_result.captures
    clean_taps = model.clean_forward(tokens, frozenset(taps), placement=None).captures

    for tap in taps:
        boundary = int(tap.split(".")[1])
        identical = bool(jnp.array_equal(masked_taps[tap], clean_taps[tap]))
        if boundary <= live_block:
            assert identical, f"{tap} is upstream of block {live_block} but differs from clean"
        else:
            assert not identical, f"{tap} is downstream of block {live_block} but matches clean"


def test_build_loss_terms_threads_hidden_acts_reconstruction_coeff():
    cfg = StochasticReconSubsetLossConfig(
        routing=UniformKSubsetRoutingConfig(),
        coeff=0.5,
        n_mask_samples=1,
        hidden_acts_reconstruction=_hidden_acts_reconstruction(0.1),
    )
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(
                    max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            cfg,
        ),
        ("a", "b"),
    )
    (term,) = losses.recon
    assert term.hidden_acts_reconstruction is not None
    assert term.hidden_acts_reconstruction.coeff == 0.1


def test_hidden_acts_reconstruction_config_needs_both_halves_and_sane_points():
    """Strength and points are two halves of one setting (SPEC S35), so they live in one
    object: a coefficient with nowhere to measure is not expressible, which is why there is
    no cross-section validator. What still needs checking is the points themselves."""
    HiddenActsReconstruction(coeff=0.3, points=("resid.3", "resid.4"))

    with pytest.raises(ValidationError):  # a strength alone is not a valid object
        HiddenActsReconstruction(coeff=0.3)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError):  # nor are points alone
        HiddenActsReconstruction(points=("resid.3",))  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError, match="at least one activation"):
        HiddenActsReconstruction(coeff=0.3, points=())
    with pytest.raises(ValidationError, match="duplicate points"):
        HiddenActsReconstruction(coeff=0.3, points=("resid.3", "resid.3"))
    with pytest.raises(ValidationError):
        HiddenActsReconstruction(coeff=0.0, points=("resid.3",))


def _one_step_with_recon(
    recon: StochasticReconSubsetLossConfig | PGDReconLossConfig,
):
    cfg = tiny_simple_mlp_cfg()
    seq = 16
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(
                    max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            recon,
        ),
        model.site_names,
    )
    placed = PlacedModel(model=model, placement=None)
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=losses,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        faithfulness=faithfulness_loss_for(model),
    )
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    return step(placed, state, tokens, jax.random.PRNGKey(100))


def _step_with_stochastic_subset_recon(hidden_acts_reconstruction_coeff: float | None):
    return _one_step_with_recon(
        StochasticReconSubsetLossConfig(
            routing=UniformKSubsetRoutingConfig(),
            coeff=0.5,
            n_mask_samples=1,
            hidden_acts_reconstruction=_hidden_acts_reconstruction(
                hidden_acts_reconstruction_coeff
            ),
        )
    )


def test_fresh_pgd_training_ascends_the_combined_objective():
    def run(coeff: float) -> dict[str, jax.Array]:
        _, metrics = _one_step_with_recon(
            PGDReconLossConfig(
                coeff=0.5,
                init="random",
                source_shape="bc",
                n_steps=3,
                step_size=0.2,
                hidden_acts_reconstruction=_hidden_acts_reconstruction(coeff),
            )
        )
        return {key: value for key, value in metrics.items() if key.startswith("loss/PGD")}

    low = run(1e-12)
    high = run(100.0)
    keys = ("loss/PGDReconLoss/e2e", "loss/PGDReconLoss/hidden_acts_reconstruction")
    assert any(not bool(jnp.array_equal(low[key], high[key])) for key in keys)


def test_hidden_acts_reconstruction_coeff_trains_finite():
    _, metrics = _step_with_stochastic_subset_recon(hidden_acts_reconstruction_coeff=0.3)
    assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
    assert metrics["loss/StochasticReconSubsetLoss"] > 0


def test_hidden_acts_reconstruction_coeff_logs_e2e_and_per_point_breakdown_separately():
    _, metrics = _step_with_stochastic_subset_recon(hidden_acts_reconstruction_coeff=0.3)
    name = "loss/StochasticReconSubsetLoss"
    assert name in metrics
    assert float(metrics[f"{name}/e2e"]) != float(metrics[name])  # combined != bare e2e
    # Keyed by point name, never a positional index.
    for point in SIMPLE_MLP_POINTS:
        assert f"{name}/hidden_acts_reconstruction/{point}" in metrics
    assert f"{name}/hidden_acts_reconstruction/resid.0" not in metrics


def test_hidden_acts_reconstruction_combined_is_e2e_plus_coeff_times_point_MEAN():
    """The production path's own composition (SPEC S35), not the loss fn in isolation:
    `loss/<name>` == `e2e + coeff * mean_over_points`. Pins the MEAN specifically — a
    `jnp.mean` -> `jnp.sum` slip in `train.py` fails here as soon as there is >1 point."""
    coeff = 0.3
    taps = SIMPLE_MLP_POINTS
    assert len(taps) > 1, "a single point cannot distinguish mean from sum"

    _, metrics = _step_with_stochastic_subset_recon(hidden_acts_reconstruction_coeff=coeff)
    name = "loss/StochasticReconSubsetLoss"
    per_tap = [float(metrics[f"{name}/hidden_acts_reconstruction/{tap}"]) for tap in taps]
    aggregate = float(metrics[f"{name}/hidden_acts_reconstruction"])
    assert aggregate == pytest.approx(sum(per_tap) / len(per_tap), rel=1e-6)
    assert float(metrics[name]) == pytest.approx(
        float(metrics[f"{name}/e2e"]) + coeff * aggregate, rel=1e-6
    )


def test_hidden_acts_reconstruction_coeff_disabled_logs_no_breakdown_at_all(
    monkeypatch: pytest.MonkeyPatch,
):
    # With no aux part the combined value IS the e2e loss, so a separate `/e2e` series would
    # be a duplicate of `loss/<name>`.
    state, metrics = _step_with_stochastic_subset_recon(hidden_acts_reconstruction_coeff=None)

    # Compare against the pre-auxiliary scalar objective in the same process. A fixed digest is
    # not a valid bit-identity oracle: XLA can produce different (but internally deterministic)
    # fp32 reduction orderings on different CPU architectures. This differential pin remains
    # exact while exercising the complete optimizer trajectory on every backend CI uses.
    def legacy_objective(
        recon_loss_fn: Callable[[Any, Any], jax.Array],
        *,
        masked: ForwardObservations,
        clean: ForwardObservations,
        reconstruction: ReconstructionSpec,
        valid_row_mask: jax.Array | None = None,
    ) -> OutputOnlyReconstructionLoss:
        del valid_row_mask
        assert isinstance(reconstruction, OutputOnlyReconstruction)
        return OutputOnlyReconstructionLoss(recon_loss_fn(masked.output, clean.output))

    monkeypatch.setattr(train_module, "reconstruction_loss", legacy_objective)
    legacy_state, _ = _step_with_stochastic_subset_recon(hidden_acts_reconstruction_coeff=None)
    current_leaves = tuple(leaf for leaf in jax.tree.leaves(state) if eqx.is_array(leaf))
    legacy_leaves = tuple(leaf for leaf in jax.tree.leaves(legacy_state) if eqx.is_array(leaf))
    assert len(current_leaves) == len(legacy_leaves)
    assert all(
        bool(jnp.array_equal(current, legacy))
        for current, legacy in zip(current_leaves, legacy_leaves, strict=True)
    )

    name = "loss/StochasticReconSubsetLoss"
    assert name in metrics
    assert f"{name}/e2e" not in metrics
    assert not any(k.startswith(f"{name}/hidden_acts_reconstruction") for k in metrics)


def _ppgd_run(
    hidden_acts_reconstruction_coeff: float | None,
    n_steps: int,
    n_warmup: int = 1,
    *,
    constant_source_lr: bool = False,
) -> tuple[TrainState, list[dict[str, jnp.ndarray]]]:
    """`n_steps` of a persistent-PGD term at a given hidden-activation reconstruction strength, everything else
    (seeds, batch, schedules) fixed."""
    cfg = tiny_simple_mlp_cfg()
    seq = 16
    sites = site_specs(cfg, SIMPLE_MLP_MIXED_SITE_CS)
    model = tiny_simple_mlp_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = tiny_simple_mlp_chunkwise_ci_fn(model, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)

    src = init_persistent_sources(
        model.site_names,
        tuple(s.C for s in model.sites),
        (1, seq),
        jnp.float32,
        jax.random.PRNGKey(3),
    )
    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=0.5,
        adversary_objective="term",
        source_shape="sc",
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(
                    (Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=1.0))
                    if constant_source_lr
                    else (
                        Knot(at=0.0, frac=0.0),
                        Knot(at=0.025, frac=1.0),
                        Knot(at=1.0, frac=1.0),
                    )
                ),
            ),
        ),
        n_warmup_steps=n_warmup,
        hidden_acts_reconstruction=_hidden_acts_reconstruction(hidden_acts_reconstruction_coeff),
    )
    assert ppgd_cfg.coeff is not None
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={
                ppgd_cfg.type: PersistentAdversary(
                    sources=src,
                    opt_state=init_sources_adam_state(src),
                    state_key=ppgd_cfg.type,
                    adam=ppgd_cfg.optimizer,
                    n_warmup=ppgd_cfg.n_warmup_steps,
                )
            },
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    losses = build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=ScheduleConfig(
                    max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            ppgd_cfg,
        ),
        model.site_names,
    )
    placed = PlacedModel(model=model, placement=None)
    step = make_train_step(
        model_static=placed,
        substrate=ForwardSubstrate.of(
            placed,
            remat_recon_forwards=True,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=losses,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=100,
        faithfulness=faithfulness_loss_for(model),
    )
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, seq), 0, cfg.vocab_size)
    per_step: list[dict[str, jnp.ndarray]] = []
    for i in range(n_steps):
        state, metrics = step(placed, state, tokens, jax.random.PRNGKey(100 + i))
        per_step.append(metrics)
    return state, per_step


def test_hidden_acts_reconstruction_with_persistent_pgd_adversary():
    """The adversary's warmup + final ascent must run through the combined (e2e +
    hidden-activation reconstruction) objective without shape/finiteness errors (SPEC S35 x S13'/S14')."""
    n_steps, n_warmup = 3, 1
    ppgd_key = "PersistentPGDReconLoss"
    state, per_step = _ppgd_run(hidden_acts_reconstruction_coeff=0.2, n_steps=n_steps)
    for metrics in per_step:
        assert all(bool(jnp.isfinite(v).all()) for v in metrics.values())
    assert int(state.training.step) == n_steps
    adv = state.training.adversaries[ppgd_key]
    assert float(adv.opt_state.step_count) == n_steps * (n_warmup + 1)
    for v in jax.tree.leaves(adv.sources):
        assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0


def test_persistent_warmup_ascends_the_combined_objective():
    """Metrics are scored after warmup but before the final state update. With a constant
    nonzero source LR, changing only the auxiliary coefficient must therefore change the
    warmed-source e2e value if warmup ascends the combined objective."""
    _, low = _ppgd_run(
        hidden_acts_reconstruction_coeff=1e-12,
        n_steps=1,
        n_warmup=2,
        constant_source_lr=True,
    )
    _, high = _ppgd_run(
        hidden_acts_reconstruction_coeff=100.0,
        n_steps=1,
        n_warmup=2,
        constant_source_lr=True,
    )
    key = "loss/PersistentPGDReconLoss/e2e"
    assert not bool(jnp.array_equal(low[0][key], high[0][key]))


def test_adversary_actually_ascends_the_combined_objective():
    """With no warmup and one train step, the two arms ascend against identical model/CI
    states; differing source moments therefore isolate the auxiliary term's final-ascent
    gradient. Compare moments rather than sources because Adam's first normalized update can
    produce the same projected source values from different gradient magnitudes."""
    baseline, _ = _ppgd_run(hidden_acts_reconstruction_coeff=1.0, n_steps=1, n_warmup=0)
    pulled, _ = _ppgd_run(hidden_acts_reconstruction_coeff=50.0, n_steps=1, n_warmup=0)
    key = "PersistentPGDReconLoss"
    base_m = baseline.training.adversaries[key].opt_state.m
    pulled_m = pulled.training.adversaries[key].opt_state.m
    assert set(base_m) == set(pulled_m)
    changed = [
        site
        for site in base_m
        if any(
            not bool(jnp.array_equal(a, b))
            for a, b in zip(
                jax.tree.leaves(base_m[site]), jax.tree.leaves(pulled_m[site]), strict=True
            )
        )
    ]
    assert changed, (
        "source moments are identical at hidden-activation reconstruction coeff 1 and 50, so the term "
        "is not reaching the final-ascent gradient"
    )
