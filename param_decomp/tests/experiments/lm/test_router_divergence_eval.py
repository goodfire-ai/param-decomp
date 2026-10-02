"""The `RouterDivergence` eval: its pure math on hand-built router states, its authored
config and the resolution of `persistent` strategies against a run's training terms, and
the real step on the tiny qwen36_moe target (CPU, unplaced).

The pure tests pin the three distances against hand-computed values (and the identity: a
masked router equal to the target's scores kl 0, overlap 1, mae 0), the token-weighted fold
across batches, and the log-key spelling. The end-to-end tests pin the conditioning seam
the eval stands on: every strategy traces and returns finite sums; a masked forward whose
masks reproduce the frozen weights reproduces the target's router; the masked
`router_weights` tap IS the verb applied at the conditioning's selection; a layer whose
router reads a residual upstream of every decomposed site diverges by exactly zero — the
read-out at layer `l` depends only on the sites before it; and a batch-indexed persistent
source reaches an eval batch of another size through `sample_source_rows`.
"""

import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from jax import random
from jaxtyping import TypeCheckError
from pydantic import ValidationError

from param_decomp.core.adversary import (
    SourceStacks,
    init_persistent_sources,
    source_values_to_float,
)
from param_decomp.core.components import (
    SiteC,
    init_component_stacks,
    map_site_ci,
    site_ci_values,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    BatchSourceShape,
    FaithfulnessLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
)
from param_decomp.core.masking import sample_source_rows
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.config import LMExperimentConfig
from param_decomp.experiments.lm.eval_config import (
    CIMaskedStrategy,
    FreshPGDStrategy,
    PersistentStrategy,
    RouterDivergenceConfig,
    RouterDivergenceStrategy,
    StochasticStrategy,
    assert_router_divergence_persistent_terms_exist,
)
from param_decomp.experiments.lm.router_divergence_eval import (
    RouterDivergenceAccumulation,
    RouterDivergenceSums,
    empty_router_divergence_sums,
    expert_router_model,
    fold_router_divergence,
    kl_divergence,
    make_router_divergence_step,
    router_divergence_log_entries,
    router_probs_capture_keys,
    topk_overlap,
    weight_mae,
)
from param_decomp.lm.batch import LMBatch
from param_decomp.targets.qwen36_moe import (
    expert_mixing_weights,
    full_site_cs,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    tiny_glu_cfg,
    tiny_glu_decomposed_lm,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn,
)
from param_decomp.targets.transformer import glu_site_specs

REPO = Path(__file__).resolve().parents[4]

# One token, E=4 experts, k=2: the target routes to experts 0 and 1; the masked router
# reverses the preference order, so its weights at the target's experts renormalize 0.1, 0.2.
TARGET_PROBS = np.array([0.5, 0.3, 0.15, 0.05], np.float32)
MASKED_PROBS = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
TARGET_WEIGHTS = np.array([0.5 / 0.8, 0.3 / 0.8], np.float32)
MASKED_WEIGHTS = np.array([0.1 / 0.3, 0.2 / 0.3], np.float32)

DISTANCES = tuple(field.name for field in fields(RouterDivergenceSums))

ALL_STRATEGIES = (
    CIMaskedStrategy(),
    StochasticStrategy(),
    FreshPGDStrategy(n_steps=2, step_size=0.1),
    PersistentStrategy(state_key="ppgd"),
)


# ───────────────────────────────── the pure math ─────────────────────────────────


def test_identical_routers_are_at_distance_zero_overlap_one() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 3, 6)).astype(np.float32)
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    indices = np.argsort(-probs, axis=-1, kind="stable")[..., :2].astype(np.int32)
    weights = np.take_along_axis(probs, indices, axis=-1)
    weights = weights / weights.sum(-1, keepdims=True)

    for value, expected in (
        (kl_divergence(jnp.asarray(probs), jnp.asarray(probs)), 0.0),
        (topk_overlap(jnp.asarray(indices), jnp.asarray(indices)), 1.0),
        (weight_mae(jnp.asarray(weights), jnp.asarray(weights)), 0.0),
    ):
        assert value.shape == (2, 3)
        np.testing.assert_allclose(np.asarray(value), expected, atol=1e-6)


def test_kl_matches_the_hand_computed_sum() -> None:
    expected = (
        0.5 * math.log(0.5 / 0.1)
        + 0.3 * math.log(0.3 / 0.2)
        + 0.15 * math.log(0.15 / 0.3)
        + 0.05 * math.log(0.05 / 0.4)
    )
    kl = kl_divergence(jnp.asarray(TARGET_PROBS), jnp.asarray(MASKED_PROBS))
    assert kl.shape == () and kl.dtype == jnp.float32
    np.testing.assert_allclose(float(kl), expected, rtol=1e-5)
    assert float(kl_divergence(jnp.asarray(MASKED_PROBS), jnp.asarray(TARGET_PROBS))) != float(kl)


def test_kl_is_finite_when_the_masked_router_zeroes_a_target_expert() -> None:
    masked = np.array([0.0, 0.2, 0.4, 0.4], np.float32)
    kl = float(kl_divergence(jnp.asarray(TARGET_PROBS), jnp.asarray(masked)))
    assert math.isfinite(kl) and kl > 0.0


def test_topk_overlap_counts_the_intersection_regardless_of_order() -> None:
    target = jnp.asarray(np.array([[0, 1], [0, 1], [0, 1], [0, 1]], np.int32))
    masked = jnp.asarray(np.array([[3, 2], [1, 2], [2, 1], [1, 0]], np.int32))
    np.testing.assert_array_equal(np.asarray(topk_overlap(target, masked)), [0.0, 0.5, 0.5, 1.0])

    k3_target = jnp.asarray(np.array([4, 0, 2], np.int32))
    k3_masked = jnp.asarray(np.array([2, 5, 4], np.int32))
    np.testing.assert_allclose(float(topk_overlap(k3_target, k3_masked)), 2 / 3, rtol=1e-6)


def test_weight_mae_matches_the_hand_computed_mean() -> None:
    expected = (abs(0.625 - 1 / 3) + abs(0.375 - 2 / 3)) / 2
    mae = weight_mae(jnp.asarray(TARGET_WEIGHTS), jnp.asarray(MASKED_WEIGHTS))
    assert mae.shape == () and mae.dtype == jnp.float32
    np.testing.assert_allclose(float(mae), expected, rtol=1e-5)


def test_distances_refuse_mismatched_ranks_and_dtypes() -> None:
    with pytest.raises(TypeCheckError):
        kl_divergence(jnp.zeros((2, 4)), jnp.zeros((2, 5)))
    with pytest.raises(TypeCheckError):
        topk_overlap(jnp.zeros((2, 2), jnp.float32), jnp.zeros((2, 2), jnp.int32))
    with pytest.raises(TypeCheckError):
        weight_mae(jnp.zeros((2, 2)), jnp.zeros((3, 2)))


def test_fold_is_token_weighted_across_batches() -> None:
    n_layers = 2
    batch_a = RouterDivergenceSums(
        kl=jnp.asarray([2.0, 8.0]),
        topk_overlap=jnp.asarray([4.0, 2.0]),
        weight_mae=jnp.asarray([1.0, 0.0]),
    )
    batch_b = RouterDivergenceSums(
        kl=jnp.asarray([12.0, 0.0]),
        topk_overlap=jnp.asarray([12.0, 6.0]),
        weight_mae=jnp.asarray([3.0, 12.0]),
    )
    folded = fold_router_divergence(
        fold_router_divergence(empty_router_divergence_sums(n_layers), batch_a, 4), batch_b, 12
    )
    assert folded.n_tokens == 16
    np.testing.assert_array_equal(folded.sums.kl, [14.0, 8.0])

    entries = router_divergence_log_entries(StochasticStrategy(), (0, 3), folded)
    # Token-weighted: (2 + 12) / 16 and (8 + 0) / 16 — not the means of the per-batch
    # means, (0.5 + 1.0) / 2 and (2.0 + 0.0) / 2.
    assert entries["eval/router_divergence/stochastic/kl/layer_0"] == 0.875
    assert entries["eval/router_divergence/stochastic/kl/layer_3"] == 0.5
    assert entries["eval/router_divergence/stochastic/kl"] == 0.6875
    assert entries["eval/router_divergence/stochastic/topk_overlap/layer_3"] == 0.5
    assert entries["eval/router_divergence/stochastic/weight_mae/layer_0"] == 0.25
    assert entries["eval/router_divergence/stochastic/weight_mae"] == 0.5
    assert set(entries) == {
        f"eval/router_divergence/stochastic/{distance}{suffix}"
        for distance in DISTANCES
        for suffix in ("", "/layer_0", "/layer_3")
    }


def test_log_entries_refuse_an_empty_accumulator_and_a_layer_count_mismatch() -> None:
    with pytest.raises(AssertionError, match="no router-divergence data"):
        router_divergence_log_entries(CIMaskedStrategy(), (0,), empty_router_divergence_sums(1))
    accumulated = RouterDivergenceAccumulation(
        sums=RouterDivergenceSums(kl=np.ones(2), topk_overlap=np.ones(2), weight_mae=np.ones(2)),
        n_tokens=1,
    )
    with pytest.raises(ValueError):
        router_divergence_log_entries(CIMaskedStrategy(), (0, 1, 2), accumulated)


# ──────────────────────────────── the authored config ────────────────────────────────


def test_config_parses_the_closed_strategy_union_from_yaml_shape() -> None:
    metric = RouterDivergenceConfig.model_validate(
        {
            "type": "RouterDivergence",
            "strategies": [
                {"kind": "ci_masked"},
                {"kind": "stochastic"},
                {"kind": "fresh_pgd", "n_steps": 20, "step_size": 0.1},
                {"kind": "persistent", "state_key": "PersistentPGDReconLoss"},
            ],
        }
    )
    assert metric.strategies == (
        CIMaskedStrategy(),
        StochasticStrategy(),
        FreshPGDStrategy(n_steps=20, step_size=0.1),
        PersistentStrategy(state_key="PersistentPGDReconLoss"),
    )
    assert not RouterDivergenceConfig.slow
    eval_config = EvalConfig(batch_size=8, n_steps=1, every=100, slow_every=200, metrics=[metric])
    assert eval_config.metrics == [metric]


def test_config_refuses_no_strategies_and_repeated_kinds() -> None:
    with pytest.raises(ValidationError, match="at least one strategy"):
        RouterDivergenceConfig(strategies=())
    with pytest.raises(ValidationError, match="repeat a kind"):
        RouterDivergenceConfig(strategies=(StochasticStrategy(), StochasticStrategy()))
    with pytest.raises(ValidationError):
        RouterDivergenceConfig.model_validate(
            {"type": "RouterDivergence", "strategies": [{"kind": "dense"}]}
        )


def _persistent_term(
    source_shape: BatchSourceShape, name: str | None
) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=1.0,
        name=name,
        optimizer=AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.1)),
        source_shape=source_shape,
    )


@pytest.mark.parametrize("source_shape", ["bc", "bsc"])
def test_persistent_strategy_resolves_a_term_of_any_source_shape_by_instance_key(
    source_shape: BatchSourceShape,
) -> None:
    """The strategy draws one training row per eval sequence for either source shape."""
    metric = RouterDivergenceConfig(strategies=(PersistentStrategy(state_key="adv"),))
    assert_router_divergence_persistent_terms_exist(
        metric, [FaithfulnessLossConfig(coeff=1.0), _persistent_term(source_shape, "adv")]
    )
    by_type = RouterDivergenceConfig(
        strategies=(PersistentStrategy(state_key="MergedStochasticSubsetPPGDReconLoss"),)
    )
    assert_router_divergence_persistent_terms_exist(
        by_type,
        [
            MergedStochasticSubsetPPGDReconLossConfig(
                coeff=1.0,
                adv_fraction=ScheduleConfig.constant(0.5),
                optimizer=AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.1)),
                source_shape=source_shape,
            )
        ],
    )


def test_persistent_strategy_refuses_a_state_key_the_run_does_not_train() -> None:
    metric = RouterDivergenceConfig(strategies=(PersistentStrategy(state_key="adv"),))
    with pytest.raises(AssertionError, match=r"state_key 'adv'.*\['other'\]"):
        assert_router_divergence_persistent_terms_exist(metric, [_persistent_term("bc", "other")])
    without_persistent = RouterDivergenceConfig(
        strategies=(CIMaskedStrategy(), StochasticStrategy())
    )
    assert_router_divergence_persistent_terms_exist(without_persistent, [])


def _seat_with_router_divergence(seat: str, strategy: dict[str, Any]) -> dict[str, Any]:
    raw = yaml.safe_load((REPO / "param_decomp/experiments/lm/configs" / seat).read_text())
    raw["eval"]["metrics"].append({"type": "RouterDivergence", "strategies": [strategy]})
    return raw


def test_lm_run_shape_resolves_persistent_strategies_against_its_own_terms() -> None:
    """Both batch-shaped variants resolve their persistent term by state key."""
    persistent = {"kind": "persistent", "state_key": "PersistentPGDReconLoss"}
    for source_shape in ("bc", "bsc"):
        raw = _seat_with_router_divergence("qwen3_6_35b_a3b.yaml", persistent)
        source = next(
            loss for loss in raw["pd"]["loss_metrics"] if loss["type"] == "PersistentPGDReconLoss"
        )
        source["source_shape"] = source_shape
        LMExperimentConfig.model_validate(raw)
    with pytest.raises(ValidationError, match=r"state_key 'other'"):
        LMExperimentConfig.model_validate(
            _seat_with_router_divergence(
                "qwen3_6_35b_a3b.yaml",
                {"kind": "persistent", "state_key": "other"},
            )
        )


# ───────────────────────────── the step on tiny qwen36 ─────────────────────────────

BATCH, SEQ = 2, 6
# A batch-indexed source's stored row count — deliberately not the eval batch's.
N_TRAIN = 3


class _TinyQwen36:
    """The shared batch-context values the step consumes, prepared eagerly (unplaced)."""

    def __init__(self) -> None:
        cfg = tiny_qwen36_cfg()
        self.cfg = cfg
        self.sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, TINY_QWEN36_CS))
        model = tiny_qwen36_decomposed_model(cfg, self.sites, random.PRNGKey(0))
        self.placed = PlacedModel(model=model, placement=None)
        self.components = init_component_stacks(self.sites, random.PRNGKey(1))
        ci_fn = tiny_qwen36_moe_ci_fn(model, random.PRNGKey(2))
        self.tokens = random.randint(random.PRNGKey(3), (BATCH, SEQ), 0, cfg.vocab_size)
        self.probs_keys = router_probs_capture_keys(self.placed)
        clean = self.placed.clean_forward(
            LMBatch(self.tokens),
            ci_fn.capture_keys | frozenset(self.probs_keys),
        )
        self.conditioning = clean.conditioning
        self.clean_output = clean.output
        self.clean_probs = {key: clean.captures[key] for key in self.probs_keys}
        self.ci_lower = ci_fn.prepare()(
            {key: clean.captures[key] for key in ci_fn.capture_keys},
            clean.conditioning,
            self.placed.prepare_compute_weights(self.components),
            sequence=clean.sequence,
            remat=False,
        ).lower
        self.prepared_weights = self.placed.prepare_compute_weights(self.components)

    def persistent_sources(self, leading: tuple[int, int]) -> dict[str, SourceStacks]:
        """`leading` spells the `source_shape` over a positioned target's `(B, T)`: `(1, 1)`
        for `c`, `(N_TRAIN, 1)` for a `bc` term over `N_TRAIN` training samples."""
        return {
            "ppgd": init_persistent_sources(self.sites, leading, jnp.float32, random.PRNGKey(7))
        }

    def run_step(
        self,
        strategies: tuple[RouterDivergenceStrategy, ...],
        persistent_sources: dict[str, SourceStacks],
    ) -> dict[str, tuple[RouterDivergenceSums, int]]:
        metric = RouterDivergenceConfig(strategies=strategies)
        step = jax.jit(
            make_router_divergence_step(
                self.placed,
                metric,
                None,
            )
        )
        return step(
            self.placed,
            self.prepared_weights,
            self.conditioning,
            self.ci_lower,
            self.clean_probs,
            self.clean_output,
            persistent_sources,
            random.PRNGKey(9),
        )


@pytest.fixture(scope="module")
def tiny() -> _TinyQwen36:
    return _TinyQwen36()


def _assert_finite_layer_sums(
    sums: RouterDivergenceSums, n_layers: int, n_tokens: int, k: int
) -> None:
    for distance in DISTANCES:
        value = np.asarray(getattr(sums, distance))
        assert value.shape == (n_layers,) and np.isfinite(value).all(), distance
        assert (value >= 0).all(), distance
    assert (np.asarray(sums.topk_overlap) <= n_tokens).all()
    # Both weight vectors sum to 1, so a token's mean absolute error is at most 2/k.
    assert (np.asarray(sums.weight_mae) <= 2 * n_tokens / k).all()


def test_every_strategy_traces_and_sums_finite_per_layer_distances(tiny: _TinyQwen36) -> None:
    results = tiny.run_step(ALL_STRATEGIES, tiny.persistent_sources((1, 1)))

    assert results.keys() == {strategy.kind for strategy in ALL_STRATEGIES}
    n_layers = tiny.cfg.n_layer
    for strategy in ALL_STRATEGIES:
        sums, n_tokens = results[strategy.kind]
        assert n_tokens == BATCH * SEQ, strategy
        _assert_finite_layer_sums(sums, n_layers, n_tokens, tiny.cfg.n_experts_per_token)
        # Layer 0's router reads the residual before any decomposed site (the tiny target
        # decomposes MoE matrices only), so under the target's routing it cannot diverge —
        # the read-out at layer `l` depends only on the sites upstream of it.
        assert float(sums.kl[0]) == 0.0 and float(sums.weight_mae[0]) == 0.0, strategy
        assert float(sums.topk_overlap[0]) == n_tokens, strategy
        assert float(sums.kl[-1]) > 0.0, strategy

    entries = router_divergence_log_entries(
        ALL_STRATEGIES[1],
        expert_router_model(tiny.placed).expert_router_layers,
        fold_router_divergence(empty_router_divergence_sums(n_layers), *results["stochastic"]),
    )
    assert entries["eval/router_divergence/stochastic/topk_overlap/layer_0"] == 1.0
    assert 0.0 < entries["eval/router_divergence/stochastic/kl"] < math.inf


def test_persistent_strategy_reads_a_batch_indexed_source_at_the_eval_batch_size(
    tiny: _TinyQwen36,
) -> None:
    """A `bc` term stores one attack per TRAINING sample; an eval batch of another size
    draws one training row per sequence, so the strategy traces over the eval batch and
    sums over exactly its tokens."""
    results = tiny.run_step(
        (PersistentStrategy(state_key="ppgd"),), tiny.persistent_sources((N_TRAIN, 1))
    )
    assert results.keys() == {"persistent"}
    sums, n_tokens = results["persistent"]
    assert n_tokens == BATCH * SEQ
    _assert_finite_layer_sums(sums, tiny.cfg.n_layer, n_tokens, tiny.cfg.n_experts_per_token)


def test_sample_source_rows_gathers_whole_training_rows(tiny: _TinyQwen36) -> None:
    """Every sampled row IS one training row — the same row at every site, components and
    delta alike — and a batch-free source passes through as the same objects."""
    sources = source_values_to_float(tiny.persistent_sources((N_TRAIN, 1))["ppgd"]).per_site()
    sampled = sample_source_rows(random.PRNGKey(5), tiny.ci_lower, sources)
    assert sampled.keys() == sources.keys()

    # Recover each eval row's training row from one site's delta: distinct U[0,1] draws,
    # gathered exactly, so equality identifies the row.
    first = next(iter(sources))
    training_deltas = np.asarray(sources[first].delta)
    sampled_deltas = np.asarray(sampled[first].delta)
    assert sampled_deltas.shape == (BATCH, 1)
    rows = []
    for eval_row in range(BATCH):
        (matches,) = np.nonzero((training_deltas == sampled_deltas[eval_row]).all(axis=1))
        assert matches.shape == (1,), matches
        rows.append(int(matches[0]))
    for site, source in sources.items():
        expected = jax.tree.map(lambda leaf: leaf[jnp.asarray(rows)], source)
        assert jax.tree.structure(sampled[site]) == jax.tree.structure(expected), site
        for got, want in zip(
            jax.tree.leaves(sampled[site]), jax.tree.leaves(expected), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    batch_free = source_values_to_float(tiny.persistent_sources((1, 1))["ppgd"]).per_site()
    passed = sample_source_rows(random.PRNGKey(5), tiny.ci_lower, batch_free)
    assert all(passed[site] is batch_free[site] for site in batch_free)


def test_masks_reproducing_the_frozen_weights_reproduce_the_target_router(
    tiny: _TinyQwen36,
) -> None:
    """`mask ≡ 1` with the weight-delta mask `≡ 1` makes every site `x @ W` again; under
    the conditioning's selection the masked router then matches the target's at every
    layer, and the masked `router_weights` tap IS `expert_mixing_weights` at the
    conditioning's indices."""
    ones = {site: map_site_ci(jnp.ones_like, ci) for site, ci in tiny.ci_lower.items()}
    deltas = {
        site: jnp.ones(site_ci_values(ci).shape[:-1], site_ci_values(ci).dtype)
        for site, ci in tiny.ci_lower.items()
    }
    router_model = expert_router_model(tiny.placed)
    weights_keys = [
        router_model.router_weights_capture_key(layer)
        for layer in router_model.expert_router_layers
    ]
    captures = tiny.placed.masked_forward(
        tiny.prepared_weights,
        tiny.conditioning,
        masking=tiny.placed.model.prepare_masking(
            MaterializedMasking(component_masks=ones, weight_delta_masks=deltas)
        ),
        routes=None,
        capture_keys=frozenset(tiny.probs_keys) | frozenset(weights_keys),
        remat=False,
    ).captures

    selection = tiny.conditioning.selection
    for row, (probs_key, weights_key) in enumerate(zip(tiny.probs_keys, weights_keys, strict=True)):
        target_probs, target_indices, target_weights = (
            tiny.clean_probs[probs_key],
            selection.indices[row],
            selection.weights[row],
        )
        masked_probs, masked_weights = captures[probs_key], captures[weights_key]
        masked_indices = router_model.select_experts(masked_probs)
        assert masked_probs.dtype == jnp.float32 and masked_weights.dtype == jnp.float32
        np.testing.assert_allclose(
            np.asarray(masked_weights),
            np.asarray(expert_mixing_weights(masked_probs, target_indices)),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(kl_divergence(target_probs, masked_probs)), 0.0, atol=1e-5
        )
        np.testing.assert_array_equal(np.asarray(topk_overlap(target_indices, masked_indices)), 1.0)
        np.testing.assert_allclose(
            np.asarray(weight_mae(target_weights, masked_weights)), 0.0, atol=1e-5
        )


def test_a_target_without_an_expert_router_refuses_at_step_build() -> None:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, (SiteC("layers.0.self_attn.q_proj", 4),))
    glu = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    metric = RouterDivergenceConfig(strategies=(CIMaskedStrategy(),))
    with pytest.raises(AssertionError, match="ExpertRouterModel"):
        jax.jit(
            make_router_divergence_step(
                glu,
                metric,
                None,
            )
        )
    with pytest.raises(AssertionError, match="ExpertRouterModel"):
        router_probs_capture_keys(glu)


@pytest.mark.multidevice
def test_persistent_rows_preserve_selected_expert_masks():
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from param_decomp.core.adversary import BlockedSourceComponents, SiteSource
    from param_decomp.core.components import SelectedCI
    from param_decomp.core.masking import materialize_masking, source_masking

    if jax.device_count() < 4:
        pytest.skip("requires four devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    indices = np.array([[[0, 1], [2, 3]], [[1, 3], [0, 2]]], np.int32)
    table = np.arange(32, dtype=np.float32).reshape(4, 1, 4, 2) / 32
    deltas = np.arange(4, dtype=np.float32).reshape(4, 1) / 4

    def sample(
        ids: jax.Array,
        values: jax.Array,
        dense_ci: jax.Array,
        blocked_table: jax.Array,
        dense_table: jax.Array,
        delta: jax.Array,
    ):
        selected = SelectedCI(values.reshape(*values.shape[:-2], -1), ids, 4)
        sources = {
            "expert": SiteSource(components=BlockedSourceComponents(blocked_table), delta=delta),
            "dense": SiteSource(components=dense_table, delta=delta),
        }
        cis = {"expert": selected, "dense": dense_ci}
        sampled = sample_source_rows(random.PRNGKey(17), cis, sources)
        masking = materialize_masking(source_masking(cis, sampled))
        expert_mask = masking.component_masks["expert"]
        assert isinstance(expert_mask, SelectedCI)
        return (
            sampled,
            expert_mask,
            expert_mask.values.reshape(values.shape),
            masking.component_masks["dense"],
            masking.weight_delta_masks,
        )

    def placed(value: np.ndarray, spec: P) -> jax.Array:
        return jax.device_put(value, NamedSharding(mesh, spec))

    with jax.set_mesh(mesh):
        sampled, expert_mask, token_masks, dense_mask, delta_masks = jax.jit(sample)(
            placed(indices, P("data", None, None)),
            placed(np.full((2, 2, 2, 2), 0.25, np.float32), P("data", None, None, None)),
            placed(np.full((2, 2, 8), 0.25, np.float32), P("data", None, "tp")),
            placed(table, P("data", None, "tp", None)),
            placed(table.reshape(4, 1, 8), P("data", None, "tp")),
            placed(deltas, P("data", None)),
        )
    rows = (np.asarray(sampled["expert"].delta)[:, 0] * 4).astype(int)
    np.testing.assert_array_equal(sampled["dense"].components, table[rows].reshape(2, 1, 8))
    np.testing.assert_array_equal(delta_masks["expert"], deltas[rows])
    np.testing.assert_array_equal(delta_masks["dense"], deltas[rows])
    selected = table[rows, 0][np.arange(2)[:, None, None], indices]
    np.testing.assert_allclose(token_masks, 0.25 + 0.75 * selected)
    np.testing.assert_allclose(
        dense_mask, np.broadcast_to(0.25 + 0.75 * table[rows].reshape(2, 1, 8), (2, 2, 8))
    )
    assert jax.typeof(expert_mask.values).sharding.spec == P("data", None, None)
    blocked = sampled["expert"].components
    assert isinstance(blocked, BlockedSourceComponents)
    assert jax.typeof(blocked.values).sharding.spec == P("data", None, "tp", None)
