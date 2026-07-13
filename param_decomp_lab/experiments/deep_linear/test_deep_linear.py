"""CPU tests for the deep-linear identity target over the generic positionless core.

Covers the `DecomposedModel` contract (clean == frozen passthrough of one-hot inputs,
masked identity, KL recon), chunkwise plan construction over the homogeneous sites, the
full SPEC step with `ChunkwiseSubsetReconLoss` (the first positionless target to
exercise it), the loud rejection of the seq-axis-pinned persistent-PGD adversary, and an
end-to-end decompose → recovers-identity validation on a tiny d=6 stack.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
import yaml

from param_decomp.ci_fn import MLPCIArch, init_layerwise_mlp_ci_fn
from param_decomp.components import SiteC, SiteSpec, init_decomp_vu
from param_decomp.configs import (
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.losses import kl_per_position
from param_decomp.recon import ReconLossTerm, build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.train import TrainState, make_faith_warmup_step, make_train_step
from param_decomp_lab.experiments.deep_linear.config import DeepLinearExperimentConfig
from param_decomp_lab.experiments.deep_linear.model import (
    DeepLinearConfig,
    canonical_site_cs,
    deep_linear_decomposed_model,
    expand_wildcard_site_cs,
    init_deep_linear_target,
    one_hot_probe,
    sample_one_hot,
    site_inputs,
    site_specs,
)
from param_decomp_lab.experiments.deep_linear.run import build_deep_linear_built_run
from param_decomp_lab.experiments.tms.model import identity_ci_error

CONFIGS_DIR = Path(__file__).parent / "configs"


def _cfg(n_features: int = 6, n_layers: int = 4) -> DeepLinearConfig:
    return DeepLinearConfig(n_features=n_features, n_layers=n_layers)


def _site_cs(n_layers: int = 4, c: int = 12) -> tuple[SiteC, ...]:
    return tuple(SiteC(f"layers.{i}", c) for i in range(n_layers))


def test_canonical_order_and_dims():
    cfg = _cfg()
    shuffled = (_site_cs()[2], _site_cs()[0], _site_cs()[3], _site_cs()[1])
    assert canonical_site_cs(shuffled) == _site_cs()
    with pytest.raises(AssertionError):
        canonical_site_cs(_site_cs()[:2] + _site_cs()[:1])  # duplicate site
    specs = site_specs(cfg, _site_cs())
    assert all((s.d_in, s.d_out, s.C) == (6, 6, 12) for s in specs)


def test_wildcard_expansion():
    assert expand_wildcard_site_cs((SiteC("layers.*", 12),), 4) == _site_cs()
    # explicit names pass through; count must match n_layers
    assert expand_wildcard_site_cs(_site_cs(), 4) == _site_cs()
    with pytest.raises(AssertionError):
        expand_wildcard_site_cs((SiteC("layers.*", 12),) + _site_cs()[:1], 4)


def test_clean_output_is_one_hot_passthrough():
    cfg = _cfg()
    target = init_deep_linear_target(cfg)
    lm = deep_linear_decomposed_model(cfg, target, _site_cs())
    x = sample_one_hot(jax.random.PRNGKey(0), 16, cfg.n_features)
    clean = lm.clean_output(x)
    assert jnp.array_equal(clean, x), "identity stack must pass one-hot inputs through"
    # every site reads the same one-hot activation
    site_in = site_inputs(target, x)
    assert all(jnp.array_equal(v, x) for v in site_in.values())


def test_masked_identity_and_ablation():
    cfg = _cfg()
    sites = site_specs(cfg, _site_cs())
    target = init_deep_linear_target(cfg)
    lm = deep_linear_decomposed_model(cfg, target, _site_cs())
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    b = 7
    x = sample_one_hot(jax.random.PRNGKey(2), b, cfg.n_features)
    clean = lm.clean_output(x)

    # SPEC S2: live=() is the exact frozen path.
    none_masked = lm.masked_output(vu, x, {}, {}, None, (), True, remat=False)
    assert jnp.array_equal(clean, none_masked)

    # All-live, masks=1, delta=1 reconstructs the frozen path up to decomposition rounding.
    names = lm.site_names
    ones_masks = {s.name: jnp.ones((b, s.C)) for s in lm.sites}
    ones_delta = {s: jnp.ones((b,)) for s in names}
    full = lm.masked_output(vu, x, ones_masks, ones_delta, None, names, True, remat=False)
    assert jnp.allclose(clean, full, atol=1e-4)

    # Zero-masking one site changes the output.
    C = lm.sites[0].C
    ablated = lm.masked_output(
        vu, x, {"layers.0": jnp.zeros((b, C))}, {"layers.0": jnp.zeros((b,))},
        None, ("layers.0",), True, remat=False,
    )  # fmt: skip
    assert not jnp.allclose(clean, ablated, atol=1e-5)

    deltas = lm.weight_deltas(vu)
    assert all(v.shape == (cfg.n_features, cfg.n_features) for v in deltas.values())
    assert all(v.dtype == jnp.float32 for v in deltas.values())


def test_recon_loss_fn_is_kl():
    cfg = _cfg()
    lm = deep_linear_decomposed_model(cfg, init_deep_linear_target(cfg), _site_cs())
    a = jax.random.normal(jax.random.PRNGKey(0), (4, cfg.n_features))
    b = jax.random.normal(jax.random.PRNGKey(1), (4, cfg.n_features))
    assert jnp.array_equal(lm.recon_loss_fn(a, b), kl_per_position(a, b))
    assert float(lm.recon_loss_fn(a, a)) == pytest.approx(0.0, abs=1e-6)
    assert float(lm.recon_loss_fn(a, b)) > 0.0


@pytest.mark.parametrize(("sites_per_chunk", "n_chunks"), [(1, 4), (2, 2), (4, 1)])
def test_chunkwise_plan_over_layers(sites_per_chunk: int, n_chunks: int):
    names = tuple(f"layers.{i}" for i in range(4))
    terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1e-3, pnorm=ScheduleConfig(start_val=1.0, fn_type="constant")
            ),
            ChunkwiseSubsetReconLossConfig(coeff=1.0, sites_per_chunk=sites_per_chunk),
        ),
        names,
        100,
    )
    (term,) = terms.recon
    assert isinstance(term, ReconLossTerm)
    assert len(term.plan) == n_chunks
    covered = tuple(site for entry in term.plan for site in entry.live_sites)
    assert covered == names, "chunks must tile the sites in canonical order"


def _loss_metrics(sites_per_chunk: int):
    return (
        FaithfulnessLossConfig(coeff=1.0),
        ImportanceMinimalityLossConfig(
            coeff=1e-3, pnorm=ScheduleConfig(start_val=1.0, fn_type="constant")
        ),
        ChunkwiseSubsetReconLossConfig(coeff=1.0, sites_per_chunk=sites_per_chunk),
    )


def _state_and_step(
    cfg: DeepLinearConfig,
    sites: tuple[SiteSpec, ...],
    sites_per_chunk: int,
    total_steps: int,
    warmup_steps: int = 0,
):
    lm = deep_linear_decomposed_model(
        cfg, init_deep_linear_target(cfg), _site_cs(cfg.n_layers, sites[0].C)
    )
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    if warmup_steps:
        warm_opt = optax.adamw(1e-2, weight_decay=0.0)
        wstep = make_faith_warmup_step(warm_opt)
        warm_state = warm_opt.init(eqx.filter(vu, eqx.is_array))
        for _ in range(warmup_steps):
            vu, warm_state, _ = wstep(lm, vu, warm_state)
    ci_fn = init_layerwise_mlp_ci_fn(MLPCIArch(hidden_dims=(16,)), sites, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    loss_terms = build_loss_terms(_loss_metrics(sites_per_chunk), lm.site_names, total_steps)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        adversaries={}, step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    step = make_train_step(
        lm=lm, losses=loss_terms, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=total_steps, remat_recon_forwards=False, remat_ci_fn=False, mesh=None,
    )  # fmt: skip
    return lm, state, step, loss_terms


@pytest.mark.parametrize("sites_per_chunk", [1, 4])
def test_step_trains_chunkwise_stochastic(sites_per_chunk: int):
    cfg = _cfg()
    sites = site_specs(cfg, _site_cs())
    lm, state, step, _ = _state_and_step(cfg, sites, sites_per_chunk, total_steps=20)
    for i in range(6):
        x = sample_one_hot(jax.random.fold_in(jax.random.PRNGKey(99), i), 64, cfg.n_features)
        state, m = step(lm, state, x, jax.random.PRNGKey(100 + i))
        assert all(bool(jnp.isfinite(v).all()) for v in m.values()), m
    assert int(state.step) == 6
    assert state.adversaries == {}


def test_persistent_pgd_rejected_loudly():
    """The engine pins persistent sources to a sequence axis (`init_train_state` requires
    a `DataConfig`); the deep-linear build must reject the adversary up front, not at
    engine init."""
    raw = yaml.safe_load((CONFIGS_DIR / "deep_linear_30.yaml").read_text())
    raw["pd"]["loss_metrics"].append(
        {
            "type": "PersistentPGDReconLoss",
            "coeff": 0.5,
            "optimizer": {
                "type": "adam",
                "lr_schedule": {"start_val": 0.01, "fn_type": "constant"},
            },
            "scope": {"type": "bsc"},
        }
    )
    with pytest.raises(AssertionError, match="sequence axis"):
        build_deep_linear_built_run(DeepLinearExperimentConfig(**raw), "p-00000000")


@pytest.mark.parametrize("config_name", ["deep_linear_30.yaml", "deep_linear_128.yaml"])
def test_repo_configs_parse_and_build(config_name: str):
    raw = yaml.safe_load((CONFIGS_DIR / config_name).read_text())
    built = build_deep_linear_built_run(DeepLinearExperimentConfig(**raw), "p-00000000")
    assert built.data is None and built.eval is None
    assert len(built.target.sites) == 5


@pytest.mark.slow
def test_end_to_end_decompose_recovers_identity():
    """The recovers-structure gate: decompose a d=6, 2-layer identity stack with the
    chunkwise stochastic recon (one chunk per layer) and show the recovered per-site CI
    of the one-hot probe is the identity up to permutation — zero `identity_ci_error`."""
    cfg = _cfg(n_features=6, n_layers=2)
    site_cs = _site_cs(n_layers=2, c=12)
    sites = site_specs(cfg, site_cs)
    lm, state, step, _ = _state_and_step(
        cfg, sites, sites_per_chunk=1, total_steps=3000, warmup_steps=200
    )
    for i in range(3000):
        x = sample_one_hot(jax.random.fold_in(jax.random.PRNGKey(5), i), 512, cfg.n_features)
        state, _ = step(lm, state, x, jax.random.fold_in(jax.random.PRNGKey(6), i))

    probe = one_hot_probe(cfg.n_features)
    ci = state.ci_fn(lm.read_activations(probe, state.ci_fn.input_names), remat=False)
    errors = {site: identity_ci_error(v, tolerance=0.2) for site, v in ci.lower.items()}
    assert all(e == 0 for e in errors.values()), errors
