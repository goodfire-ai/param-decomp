"""Round-trip + resume-continuation tests for `checkpoint.py` (orbax) on the generic
trainer state (SPEC S22): a restored `TrainState` must continue the EXACT trajectory —
including the persistent adversary's sources and Adam moments."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh

from param_decomp.adversary import (
    init_persistent_sources,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp.checkpoint import (
    make_checkpoint_manager,
    restore_latest,
    restore_step,
    save_state,
)
from param_decomp.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    build_ci_fn,
)
from param_decomp.components import init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.lm import DecomposedModel
from param_decomp.recon import build_recon_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.sharding import dp_mesh
from param_decomp.targets.llama8b import (
    llama_site_specs,
    mlp_family_site_cs,
)
from param_decomp.targets.llama8b_sharding import (
    init_ci_fn_placed,
    init_decomp_vu_placed,
    init_sources_sharded,
)
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.train import TrainState, make_train_step
from vendored_jax.llama import LlamaConfig


def _chunkwise_arch(lm: DecomposedModel, cfg: LlamaConfig) -> ChunkwiseTransformerCIArch:
    """The old `CIArch(16, 2, 2, 32)` → one chunk reading the residual entering the first
    decomposed block and emitting CI for every site; `input_dim` is the residual width."""
    site_names = lm.site_names
    first_block = min(int(n.split(".")[1]) for n in site_names)
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
    )


def _build(seed: int):
    cfg = _tiny_cfg()
    C, seq = 8, 16
    sites = llama_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(seed))
    ci_fn = build_ci_fn(_chunkwise_arch(lm, cfg), lm.sites, jax.random.PRNGKey(seed + 1))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    src = init_persistent_sources(
        lm.site_names, tuple(s.C for s in lm.sites), (1, seq), jax.random.PRNGKey(seed + 2)
    )
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={"PersistentPGDReconLoss": src},
        sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(src)},
        step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_spec = build_recon_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6, pnorm=2.0, beta=0.2,
                p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0,
            ),
            ChunkwiseSubsetReconLossConfig(routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1),
            PersistentPGDReconLossConfig(
                coeff=0.5,
                scope=SCScope(),
                optimizer=AdamPGDConfig(
                    beta1=0.5, beta2=0.99,
                    lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
                ),
                n_warmup_steps=1,
            ),
        ),
        lm.site_names,
        n_mask_samples=1,
    )  # fmt: skip
    step = make_train_step(
        lm=lm,
        loss_spec=loss_spec,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True, mesh=None,
    )  # fmt: skip
    resid = jax.random.normal(jax.random.PRNGKey(9), (2, seq, cfg.n_embd)) * 0.5
    return lm, state, step, resid


def test_roundtrip_and_exact_resume(tmp_path: Path):
    lm, state, step, resid = _build(seed=1)
    for i in range(2):
        state, _ = step(lm, state, resid, jax.random.PRNGKey(i))

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, state)

    # Restore onto a DIFFERENTLY-seeded reference: every leaf must come from disk.
    _, fresh, _, _ = _build(seed=7)
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, ckpt_step = restored
    assert ckpt_step == 2
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(loaded), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))

    # SPEC S22: the restored state continues the exact trajectory.
    state_cont, m_cont = step(lm, state, resid, jax.random.PRNGKey(100))
    loaded_cont, m_load = step(lm, loaded, resid, jax.random.PRNGKey(100))
    for k in m_cont:
        assert float(m_cont[k]) == float(m_load[k]), k
    for a, b in zip(jax.tree.leaves(state_cont), jax.tree.leaves(loaded_cont), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))


def test_persistent_adam_step_count_roundtrip_and_post_resume_bias_correction(tmp_path: Path):
    """Issue #678 (matrix §8 + S22/S13/S23): after N persistent ascents, the orbax
    checkpoint must carry the adversary's `step_count` leaf (present, fp32, == N) and
    bit-equal Adam moments; the FIRST post-resume ascent must apply bias-correction for
    count N+1 (not N, not 1)."""
    state_key = "PersistentPGDReconLoss"
    beta1, beta2 = 0.5, 0.99

    lm, state, step, resid = _build(seed=1)
    for i in range(3):
        state, _ = step(lm, state, resid, jax.random.PRNGKey(i))

    pre_save = state.sources_opt_state[state_key]
    n_ascents = int(pre_save.step_count)
    # Each train step runs n_warmup_steps (1) supplemental ascents + 1 final ascent.
    assert n_ascents == 3 * (1 + 1)

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 3, state)

    _, fresh, _, _ = _build(seed=7)
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, _ = restored
    loaded_adam = loaded.sources_opt_state[state_key]

    # (a) the step_count leaf survived the round-trip: present, fp32 scalar, value N.
    assert state_key in loaded.sources_opt_state
    assert loaded_adam.step_count.dtype == jnp.float32
    assert loaded_adam.step_count.shape == ()
    assert float(loaded_adam.step_count) == float(n_ascents)

    # (c) the restored Adam moments are bit-equal to pre-save (per site, m and v).
    for site in pre_save.m:
        assert jnp.array_equal(loaded_adam.m[site], pre_save.m[site])
        assert jnp.array_equal(loaded_adam.v[site], pre_save.v[site])

    # (b) the first post-resume ascent applies bias-correction for count N+1.
    adam_cfg = AdamPGDConfig(
        beta1=beta1, beta2=beta2, lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)
    )
    grads = {site: jnp.ones_like(v) for site, v in loaded.sources[state_key].items()}
    _, post_resume = sources_adam_ascend_project(
        loaded.sources[state_key], grads, loaded_adam, jnp.asarray(0.01), adam_cfg
    )
    assert float(post_resume.step_count) == float(n_ascents + 1)
    expected_bc1 = 1.0 - beta1 ** (n_ascents + 1)
    expected_bc2 = 1.0 - beta2 ** (n_ascents + 1)
    actual_bc1 = 1.0 - beta1 ** float(post_resume.step_count)
    actual_bc2 = 1.0 - beta2 ** float(post_resume.step_count)
    assert abs(actual_bc1 - expected_bc1) < 1e-12
    assert abs(actual_bc2 - expected_bc2) < 1e-12
    # The N+1 denominator must differ from both the N and the count-1 alternatives.
    assert abs(expected_bc1 - (1.0 - beta1**n_ascents)) > 1e-9
    assert abs(expected_bc1 - (1.0 - beta1**1)) > 1e-9


def test_no_checkpoint_returns_none(tmp_path: Path):
    _, fresh, _, _ = _build(seed=7)
    mgr = make_checkpoint_manager(tmp_path / "empty", keep_last=2)
    assert restore_latest(mgr, fresh) is None


def _build_sharded(seed: int, mesh: Mesh):
    """A `TrainState` placed exactly as the production trainer places it
    (`run_state.init_train_state`): C-sharded V/U + ci_fn, replicated sources, over the
    `dp` mesh. Built directly from the `*_sharded` init fns so the saved/restored
    leaves carry real `NamedSharding`s — the production checkpoint path, not `mesh=None`."""
    cfg = _tiny_cfg()
    n = mesh.devices.size
    C, seq = 8 * n, 16
    sites = llama_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu_placed(sites, jax.random.PRNGKey(seed), mesh)
    ci_fn = init_ci_fn_placed(
        _chunkwise_arch(lm, cfg), lm.sites, jax.random.PRNGKey(seed + 1), mesh
    )
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    src = init_sources_sharded(
        lm.site_names,
        tuple(s.C for s in lm.sites),
        seq,
        SCScope(),
        n,
        jax.random.PRNGKey(seed + 2),
        mesh,
    )
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        sources={"PersistentPGDReconLoss": src},
        sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(src)},
        step=jnp.asarray(7, jnp.int32),
    )  # fmt: skip
    return state


def test_sharded_roundtrip_bit_equal(tmp_path: Path):
    """S22 at the PRODUCTION per-rank shape: a sharded `TrainState` (the failure-prone
    path that bit the torch job-34446 save and the jsp SIGTERM saves) must round-trip
    through orbax onto a sharded reference bit-equal, leaf shardings preserved.

    Run this at `XLA_FLAGS=--xla_force_host_platform_device_count=4` to exercise the
    real multi-shard write/read; at the default 1 device it degrades to the replicated
    case (still a real save->restore, just one shard)."""
    from jax.sharding import NamedSharding

    mesh = dp_mesh()
    state = _build_sharded(seed=1, mesh=mesh)

    # The big V/U + ci_fn + sources leaves must be genuinely C-sharded over the mesh
    # (the multi-shard write path); only the small scalars (step) stay single-device.
    n_named = sum(isinstance(x.sharding, NamedSharding) for x in jax.tree.leaves(state))
    assert n_named >= len(jax.tree.leaves(state.components)), n_named

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 3, state)

    # Restore onto a DIFFERENTLY-seeded sharded reference: every leaf comes from disk,
    # but its placement comes from the (correctly-placed) reference.
    reference = _build_sharded(seed=7, mesh=mesh)
    loaded = restore_step(mgr, reference, 3)

    state_leaves = jax.tree.leaves(state)
    loaded_leaves = jax.tree.leaves(loaded)
    ref_leaves = jax.tree.leaves(reference)
    for saved, got, ref in zip(state_leaves, loaded_leaves, ref_leaves, strict=True):
        assert jnp.array_equal(jnp.asarray(saved), jnp.asarray(got))
        assert got.sharding == ref.sharding
