"""Round-trip + resume-continuation tests for `checkpoint.py` (orbax) on the generic
trainer state (SPEC S22): a restored `TrainState` must continue the EXACT trajectory —
including the persistent adversary's sources and Adam moments."""

from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import Mesh

from param_decomp.core.adversary import (
    PersistentAdversary,
    init_persistent_sources,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp.core.checkpoint import (
    init_from_parent,
    make_checkpoint_manager,
    restore_decomposition_to_host,
    restore_latest,
    restore_step,
    save_state,
)
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    build_ci_fn,
)
from param_decomp.core.components import init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.init_placed import (
    init_ci_fn_placed,
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.model import DecomposedModel, Positioned
from param_decomp.core.muon_stacked import stacked_muon
from param_decomp.core.placement import from_config
from param_decomp.core.recon import build_loss_terms
from param_decomp.core.run_state import stacked_muon_dimension_numbers
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.core.train import Decomposition, TrainingItem, TrainState, make_train_step
from param_decomp.targets.glu_transformer import (
    glu_site_specs,
    mlp_family_site_cs,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.vendored_jax.llama import LlamaConfig


def _ppgd_cfg(n_warmup: int) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=0.5,
        source_shape="sc",
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=n_warmup,
    )  # fmt: skip


def _adversary(src: dict[str, jax.Array], cfg: PersistentPGDReconLossConfig) -> PersistentAdversary:
    assert cfg.coeff is not None
    return PersistentAdversary(
        sources=src,
        opt_state=init_sources_adam_state(src),
        state_key=cfg.type,
        coeff=cfg.coeff,
        adam=cfg.optimizer,
        n_warmup=cfg.n_warmup_steps,
    )


def _chunkwise_arch(model: DecomposedModel, cfg: LlamaConfig) -> ChunkwiseTransformerCIArch:
    """The old `CIArch(16, 2, 2, 32)` → one chunk reading the residual entering the first
    decomposed block and emitting CI for every site; `input_dim` is the residual width."""
    site_names = model.site_names
    first_block = min(int(n.split(".")[1]) for n in site_names)
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _build(
    seed: int, muon_components: bool = False, muon_ci_fn: bool = False, stacked_impl: bool = False
):
    cfg = tiny_glu_cfg()
    C, seq = 8, 16
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks(sites, jax.random.PRNGKey(seed))
    ci_fn = build_ci_fn(_chunkwise_arch(model, cfg), model.sites, jax.random.PRNGKey(seed + 1))

    def muon_impl(dim_nums: "Callable[[optax.Params], optax.Params] | None"):
        if stacked_impl:
            return stacked_muon(
                1e-3,
                beta=0.95,
                weight_decay=0.0,
                consistent_rms=0.2,
                muon_weight_dimension_numbers=dim_nums,
                ns_steps=5,
                ns_dtype=jnp.dtype(jnp.float32),
                mesh=None,
            )
        return optax.contrib.muon(1e-3, consistent_rms=0.2, muon_weight_dimension_numbers=dim_nums)

    # Production labeling (`run_state.build_optimizers`): the V/U tree is all-3D
    # `ComponentStacks` stacks, so optax's default 2D rule (dim_nums=None) would label
    # every leaf adam and leave the muon partition under test empty.
    inner_vu = (
        muon_impl(stacked_muon_dimension_numbers)
        if muon_components
        else optax.adamw(1e-3, weight_decay=0.0)
    )
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), inner_vu)
    opt_ci = (
        muon_impl(stacked_muon_dimension_numbers)
        if muon_ci_fn
        else optax.adamw(1e-3, weight_decay=0.0)
    )
    src = init_persistent_sources(
        model.site_names,
        tuple(s.C for s in model.sites),
        (1, seq),
        jnp.float32,
        jax.random.PRNGKey(seed + 2),
    )
    ppgd_cfg = _ppgd_cfg(n_warmup=1)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={ppgd_cfg.type: _adversary(src, ppgd_cfg)},
            step=jnp.zeros((), jnp.int32),
        ),
    )  # fmt: skip
    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6, pnorm=ScheduleConfig(start_val=2.0, fn_type="linear", final_val_frac=0.2),
            ),
            ChunkwiseSubsetReconLossConfig(routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1),
            ppgd_cfg,
        ),
        model.site_names,
    )  # fmt: skip
    step = make_train_step(
        model_static=model,
        losses=loss_terms,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True, remat_ci_fn=False, mesh=None,
    )  # fmt: skip
    resid = jax.random.randint(jax.random.PRNGKey(9), (2, seq), 0, cfg.vocab_size)
    return model, state, step, resid


def _roundtrip_and_exact_resume(
    tmp_path: Path, muon_components: bool, muon_ci_fn: bool = False, stacked_impl: bool = False
) -> None:
    model, state, step, resid = _build(
        seed=1, muon_components=muon_components, muon_ci_fn=muon_ci_fn, stacked_impl=stacked_impl
    )
    for i in range(2):
        state, _ = step(model, state, resid, jax.random.PRNGKey(i))

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, state)

    # Restore onto a DIFFERENTLY-seeded reference: every leaf must come from disk.
    _, fresh, _, _ = _build(
        seed=7, muon_components=muon_components, muon_ci_fn=muon_ci_fn, stacked_impl=stacked_impl
    )
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, ckpt_step = restored
    assert ckpt_step == 2
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(loaded), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))

    if muon_components:
        # The muon partition under test is non-vacuous: the restored MuonState carries
        # real (non-MaskedNode) momentum on the V/U stacks, moved by the two steps.
        [muon_state] = [
            x
            for x in jax.tree.leaves(
                loaded.training.components_opt_state,
                is_leaf=lambda x: isinstance(x, optax.contrib.MuonState),
            )
            if isinstance(x, optax.contrib.MuonState)
        ]
        mu_leaves = jax.tree.leaves(muon_state.mu)
        assert mu_leaves, "no V/U leaf labeled muon: the partition under test is empty"
        assert all(bool(jnp.any(leaf != 0)) for leaf in mu_leaves)

    # SPEC S22: the restored state continues the exact trajectory.
    state_cont, m_cont = step(model, state, resid, jax.random.PRNGKey(100))
    loaded_cont, m_load = step(model, loaded, resid, jax.random.PRNGKey(100))
    for k in m_cont:
        assert float(m_cont[k]) == float(m_load[k]), k
    for a, b in zip(jax.tree.leaves(state_cont), jax.tree.leaves(loaded_cont), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))


def test_roundtrip_and_exact_resume(tmp_path: Path):
    _roundtrip_and_exact_resume(tmp_path, muon_components=False)


def test_muon_roundtrip_and_exact_resume(tmp_path: Path):
    """SPEC S20 amendment: the muon components opt state (optax-partitioned muon/adam
    masked trees) must ALSO restore onto a rebuilt reference and continue exactly —
    this is what a scavenge preemption + requeue exercises."""
    _roundtrip_and_exact_resume(tmp_path, muon_components=True)


def test_muon_ci_fn_roundtrip_and_exact_resume(tmp_path: Path):
    """SPEC S20 amendment (2026-07-11): same guarantee with muon on BOTH groups, the ci-fn
    partitioned by `stacked_muon_dimension_numbers` (3D chunk stacks muon'd, 2D bias
    stacks in the Adam-fallback mask)."""
    _roundtrip_and_exact_resume(tmp_path, muon_components=True, muon_ci_fn=True)


def test_stacked_muon_roundtrip_and_exact_resume(tmp_path: Path):
    """SPEC S20 `impl: stacked`: the stacked-NS muon state is optax's `MuonState` pytree
    verbatim, so the same roundtrip + exact-resume guarantee holds — and a checkpoint
    written under either impl restores under the other (pinned by
    `test_muon_cross_impl_checkpoint_roundtrip`)."""
    _roundtrip_and_exact_resume(tmp_path, muon_components=True, muon_ci_fn=True, stacked_impl=True)


@pytest.mark.parametrize(
    ("save_impl", "restore_impl"), [("stacked", "optax"), ("optax", "stacked")]
)
def test_muon_cross_impl_checkpoint_roundtrip(tmp_path: Path, save_impl: str, restore_impl: str):
    """SPEC S20: `impl: optax` and `impl: stacked` carry the SAME `MuonState` pytree, so a
    checkpoint written under either impl restores bit-exact onto a reference built under
    the other — and the other impl's train step consumes the restored state (continuing it
    exactly as it continues the in-memory state). Muon on BOTH groups so the cross-impl
    claim covers the V/U stacks and the ci-fn partition."""
    model, state, save_step, resid = _build(
        seed=1, muon_components=True, muon_ci_fn=True, stacked_impl=save_impl == "stacked"
    )
    for i in range(2):
        state, _ = save_step(model, state, resid, jax.random.PRNGKey(i))
    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, state)

    # The reference — and the continuation step — are built under the OTHER impl.
    _, fresh, restore_step_fn, _ = _build(
        seed=7, muon_components=True, muon_ci_fn=True, stacked_impl=restore_impl == "stacked"
    )
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, ckpt_step = restored
    assert ckpt_step == 2
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(loaded), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))

    # Non-vacuous: the cross-restored muon momentum is real, moved by the two save-side
    # steps — not an untouched fresh-init tree.
    [muon_state] = [
        x
        for x in jax.tree.leaves(
            loaded.training.components_opt_state,
            is_leaf=lambda x: isinstance(x, optax.contrib.MuonState),
        )
        if isinstance(x, optax.contrib.MuonState)
    ]
    assert all(bool(jnp.any(leaf != 0)) for leaf in jax.tree.leaves(muon_state.mu))

    state_cont, m_cont = restore_step_fn(model, state, resid, jax.random.PRNGKey(100))
    loaded_cont, m_load = restore_step_fn(model, loaded, resid, jax.random.PRNGKey(100))
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

    model, state, step, resid = _build(seed=1)
    for i in range(3):
        state, _ = step(model, state, resid, jax.random.PRNGKey(i))

    pre_save = state.training.adversaries[state_key].opt_state
    n_ascents = int(pre_save.step_count)
    # Each train step runs n_warmup_steps (1) supplemental ascents + 1 final ascent.
    assert n_ascents == 3 * (1 + 1)

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 3, state)

    _, fresh, _, _ = _build(seed=7)
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, _ = restored
    loaded_adam = loaded.training.adversaries[state_key].opt_state

    # (a) the step_count leaf survived the round-trip: present, fp32 scalar, value N.
    assert state_key in loaded.training.adversaries
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
    loaded_sources = loaded.training.adversaries[state_key].sources
    grads = {site: jnp.ones_like(v) for site, v in loaded_sources.items()}
    _, post_resume = sources_adam_ascend_project(
        loaded_sources, grads, loaded_adam, jnp.asarray(0.01), adam_cfg
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


def test_saved_layout_is_two_items(tmp_path: Path):
    """The on-disk item names are the cross-version contract every consumer keys on —
    pin them."""
    _, state, _, _ = _build(seed=1)
    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 0, state)
    step_dir = tmp_path / "ckpts" / "0"
    assert (step_dir / "decomposition").is_dir()
    assert (step_dir / "training").is_dir()
    assert not (step_dir / "default").exists()


def test_consumer_restores_decomposition_to_host(tmp_path: Path):
    """The consumer path (`open_jax_run`): an eval_shape'd `Decomposition` abstract —
    no optimizer/adversary knowledge — restores host-side, bit-equal to the saved
    components + ci_fn."""
    model, state, step, resid = _build(seed=1)
    for i in range(2):
        state, _ = step(model, state, resid, jax.random.PRNGKey(i))
    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, state)

    # A FRESH manager, as every real consumer opens: orbax pins an item's handler per
    # manager instance, so the saving manager can't PyTreeRestore what it StandardSave'd.
    consumer_mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    abstract = jax.eval_shape(lambda: state.decomposition)
    restored = restore_decomposition_to_host(consumer_mgr, 2, abstract)
    for a, b in zip(
        jax.tree.leaves(state.decomposition),
        jax.tree.leaves(restored),
        strict=True,
    ):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))


def test_init_from_parent_restores_decomposition_only(tmp_path: Path):
    """Fine-tune init (S33): `init_from_parent` restores ONLY the parent's
    `decomposition` item (components + ci_fn) onto a fresh reference, keeping the fresh
    reference's optimizer states / adversaries / `step=0` — never reading the parent's
    `training` item, which may differ freely."""
    model, parent, step, resid = _build(seed=1)
    for i in range(2):
        parent, _ = step(model, parent, resid, jax.random.PRNGKey(i))
    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, parent)

    # A fresh reference from a DIFFERENT seed: its components/ci_fn, optimizer state and
    # adversaries all differ from the parent's, so the carry-over vs keep-fresh split is
    # observable on every leaf.
    _, fresh, _, _ = _build(seed=7)
    finetuned = init_from_parent(tmp_path / "ckpts", parent_step=2, reference=fresh)

    # decomposition carries over from the parent...
    for a, b in zip(
        jax.tree.leaves(finetuned.decomposition),
        jax.tree.leaves(parent.decomposition),
        strict=True,
    ):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))
    # ...while optimizer state, adversaries and step stay the fresh reference's.
    for a, b in zip(
        jax.tree.leaves(
            (finetuned.training.components_opt_state, finetuned.training.ci_fn_opt_state)
        ),
        jax.tree.leaves((fresh.training.components_opt_state, fresh.training.ci_fn_opt_state)),
        strict=True,
    ):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))
    assert int(finetuned.training.step) == 0


def _build_sharded(seed: int, mesh: Mesh):
    """A `TrainState` placed exactly as the production trainer places it
    (`run_state.init_train_state`): C-sharded V/U + ci_fn, replicated sources, over the
    `dp` mesh. Built directly from the `*_sharded` init fns so the saved/restored
    leaves carry real `NamedSharding`s — the production checkpoint path, not `mesh=None`."""
    cfg = tiny_glu_cfg()
    n = mesh.devices.size
    C, seq = 8 * n, 16
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_component_stacks_placed(
        sites, jax.random.PRNGKey(seed), from_config("owner", mesh, sites)
    )
    ci_fn = init_ci_fn_placed(
        _chunkwise_arch(model, cfg), model.sites, jax.random.PRNGKey(seed + 1), mesh
    )
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    src = init_sources_sharded(
        model.site_names,
        tuple(s.C for s in model.sites),
        Positioned(seq),
        "sc",
        n,
        jnp.float32,
        jax.random.PRNGKey(seed + 2),
        mesh,
    )
    ppgd_cfg = _ppgd_cfg(n_warmup=1)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={ppgd_cfg.type: _adversary(src, ppgd_cfg)},
            step=jnp.asarray(7, jnp.int32),
        ),
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

    mesh = hsdp_mesh()
    state = _build_sharded(seed=1, mesh=mesh)

    # The big V/U + ci_fn + sources leaves must be genuinely C-sharded over the mesh
    # (the multi-shard write path); only the small scalars (step) stay single-device.
    n_named = sum(isinstance(x.sharding, NamedSharding) for x in jax.tree.leaves(state))
    assert n_named >= len(jax.tree.leaves(state.decomposition.components)), n_named

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
