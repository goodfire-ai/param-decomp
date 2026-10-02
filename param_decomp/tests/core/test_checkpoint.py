"""Round-trip + resume-continuation tests for `checkpoint.py` (orbax) on the generic
trainer state: a restored `PDState` must continue the EXACT trajectory —
including the persistent adversary's sources and Adam moments."""

from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from jax.sharding import AxisType, Mesh

from param_decomp.core.adversary import (
    PersistentAdversary,
    SourcesAdamState,
    SourceStacks,
    init_persistent_sources,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp.core.checkpoint import (
    make_checkpoint_manager,
    make_read_only_checkpoint_manager,
    restore_decomposition,
    restore_destination,
    restore_latest,
    restore_step,
    save_state,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.optimizer import ci_fn_muon_dimension_numbers
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    group_factorizations,
    init_component_stacks,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    KeepAllCheckpoints,
    KeepLastNCheckpoints,
    MuonOptimizerConfig,
    PersistentPGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import (
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.losses import BatchFrequency, init_frequency_estimator
from param_decomp.core.model import DecomposedModel, PlacedModel, Positioned
from param_decomp.core.objective import build_objective
from param_decomp.core.optimizer import ScheduledOptimizer, ScheduledOptimizerState
from param_decomp.core.placement import from_config
from param_decomp.core.run_state import (
    _adamw_optimizer,
    _muon_optimizer,
    component_muon_dimension_numbers,
)
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    PDState,
    PDTrainingState,
    TrainState,
    make_train_step,
)
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.target_ports.llama import LlamaConfig
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    TransformerPreparedMasking,
    TransformerPreparedWeights,
    glu_site_specs,
    mlp_family_site_cs,
)
from param_decomp.tests.placed_ci_fn import placed_ci_fn


def _ppgd_cfg(n_warmup: int) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=0.5,
        source_shape="bsc",
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
            ),
        ),
        n_warmup_steps=n_warmup,
    )


def _adversary(src: SourceStacks, cfg: PersistentPGDReconLossConfig) -> PersistentAdversary:
    return PersistentAdversary(
        sources=src,
        opt_state=init_sources_adam_state(src),
        state_key=cfg.type,
        optimizer=cfg.optimizer,
        n_warmup=cfg.n_warmup_steps,
    )


def _chunkwise_arch(
    model: DecomposedModel[
        LMBatchWithDocuments,
        LMOutput,
        TransformerPreparedWeights,
        LMBatchWithDocuments,
        TransformerPreparedMasking,
    ],
    cfg: LlamaConfig,
) -> ChunkwiseTransformerCIFnArch:
    """The old `CIArch(16, 2, 2, 32)` → one chunk reading the residual entering the first
    decomposed block and emitting CI for every site; `input_dim` is the residual width."""
    site_names = model.site_names
    first_block = min(int(n.split(".")[1]) for n in site_names)
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


_C, _SEQ = 8, 16


def _objective(
    sites: tuple[SiteSpec, ...],
    ppgd_cfg: PersistentPGDReconLossConfig,
    frequency: FrequencyMinimalityConfig | None,
):
    return build_objective(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                gamma=ScheduleConfig(
                    max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.01))
                ),
                frequency=frequency,
            ),
            StochasticReconSubsetLossConfig(routing=UniformKSubsetRoutingConfig(), coeff=0.5),
            ppgd_cfg,
        ),
        sites,
    )


def _optimizer(
    muon: bool,
    grad_clip_norm: float | None,
    dimension_numbers: Callable[[optax.Params], optax.Params] | None,
) -> ScheduledOptimizer:
    schedule = ScheduleConfig(
        max_val=1e-3,
        points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1, interp="cosine")),
    )
    match muon:
        case True:
            assert dimension_numbers is not None
            config = MuonOptimizerConfig(
                type="muon", lr_schedule=schedule, grad_clip_norm=grad_clip_norm
            )
            return _muon_optimizer(config, 100, dimension_numbers, None)
        case False:
            config = AdamWOptimizerConfig(lr_schedule=schedule, grad_clip_norm=grad_clip_norm)
            return _adamw_optimizer(config, 100)


def _different_scalar_settings(
    state: PDState[LMBatchWithDocuments],
) -> PDState[LMBatchWithDocuments]:
    def change_lr(optimizer_state: ScheduledOptimizerState) -> ScheduledOptimizerState:
        return eqx.tree_at(
            lambda value: value.schedule.magnitude,
            optimizer_state,
            optimizer_state.schedule.magnitude * 3,
        )

    objective = state.training.objective
    return replace(
        state,
        training=replace(
            state.training,
            objective=eqx.tree_at(
                lambda value: value.minimality.activity_coeff.magnitude,
                objective,
                objective.minimality.activity_coeff.magnitude * 2,
            ),
            components_opt_state=change_lr(state.training.components_opt_state),
            ci_fn_opt_state=change_lr(state.training.ci_fn_opt_state),
        ),
    )


@cache
def _optimizers_and_step(muon_components: bool, muon_ci_fn: bool):
    """Reuse the trace across independently seeded checkpoint fixtures."""
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 4, _C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

    opt_vu = _optimizer(
        muon_components, 0.01, component_muon_dimension_numbers(group_factorizations(sites))
    )
    opt_ci = _optimizer(muon_ci_fn, None, ci_fn_muon_dimension_numbers)
    ppgd_cfg = _ppgd_cfg(n_warmup=1)
    placed = PlacedModel(model=model, placement=None)
    step = jax.jit(
        make_train_step(
            model_static=placed,
            substrate=ForwardSubstrate.of(
                placed,
                remat_recon_forwards=True,
                remat_ci_fn=False,
                ci_capture_keys=_chunkwise_arch(model, cfg).capture_keys,
            ),
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=100,
            faithfulness=faithfulness_loss_for(placed),
        )
    )
    resid = jax.random.randint(jax.random.PRNGKey(9), (2, _SEQ), 0, cfg.vocab_size)
    return (
        cfg,
        sites,
        placed,
        opt_vu,
        opt_ci,
        ppgd_cfg,
        step,
        LMBatchWithDocuments.from_unsegmented_sequences(resid),
    )


def _build(
    seed: int,
    muon_components: bool = False,
    muon_ci_fn: bool = False,
    freq_ema: bool = False,
):
    cfg, sites, model, opt_vu, opt_ci, ppgd_cfg, step, resid = _optimizers_and_step(
        muon_components, muon_ci_fn
    )
    vu = init_component_stacks(sites, jax.random.PRNGKey(seed))
    ci_fn = _chunkwise_arch(model.model, cfg).initialize(
        model.model.sites, None, jax.random.PRNGKey(seed + 1)
    )
    src = init_persistent_sources(
        model.model.sites,
        (2, _SEQ),
        jnp.float32,
        jax.random.PRNGKey(seed + 2),
    )
    frequency = (
        FrequencyMinimalityConfig(coeff=1e-6, reference_datapoint_count=128, ema_halflife_steps=8.0)
        if freq_ema
        else None
    )
    state = TrainState(
        decomposition=Decomposition[LMBatchWithDocuments](components=vu, ci_fn=ci_fn),
        training=PDTrainingState(
            frequency=init_frequency_estimator(frequency, model.model.sites),
            objective=_objective(model.model.sites, ppgd_cfg, frequency),
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={ppgd_cfg.type: _adversary(src, ppgd_cfg)},
            step=jnp.zeros((), jnp.int32),
        ),
    )
    return model, state, step, resid


def _roundtrip_and_exact_resume(
    tmp_path: Path,
    muon_components: bool,
    muon_ci_fn: bool = False,
    freq_ema: bool = False,
) -> None:
    model, state, step, resid = _build(
        seed=1,
        muon_components=muon_components,
        muon_ci_fn=muon_ci_fn,
        freq_ema=freq_ema,
    )
    for i in range(2):
        state, _ = step(model, state, resid, jax.random.PRNGKey(i))

    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 2, state)

    abstract_inputs = jax.tree.map(
        ocp.utils.to_shape_dtype_struct, (model, resid, jax.random.PRNGKey(100))
    )
    abstract_model, abstract_batch, abstract_key = abstract_inputs
    with jax.transfer_guard("disallow"):
        abstract = jax.eval_shape(
            lambda: _different_scalar_settings(
                _build(
                    seed=7,
                    muon_components=muon_components,
                    muon_ci_fn=muon_ci_fn,
                    freq_ema=freq_ema,
                )[1]
            )
        )
        compiled = (
            jax.jit(step).lower(abstract_model, abstract, abstract_batch, abstract_key).compile()
        )
        destination = restore_destination(
            abstract, compiled.input_formats[0][1], hsdp_mesh(1, 1, 1)
        )
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(destination))
    restored = restore_latest(mgr, destination)
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

    for leaf, fmt in zip(
        jax.tree.leaves(loaded), jax.tree.leaves(compiled.input_formats[0][1]), strict=True
    ):
        if fmt.sharding is not None:
            assert leaf.format == fmt

    state_cont, m_cont = compiled(model, state, resid, jax.random.PRNGKey(100))
    loaded_cont, m_load = compiled(model, loaded, resid, jax.random.PRNGKey(100))
    for k in m_cont:
        assert float(m_cont[k]) == float(m_load[k]), k
    for a, b in zip(jax.tree.leaves(state_cont), jax.tree.leaves(loaded_cont), strict=True):
        assert jnp.array_equal(jnp.asarray(a), jnp.asarray(b))


@pytest.mark.slow
def test_roundtrip_and_exact_resume(tmp_path: Path):
    _roundtrip_and_exact_resume(tmp_path, muon_components=False)


@pytest.mark.slow
def test_freq_ema_roundtrip_and_exact_resume(tmp_path: Path):
    """EMA histories, counts, and adversary state restore exactly without a live reference."""
    _roundtrip_and_exact_resume(tmp_path, muon_components=False, freq_ema=True)


def test_muon_roundtrip_and_exact_resume(tmp_path: Path):
    """The muon components opt state (optax's `MuonState` pytree
    verbatim, partitioned into muon/adam masked trees) must restore from abstract
    initialization and continue exactly — this is what a scavenge preemption + requeue
    exercises."""
    _roundtrip_and_exact_resume(tmp_path, muon_components=True)


@pytest.mark.slow
def test_muon_ci_fn_roundtrip_and_exact_resume(tmp_path: Path):
    """Restore Muon state for both components and CI, then continue the exact trajectory.

    The CI tree mixes 3D matrix stacks optimized by Muon and 2D bias stacks using
    Adam, so both optimizer partitions must survive the checkpoint."""
    _roundtrip_and_exact_resume(tmp_path, muon_components=True, muon_ci_fn=True)


def test_persistent_adam_step_count_roundtrip_and_post_resume_bias_correction(tmp_path: Path):
    """After N persistent ascents, the Orbax
    checkpoint must carry the adversary's `step_count` leaf (present, fp32, == N) and
    bit-equal Adam moments; the FIRST post-resume ascent must apply bias-correction for
    count N+1 (not N, not 1)."""
    state_key = "PersistentPGDReconLoss"
    beta1, beta2 = 0.5, 0.99

    model, state, step, resid = _build(seed=1)
    for i in range(3):
        state, _ = step(model, state, resid, jax.random.PRNGKey(i))

    pre_save = state.training.adversaries[state_key].opt_state
    assert isinstance(pre_save, SourcesAdamState)
    n_ascents = int(pre_save.step_count)
    # Each train step runs n_warmup_steps (1) supplemental ascents + 1 final ascent.
    assert n_ascents == 3 * (1 + 1)

    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 3, state)

    _, fresh, _, _ = _build(seed=7)
    restored = restore_latest(mgr, jax.tree.map(ocp.utils.to_shape_dtype_struct, fresh))
    assert restored is not None
    loaded, _ = restored
    loaded_adam = loaded.training.adversaries[state_key].opt_state
    assert isinstance(loaded_adam, SourcesAdamState)

    # (a) the step_count leaf survived the round-trip: present, fp32 scalar, value N.
    assert state_key in loaded.training.adversaries
    assert loaded_adam.step_count.dtype == jnp.float32
    assert loaded_adam.step_count.shape == ()
    assert float(loaded_adam.step_count) == float(n_ascents)

    # (c) the restored Adam moments are bit-equal to pre-save (every leaf of m and v).
    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(loaded_adam.m), jax.tree.leaves(pre_save.m), strict=True)
    )
    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(loaded_adam.v), jax.tree.leaves(pre_save.v), strict=True)
    )

    # (b) the first post-resume ascent applies bias-correction for count N+1.
    adam_cfg = AdamPGDConfig(
        beta1=beta1,
        beta2=beta2,
        lr_schedule=ScheduleConfig(
            max_val=0.01,
            points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
        ),
    )
    loaded_sources = loaded.training.adversaries[state_key].sources
    grads = jax.tree.map(jnp.ones_like, loaded_sources)
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
    mgr = make_checkpoint_manager(tmp_path / "empty", KeepLastNCheckpoints(n=2))
    assert restore_latest(mgr, jax.tree.map(ocp.utils.to_shape_dtype_struct, fresh)) is None


def test_saved_layout_is_two_items(tmp_path: Path):
    """The on-disk item names are the cross-version contract every consumer keys on —
    pin them."""
    _, state, _, _ = _build(seed=1)
    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 0, state)
    step_dir = tmp_path / "ckpts" / "0"
    assert (step_dir / "decomposition").is_dir()
    assert (step_dir / "training").is_dir()
    assert not (step_dir / "default").exists()


@pytest.mark.parametrize(
    ("retention", "surviving_steps"),
    [(KeepLastNCheckpoints(n=2), [3, 4]), (KeepAllCheckpoints(), [0, 1, 2, 3, 4])],
)
def test_retention_decides_what_survives_on_disk(
    tmp_path: Path, retention: KeepLastNCheckpoints | KeepAllCheckpoints, surviving_steps: list[int]
):
    """`PeriodicCheckpointing.retention` is a claim about the FILESYSTEM, so assert the
    filesystem: after five saves, exactly these `ckpts/<step>/` directories remain (and
    orbax agrees via `all_steps`). Deletion is synchronous — no post-save settling."""
    _, state, _, _ = _build(seed=1)
    ckpt_dir = tmp_path / "ckpts"
    mgr = make_checkpoint_manager(ckpt_dir, retention)
    for step in range(5):
        save_state(mgr, step, state)

    on_disk = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir())
    assert on_disk == surviving_steps
    assert sorted(mgr.all_steps()) == surviving_steps


def test_read_only_manager_never_prunes(tmp_path: Path):
    """A consumer opening someone else's run must not garbage-collect it: the read-only
    manager carries no retention policy at all, so every step survives being read."""
    _, state, _, _ = _build(seed=1)
    ckpt_dir = tmp_path / "ckpts"
    writer = make_checkpoint_manager(ckpt_dir, KeepAllCheckpoints())
    for step in range(3):
        save_state(writer, step, state)

    reader = make_read_only_checkpoint_manager(ckpt_dir)
    abstract = jax.eval_shape(lambda: state.decomposition)
    restore_decomposition(reader, 0, abstract)
    assert sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir()) == [0, 1, 2]


def _build_sharded(
    seed: int,
    mesh: Mesh,
    place_vu: "Callable[[tuple[SiteSpec, ...], jax.Array], ComponentStacks]",
    C: int,
    global_batch: int,
):
    """A `PDState` placed exactly as the production trainer places it
    (`run_state.init_pd_state`): C-sharded V/U + ci_fn, batch-sharded sources, over the
    `dp` mesh. Built directly from the `*_sharded` init fns so the saved/restored
    leaves carry real `NamedSharding`s — the production checkpoint path, not `mesh=None`.
    `place_vu` seeds AND places the V/U masters (their moments inherit the placement via
    `opt.init`), so a test can pin any master layout, current or historical. `C` and
    `global_batch` are explicit: changing placement preserves every logical shape."""
    cfg = tiny_glu_cfg()
    seq = 16
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = place_vu(sites, jax.random.PRNGKey(seed))
    ci_fn = placed_ci_fn(
        _chunkwise_arch(model, cfg),
        model.sites,
        jax.random.PRNGKey(seed + 1),
        mesh,
        from_config("zero1", mesh, model.sites),
    )
    opt_vu = _optimizer(False, 0.01, None)
    opt_ci = _optimizer(False, None, None)
    src = init_sources_sharded(
        model.sites,
        Positioned(seq),
        "bsc",
        global_batch,
        jnp.float32,
        jax.random.PRNGKey(seed + 2),
        mesh,
    )
    ppgd_cfg = _ppgd_cfg(n_warmup=1)
    state = TrainState(
        decomposition=Decomposition[LMBatchWithDocuments](components=vu, ci_fn=ci_fn),
        training=PDTrainingState(
            frequency=BatchFrequency(),
            objective=_objective(model.sites, ppgd_cfg, None),
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={ppgd_cfg.type: _adversary(src, ppgd_cfg)},
            step=jnp.asarray(7, jnp.int32),
        ),
    )
    return state


def test_sharded_roundtrip_bit_equal(tmp_path: Path):
    """At the PRODUCTION per-rank shape: a sharded `PDState` (the failure-prone
    path that bit legacy Torch and JAX interrupted saves) must round-trip
    through orbax onto a sharded reference bit-equal, leaf shardings preserved.

    Run this at `XLA_FLAGS=--xla_force_host_platform_device_count=4` to exercise the
    real multi-shard write/read; at the default 1 device it degrades to the replicated
    case (still a real save->restore, just one shard)."""
    from jax.sharding import NamedSharding

    mesh = hsdp_mesh(1, jax.device_count(), 1)

    def place_vu(sites: tuple[SiteSpec, ...], key: jax.Array) -> ComponentStacks:
        return init_component_stacks_placed(sites, key, from_config("owner", mesh, sites))

    state = _build_sharded(
        seed=1,
        mesh=mesh,
        place_vu=place_vu,
        C=8 * mesh.devices.size,
        global_batch=mesh.devices.size,
    )

    # The big V/U + ci_fn + sources leaves must be genuinely C-sharded over the mesh
    # (the multi-shard write path); only the small scalars (step) stay single-device.
    n_named = sum(isinstance(x.sharding, NamedSharding) for x in jax.tree.leaves(state))
    assert n_named >= len(jax.tree.leaves(state.decomposition.components)), n_named

    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 3, state)

    # Restore onto a DIFFERENTLY-seeded sharded reference: every leaf comes from disk,
    # but its placement comes from the (correctly-placed) reference.
    reference = _build_sharded(
        seed=7,
        mesh=mesh,
        place_vu=place_vu,
        C=8 * mesh.devices.size,
        global_batch=mesh.devices.size,
    )
    loaded = restore_step(mgr, jax.tree.map(ocp.utils.to_shape_dtype_struct, reference), 3)

    state_leaves = jax.tree.leaves(state)
    loaded_leaves = jax.tree.leaves(loaded)
    ref_leaves = jax.tree.leaves(reference)
    for saved, got, ref in zip(state_leaves, loaded_leaves, ref_leaves, strict=True):
        assert jnp.array_equal(jnp.asarray(saved), jnp.asarray(got))
        assert got.sharding == ref.sharding


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_restore_reshards_a_checkpoint_saved_in_another_master_layout(tmp_path: Path):
    """Layout-migration compat: orbax saves the LOGICAL array and `restore_step` places
    it by the reference's shardings, so a checkpoint whose V/U masters (and their Adam
    moments) persist d_in-major — the pre-C-minor matrix layout — restores onto today's
    zero1 reference with no migration pass. Values bit-equal, placement the reference's."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    old_v = NamedSharding(mesh, P(None, ("fsdp", "replicate"), "tp"))
    old_u = NamedSharding(mesh, P(None, "tp", ("fsdp", "replicate")))

    def place_vu_old(sites: tuple[SiteSpec, ...], key: jax.Array) -> ComponentStacks:
        vu = init_component_stacks(sites, key)
        return jax.device_put(
            vu,
            ComponentStacks(
                stacks={group: (old_v, old_u) for group in vu.stacks},
                site_stack_indices=vu.site_stack_indices,
            ),
        )

    def place_vu_new(sites: tuple[SiteSpec, ...], key: jax.Array) -> ComponentStacks:
        return init_component_stacks_placed(sites, key, from_config("zero1", mesh, sites))

    saved = _build_sharded(
        seed=1,
        mesh=mesh,
        place_vu=place_vu_old,
        C=8 * mesh.devices.size,
        global_batch=mesh.devices.size,
    )
    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 3, saved)

    reference = _build_sharded(
        seed=7,
        mesh=mesh,
        place_vu=place_vu_new,
        C=8 * mesh.devices.size,
        global_batch=mesh.devices.size,
    )
    loaded = restore_step(mgr, jax.tree.map(ocp.utils.to_shape_dtype_struct, reference), 3)

    for was, got, ref in zip(
        jax.tree.leaves(saved), jax.tree.leaves(loaded), jax.tree.leaves(reference), strict=True
    ):
        # Host-side value compare: the two layouts commit to different specs, and
        # explicit-mode ops refuse mixed-sharding operands.
        assert np.array_equal(np.asarray(was), np.asarray(got))
        assert got.sharding == ref.sharding


def _filled_with_asymmetric_content(
    state: PDState[LMBatchWithDocuments], seed: int
) -> PDState[LMBatchWithDocuments]:
    """Every array leaf refilled with distinct nonzero position-asymmetric values
    (a per-leaf-offset ramp plus noise), placement preserved. Seeded inits leave CI
    biases (and other leaves) at zero — symmetric content a value-dropping or
    value-permuting reshard bug survives — so parity tests compare against this."""
    leaves, treedef = jax.tree.flatten(state)
    rng = np.random.default_rng(seed)
    filled = []
    for index, leaf in enumerate(leaves):
        ramp = np.arange(1, leaf.size + 1).reshape(leaf.shape)
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            values = (ramp / leaf.size + rng.standard_normal(leaf.shape) + index).astype(leaf.dtype)
        else:
            values = (ramp + index).astype(leaf.dtype)
        filled.append(jax.device_put(values, leaf.sharding))
    return jax.tree.unflatten(treedef, filled)


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 8, reason="requires eight local devices")
@pytest.mark.parametrize(("replicate", "fsdp", "tp"), [(1, 2, 2), (2, 2, 1), (2, 2, 2)])
def test_restore_reshards_a_train_topology_checkpoint_onto_a_single_device(
    tmp_path: Path, replicate: int, fsdp: int, tp: int
):
    """Cross-topology restore parity: a checkpoint saved at a placed train mesh restores
    onto the (1, 1, 1) single-device reference — the consumer/debug topology — with
    every leaf value-equal to the semantic original. The state carries randomized
    asymmetric content (`_filled_with_asymmetric_content`), so a reshard that drops,
    permutes, or zero-fills values fails even on the leaves seeded init leaves at zero."""
    n = replicate * fsdp * tp
    C = 8 * n
    train_mesh = Mesh(
        np.asarray(jax.devices()[:n]).reshape(replicate, fsdp, tp),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    single_mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1),
        ("replicate", "fsdp", "tp"),
        axis_types=(AxisType.Explicit,) * 3,
    )

    def place_vu(mesh: Mesh) -> "Callable[[tuple[SiteSpec, ...], jax.Array], ComponentStacks]":
        def place(sites: tuple[SiteSpec, ...], key: jax.Array) -> ComponentStacks:
            return init_component_stacks_placed(sites, key, from_config("owner", mesh, sites))

        return place

    saved = _filled_with_asymmetric_content(
        _build_sharded(seed=1, mesh=train_mesh, place_vu=place_vu(train_mesh), C=C, global_batch=n),
        seed=11,
    )
    assert any(len(leaf.sharding.device_set) > 1 for leaf in jax.tree.leaves(saved))
    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 3, saved)

    reference = _build_sharded(
        seed=7, mesh=single_mesh, place_vu=place_vu(single_mesh), C=C, global_batch=n
    )
    loaded = restore_step(mgr, jax.tree.map(ocp.utils.to_shape_dtype_struct, reference), 3)

    for was, got, ref in zip(
        jax.tree.leaves(saved), jax.tree.leaves(loaded), jax.tree.leaves(reference), strict=True
    ):
        # Host-side comparison: the two sides are committed to different device sets, so
        # a device-side equal would refuse the mixed placement.
        assert np.array_equal(np.asarray(was), np.asarray(got))
        assert got.sharding == ref.sharding
