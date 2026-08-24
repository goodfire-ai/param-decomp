"""Production-topology orbax round-trip (SPEC S22, issue #617).

Preemptions can miss the SIGTERM save and fall back to periodic checkpoints, so it is
load-bearing that the SHARDED orbax save at the production placement actually persists
the persistent adversary's Adam moments — a missing moment tree would silently reset
Adam after a real preemption. `test_checkpoint.py` covers the `mesh=None` single-term
case with a leaf-equality sweep; this adds the parts that only bite at production
topology:

  * the V/U + Adam states are sharded along their matrix dimensions and the
    sources/moments are replicated over a multi-device mesh (`init_placed.py`), exactly
    as `init_train_state` places them — so the test exercises the sharded save/restore
    path, not the all-on-one path;
  * MULTIPLE persistent terms (SPEC S23: one `adversaries` entry per term), so a
    per-term moment tree that got dropped would surface;
  * an explicit structural assertion that the RESTORED pytree carries `m`, `v`, and a
    non-zero `step_count` for every persistent term and every site — not just that the
    leaves happen to be equal;
  * an assertion that restore reconstructs each leaf onto the REFERENCE sharding.

Run at >1 device to actually shard. The Makefile simulates four logical CPU devices
because four is the minimum required by the multidevice suite as a whole:
`XLA_FLAGS="--xla_force_host_platform_device_count=4" pytest <this file>`.
"""

from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from param_decomp.core.adversary import (
    PersistentAdversary,
    SourcesAdamState,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp.core.checkpoint import make_checkpoint_manager, restore_latest, save_state
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    KeepLastNCheckpoints,
    PersistentPGDReconLossConfig,
)
from param_decomp.core.init_placed import (
    init_ci_fn_placed,
    init_component_stacks_placed,
    init_sources_sharded,
)
from param_decomp.core.model import Positioned
from param_decomp.core.placement import from_config
from param_decomp.core.run import _ensure_global
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.core.train import Decomposition, TrainingItem, TrainState
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

# Needs >1 jax device (production topology); hangs at the default 1 device, so gated behind
# --runmultidevice. Run via `make test-multidevice` (simulated CPU devices). See conftest.
pytestmark = pytest.mark.multidevice

PERSISTENT_TERMS = ("PersistentPGDReconLoss", "ppgd_second")


def _persistent_cfg(name: str | None) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        name=name,
        coeff=0.5,
        source_shape="sc",
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                max_val=0.01,
                points=(Knot(at=0.0, frac=0.0), Knot(at=0.025, frac=1.0), Knot(at=1.0, frac=1.0)),
            ),
        ),
        n_warmup_steps=1,
    )


def _build_sharded(seed: int):
    """A TrainState placed exactly as `init_train_state` places a production run on the
    `(replicate, fsdp, tp)` HSDP mesh: V/U + their Adam moments sharded ÷N over the data
    axes (V d_in, U d_out; C replicated), the CI fn ÷N over the data axes (d_model), and
    sources + their Adam moments replicated, with TWO persistent terms. On the four-device
    CPU simulation: `replicate=1, fsdp=4, tp=1` (N=4); `C=8` and the V d_in / U d_out
    dimensions tile N."""
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    cfg = tiny_glu_cfg()
    C, seq = 8, 16
    sites = glu_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

    by_layer: dict[int, list[str]] = {}
    for name in model.site_names:
        by_layer.setdefault(int(name.split(".")[1]), []).append(name)
    ci_arch = ChunkwiseTransformerCIArch(
        chunks=tuple(
            Chunk(input_taps=(f"resid.{layer}",), output_sites=tuple(names))
            for layer, names in sorted(by_layer.items())
        ),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    vu = init_component_stacks_placed(
        model.sites, jax.random.PRNGKey(seed), from_config("owner", mesh, model.sites)
    )
    ci_fn = init_ci_fn_placed(
        ci_arch,
        model.sites,
        jax.random.PRNGKey(seed + 1),
        mesh,
        from_config("zero1", mesh, model.sites),
    )
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    site_cs = tuple(s.C for s in model.sites)
    ppgd_cfgs = (_persistent_cfg(None), _persistent_cfg("ppgd_second"))
    adversaries: dict[str, PersistentAdversary] = {}
    for i, (state_key, ppgd_cfg) in enumerate(zip(PERSISTENT_TERMS, ppgd_cfgs, strict=True)):
        assert ppgd_cfg.coeff is not None
        src = init_sources_sharded(
            model.site_names,
            site_cs,
            Positioned(seq),
            "sc",
            mesh.devices.size,
            jnp.float32,
            jax.random.fold_in(jax.random.PRNGKey(seed + 2), i),
            mesh,
        )
        # The checkpoint needs realistic non-zero moments, not a full training step.
        # A direct Adam ascent creates the same state shape with deterministic values that
        # differ between the saved and restore-reference seeds, without any collectives.
        opt_state = init_sources_adam_state(src)
        sources, opt_state = sources_adam_ascend_project(
            src,
            jax.tree.map(partial(jnp.full_like, fill_value=seed + i + 1), src),
            opt_state,
            jnp.asarray(0.01),
            ppgd_cfg.optimizer,
        )
        adversaries[state_key] = PersistentAdversary(
            sources=sources,
            opt_state=opt_state,
            state_key=state_key,
            adam=ppgd_cfg.optimizer,
            n_warmup=ppgd_cfg.n_warmup_steps,
        )
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries=adversaries,
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    # run.py:185 — normalize eager scalar stragglers (Adam `count`, `step`) onto the mesh
    # so the whole state is global-array placed, exactly as a production run / restore.
    state = _ensure_global(state, mesh)
    assert isinstance(state, TrainState)

    return state


def _assert_moments_present(adversaries: dict[str, PersistentAdversary]) -> None:
    """SPEC S22/S23: every persistent term carries m, v (one leaf per source site, same
    shape as the source) and a non-zero step_count."""
    assert tuple(adversaries) == PERSISTENT_TERMS, adversaries.keys()
    for term in PERSISTENT_TERMS:
        adv = adversaries[term]
        adam = adv.opt_state
        assert isinstance(adam, SourcesAdamState)
        assert set(adam.m) == set(adv.sources), (term, adam.m.keys())
        assert set(adam.v) == set(adv.sources), (term, adam.v.keys())
        for site, src in adv.sources.items():
            assert jax.tree.map(lambda x: x.shape, adam.m[site]) == jax.tree.map(
                lambda x: x.shape, src
            ), (term, site)
            assert jax.tree.map(lambda x: x.shape, adam.v[site]) == jax.tree.map(
                lambda x: x.shape, src
            ), (term, site)
        assert float(adam.step_count) > 0.0, (term, float(adam.step_count))


def test_sharded_roundtrip_persists_source_moments(tmp_path: Path):
    """Device-agnostic like the other invariance tests: at one device the round-trip is
    a trivial-mesh structural check; multiple logical devices make the placement assertions
    exercise real shards. The suite uses four because other multidevice tests require it."""
    state = _build_sharded(seed=1)
    _assert_moments_present(state.training.adversaries)

    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=2))
    save_state(mgr, 2, state)

    # Restore onto a DIFFERENTLY-seeded reference at the same (sharded) placement: every
    # leaf, including each term's Adam moments, must come from disk.
    fresh = _build_sharded(seed=7)
    for term in PERSISTENT_TERMS:
        saved_adam = state.training.adversaries[term].opt_state
        reference_adam = fresh.training.adversaries[term].opt_state
        assert any(
            not np.array_equal(np.asarray(saved), np.asarray(reference))
            for saved, reference in zip(
                jax.tree.leaves(saved_adam.m),
                jax.tree.leaves(reference_adam.m),
                strict=True,
            )
        ), f"{term}: saved and reference moments must differ for the restore check to bite"
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, ckpt_step = restored
    assert ckpt_step == 2

    # The persisted pytree itself carries the moments + step_count for every term.
    _assert_moments_present(loaded.training.adversaries)

    # Values come from disk, and each leaf is reconstructed onto the REFERENCE sharding.
    # Pull leaves to host before comparing: a sharded leaf and a single-device leaf can't
    # be compared on-device (jit refuses the mismatched device sets), so equality goes
    # through numpy while sharding equality is read off the live arrays.
    saved_leaves, saved_def = jax.tree.flatten(state)
    loaded_leaves = jax.tree.leaves(loaded)
    ref_leaves = jax.tree.leaves(fresh)
    assert jax.tree.structure(loaded) == saved_def
    for saved, ref, got in zip(saved_leaves, ref_leaves, loaded_leaves, strict=True):
        assert np.array_equal(np.asarray(saved), np.asarray(got))
        assert got.sharding == ref.sharding, (got.sharding, ref.sharding)
