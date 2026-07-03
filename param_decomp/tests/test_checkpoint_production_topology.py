"""Production-topology orbax round-trip (SPEC S22, issue #617).

The jsp SIGTERM lore (preemptions fall back to periodic checkpoints) makes it
load-bearing that the SHARDED orbax save at the production placement actually persists
the persistent adversary's Adam moments — a missing moment tree would silently reset
Adam after a real preemption. `test_checkpoint.py` covers the `mesh=None` single-term
case with a leaf-equality sweep; this adds the parts that only bite at production
topology:

  * the V/U + Adam states are C-SHARDED and the sources/moments REPLICATED over a
    multi-device `dp` mesh (`llama8b_sharding.py`), exactly as `init_train_state` places
    them — so the test exercises the sharded save/restore path, not the all-on-one path;
  * MULTIPLE persistent terms (SPEC S23: one `adversaries` entry per term), so a
    per-term moment tree that got dropped would surface;
  * an explicit structural assertion that the RESTORED pytree carries `m`, `v`, and a
    non-zero `step_count` for every persistent term and every site — not just that the
    leaves happen to be equal;
  * an assertion that restore reconstructs each leaf onto the REFERENCE sharding.

Run at >1 device to actually shard:
`XLA_FLAGS="--xla_force_host_platform_device_count=4" pytest <this file>`.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import NamedSharding

from param_decomp.adversary import (
    PersistentAdversary,
    SourcesAdamState,
    init_sources_adam_state,
)
from param_decomp.checkpoint import make_checkpoint_manager, restore_latest, save_state
from param_decomp.ci_fn import Chunk, ChunkwiseTransformerCIArch
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.recon import build_loss_terms, persistent_configs
from param_decomp.run import _ensure_global
from param_decomp.schedule import ScheduleConfig
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.targets.llama8b_sharding import (
    hsdp_mesh,
    init_ci_fn_placed,
    init_decomp_vu_placed,
    init_sources_sharded,
)
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.train import TrainState, make_train_step

# Needs >1 jax device (production topology); hangs at the default 1 device, so gated behind
# --runmultidevice. Run via `make test-multidevice` (simulated CPU devices). See conftest.
pytestmark = pytest.mark.multidevice

PERSISTENT_TERMS = ("PersistentPGDReconLoss", "ppgd_second")


def _persistent_cfg(name: str | None) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        name=name,
        coeff=0.5,
        scope=SCScope(),
        optimizer=AdamPGDConfig(
            beta1=0.5, beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
        ),
        n_warmup_steps=1,
    )  # fmt: skip


def _build_sharded(seed: int):
    """A TrainState placed exactly as `init_train_state` places a production run on the 2-D
    `(replicate, fsdp)` HSDP mesh: V/U + their Adam moments sharded ÷N over the FULL mesh
    (V d_in, U d_out; C replicated), the CI fn ÷N over the full mesh (d_model), sources + their
    Adam moments replicated, with TWO persistent terms. On the 4-device sim: `replicate=1,
    fsdp=4` (N=4); `C=8` and the V d_in / U d_out tile N."""
    mesh = hsdp_mesh()
    cfg = _tiny_cfg()
    C, seq = 8, 16
    sites = llama_site_specs(cfg, mlp_family_site_cs(3, 4, C))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

    by_layer: dict[int, list[str]] = {}
    for name in lm.site_names:
        by_layer.setdefault(int(name.split(".")[1]), []).append(name)
    ci_arch = ChunkwiseTransformerCIArch(
        chunks=tuple(
            Chunk(input_taps=(f"resid.{layer}",), output_sites=tuple(names))
            for layer, names in sorted(by_layer.items())
        ),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        mlp_hidden=32,
    )
    vu = init_decomp_vu_placed(lm.sites, jax.random.PRNGKey(seed), mesh)
    ci_fn = init_ci_fn_placed(ci_arch, lm.sites, jax.random.PRNGKey(seed + 1), mesh)
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    site_cs = tuple(s.C for s in lm.sites)
    ppgd_cfgs = (_persistent_cfg(None), _persistent_cfg("ppgd_second"))
    adversaries: dict[str, PersistentAdversary] = {}
    for i, (state_key, ppgd_cfg) in enumerate(zip(PERSISTENT_TERMS, ppgd_cfgs, strict=True)):
        assert ppgd_cfg.coeff is not None
        src = init_sources_sharded(
            lm.site_names,
            site_cs,
            seq,
            SCScope(),
            mesh.devices.size,
            jnp.float32,
            jax.random.fold_in(jax.random.PRNGKey(seed + 2), i),
            mesh,
        )
        adversaries[state_key] = PersistentAdversary(
            sources=src,
            opt_state=init_sources_adam_state(src),
            state_key=state_key,
            coeff=ppgd_cfg.coeff,
            adam=ppgd_cfg.optimizer,
            n_warmup=ppgd_cfg.n_warmup_steps,
        )
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        adversaries=adversaries,
        step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    # run.py:185 — normalize eager scalar stragglers (Adam `count`, `step`) onto the mesh
    # so the whole state is global-array placed, exactly as a production run / restore.
    state = _ensure_global(state, mesh)
    assert isinstance(state, TrainState)

    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6, pnorm=2.0, 
                p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0,
            ),
            ChunkwiseSubsetReconLossConfig(routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=3, n_samples=1),
            *ppgd_cfgs,
        ),
        lm.site_names,
    )  # fmt: skip
    assert tuple(persistent_configs(loss_terms.recon)) == PERSISTENT_TERMS, loss_terms

    step = make_train_step(
        lm=lm,
        losses=loss_terms,
        components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100,
        remat_recon_forwards=True, remat_ci_fn=False, mesh=mesh,
    )  # fmt: skip
    tokens = jax.device_put(
        jax.random.randint(jax.random.PRNGKey(9), (4, seq), 0, cfg.vocab_size),
        NamedSharding(mesh, jax.sharding.PartitionSpec(("replicate", "fsdp"))),
    )
    return lm, state, step, tokens


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
            assert adam.m[site].shape == src.shape, (term, site)
            assert adam.v[site].shape == src.shape, (term, site)
        assert float(adam.step_count) > 0.0, (term, float(adam.step_count))


def test_sharded_roundtrip_persists_source_moments(tmp_path: Path):
    """Device-agnostic like the other invariance tests: at 1 device the round-trip is a
    trivial-mesh structural check; only `XLA_FLAGS=--xla_force_host_platform_device_count=4`
    actually shards C, where the `sharding` equality assertions bite."""
    lm, state, step, resid = _build_sharded(seed=1)
    for i in range(2):
        state, _ = step(lm, state, resid, jax.random.PRNGKey(i))
    # The ascents must have advanced each term's Adam counter before we save.
    _assert_moments_present(state.adversaries)

    mgr = make_checkpoint_manager(tmp_path / "ckpts", keep_last=2)
    save_state(mgr, 2, state)

    # Restore onto a DIFFERENTLY-seeded reference at the same (sharded) placement: every
    # leaf, including each term's Adam moments, must come from disk.
    _, fresh, _, _ = _build_sharded(seed=7)
    restored = restore_latest(mgr, fresh)
    assert restored is not None
    loaded, ckpt_step = restored
    assert ckpt_step == 2

    # The persisted pytree itself carries the moments + step_count for every term.
    _assert_moments_present(loaded.adversaries)

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

    # SPEC S22: the restored state continues the trajectory. To fp tolerance, not bit-
    # identically: `state` and `loaded` carry distinct-but-equivalent shardings, so the step
    # jits to distinct executables, and the FSDP V all-gather/reduce is not bit-reproducible
    # — both reassociate the same math (observed rel ~1e-7).
    state_cont, m_cont = step(lm, state, resid, jax.random.PRNGKey(100))
    loaded_cont, m_load = step(lm, loaded, resid, jax.random.PRNGKey(100))
    for k in m_cont:
        assert np.allclose(float(m_cont[k]), float(m_load[k]), rtol=1e-5, atol=1e-6), (
            k,
            float(m_cont[k]),
            float(m_load[k]),
        )
    for a, b in zip(jax.tree.leaves(state_cont), jax.tree.leaves(loaded_cont), strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
