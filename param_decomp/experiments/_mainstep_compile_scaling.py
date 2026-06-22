"""Main train-step COMPILE time vs (site count, sites_per_chunk) on CPU sim devices.

The chunkwise recon runs a full frozen-suffix forward per chunk; the suspicion is the
main step graph (and thus compile) blows up with site count and/or chunk count. CPU sim
captures graph-size scaling (backend-agnostic), no GPU/autotune."""

import os

# CPU: set MAINSTEP_SIM_DEVICES=N to simulate N CPU devices. GPU: leave it unset and let
# XLA_FLAGS (e.g. autotune level) come from the environment unchanged.
if os.environ.get("MAINSTEP_SIM_DEVICES"):
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={os.environ['MAINSTEP_SIM_DEVICES']}"
    )

import sys
import time

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as P

from param_decomp.ci_fn import CIArch
from param_decomp.configs import (
    AdamPGDConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.experiments.mem_probe import (
    _abstract_replicated,
    _place_state,
    _random_target,
    _typed_state_struct,
)
from param_decomp.llama8b import (
    KIND_ORDER,
    SiteC,
    canonical_site_cs,
    llama31_8b_config,
    llama_decomposed_lm,
    llama_site_specs,
    site_name,
)
from param_decomp.recon import build_recon_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.sharding import dp_mesh
from param_decomp.train import make_train_step

PER_MATRIX_C = {"q": 2048, "k": 2048, "v": 4096, "o": 4096, "gate": 8192, "up": 8192, "down": 10240}
SEQ = 512


def run(n_layers: int, sites_per_chunk: int, mesh, cfg, opt_vu, opt_ci, ci_arch, key) -> None:
    site_cs = tuple(
        SiteC(site_name(layer, kind), PER_MATRIX_C[kind])
        for layer in range(n_layers)
        for kind in KIND_ORDER
    )
    sites = llama_site_specs(cfg, canonical_site_cs(site_cs))
    lm = llama_decomposed_lm(cfg, sites)
    n_chunks = len(sites) // sites_per_chunk
    loss_metrics = (
        FaithfulnessLossConfig(coeff=1e6),
        ImportanceMinimalityLossConfig(
            coeff=5e-6, pnorm=2.0, beta=0.2,
            p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0,
        ),
        ChunkwiseSubsetReconLossConfig(
            routing=UniformKSubsetRoutingConfig(), coeff=2.0,
            sites_per_chunk=sites_per_chunk, n_samples=1,
        ),
        PersistentPGDReconLossConfig(
            coeff=0.5, scope=SCScope(),
            optimizer=AdamPGDConfig(
                beta1=0.01, beta2=0.99,
                lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025),
            ),
            n_warmup_steps=2,
        ),
    )  # fmt: skip
    loss_spec = build_recon_terms(loss_metrics, lm.site_names, n_mask_samples=1, sampling="continuous")
    typed = jax.eval_shape(lambda k: _typed_state_struct(lm, opt_vu, opt_ci, ci_arch, (1, SEQ), k), key)
    state_in = _place_state(typed, mesh)
    tgt_shapes = jax.jit(_random_target, static_argnums=(0, 1)).lower(cfg, 0, key).out_info
    target_in = _abstract_replicated(tgt_shapes, mesh)
    gbatch = mesh.devices.size  # per-device batch 1
    resid_in = jax.ShapeDtypeStruct((gbatch, SEQ, cfg.n_embd), jnp.bfloat16, sharding=NamedSharding(mesh, P("dp")))
    key_in = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))
    step = make_train_step(
        lm=lm, loss_spec=loss_spec, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, remat_recon_forwards=True, mesh=mesh,
    )
    t0 = time.perf_counter()
    lowered = step.lower(state_in, target_in, resid_in, key_in)
    t1 = time.perf_counter()
    lowered.compile()
    t2 = time.perf_counter()
    print(f"{n_layers:>3}L {len(sites):>4}sites spc={sites_per_chunk:>3} chunks={n_chunks:>2} | "
          f"lower={t1-t0:7.2f}s compile={t2-t1:8.2f}s total={t2-t0:8.2f}s", flush=True)


def main() -> None:
    mesh = dp_mesh()
    cfg = llama31_8b_config()
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1.5e-4, weight_decay=0.0))
    opt_ci = optax.adamw(5e-5, weight_decay=0.0)
    ci_arch = CIArch(4096, 4, 64, 16384)
    key = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=NamedSharding(mesh, P()))
    print(f"devices={jax.device_count()} (CPU sim), seq={SEQ}\n", flush=True)
    # (n_layers, sites_per_chunk) — 7 sites/layer; spc tracks #chunks
    configs = [(2, 14), (4, 28), (8, 56), (16, 56), (32, 56), (32, 112), (32, 224)]
    if len(sys.argv) > 1:  # optional single (n_layers, spc)
        configs = [(int(sys.argv[1]), int(sys.argv[2]))]
    for n_layers, spc in configs:
        run(n_layers, spc, mesh, cfg, opt_vu, opt_ci, ci_arch, key)


if __name__ == "__main__":
    main()
