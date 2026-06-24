"""CPU repro of the full-step {devices=[2,1,8]} involuntary-remat reshard.

16 forced CPU devices -> (dp=2, tp=8) mesh. Tiny model DIMS but REAL per-site Cs, so the
CI fn's concatenated c_chunk (sum of 7 sites' Cs = 38912) tp-shards at boundaries that
misalign with the per-site C boundaries (2048/4096/8192/10240) -> the suspected reshard.
Lower+compile the train step on CPU (SPMD partitioner runs on any backend) and dump the HLO.
"""
import os
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=16 "
    "--xla_dump_to=/tmp/claude-oli/reshard_hlo --xla_dump_hlo_as_text"
)
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from param_decomp.ci_fn import Chunk, ChunkwiseTransformerCIArch
from param_decomp.components import SiteC, init_decomp_vu
from param_decomp.configs import (
    AdamPGDConfig, ChunkwiseSubsetReconLossConfig, FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig, PersistentPGDReconLossConfig, SCScope,
    UniformKSubsetRoutingConfig,
)
from param_decomp.adversary import init_sources_adam_state
from param_decomp.recon import build_loss_terms
from param_decomp.schedule import ScheduleConfig
from param_decomp.sharding import dp_mesh
from param_decomp.targets.llama8b import canonical_site_cs, llama_site_specs, site_name
from param_decomp.targets.llama8b_sharding import (
    init_ci_fn_placed, init_decomp_vu_placed, init_sources_sharded,
)
from param_decomp.tests.test_llama8b import _tiny_decomposed_lm
from param_decomp.train import TrainState, make_train_step
from vendored_jax.llama import LlamaConfig

print("devices:", jax.device_count())
mesh = dp_mesh(tp=8)  # (dp=2, tp=8)
print("mesh:", dict(mesh.shape))
jax.set_mesh(mesh)

cfg = LlamaConfig(
    vocab_size=256, n_layer=2, n_head=32, n_kv_head=8, n_embd=4096, n_intermediate=14336,
    rope_theta=500000.0, rms_norm_eps=1e-5, max_position_embeddings=512,
    rope_factor=8.0, rope_low_freq_factor=1.0, rope_high_freq_factor=4.0,
    rope_original_max_position_embeddings=128,
)
REAL_C = {"q": 2048, "k": 2048, "v": 4096, "o": 4096, "gate": 8192, "up": 8192, "down": 10240}
site_cs = canonical_site_cs(tuple(
    SiteC(site_name(L, k), c) for L in range(2) for k, c in REAL_C.items()
))
sites = llama_site_specs(cfg, site_cs)
lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))

by_layer: dict[int, list[str]] = {}
for n in lm.site_names:
    by_layer.setdefault(int(n.split(".")[1]), []).append(n)
ci_arch = ChunkwiseTransformerCIArch(
    chunks=tuple(Chunk(input_taps=(f"resid.{L}",), output_sites=tuple(ns))
                 for L, ns in sorted(by_layer.items())),
    input_dim=cfg.n_embd, d_model=4096, n_blocks=4, n_heads=64, mlp_hidden=16384,
)
vu = init_decomp_vu_placed(sites, jax.random.PRNGKey(1), mesh)
ci_fn = init_ci_fn_placed(ci_arch, sites, jax.random.PRNGKey(2), mesh)
opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3))
opt_ci = optax.adamw(1e-3)
seq = 512
src = init_sources_sharded(lm.site_names, tuple(s.C for s in sites), seq, SCScope(),
                           mesh.devices.size, jax.random.PRNGKey(3), mesh)
state = TrainState(
    components=vu, ci_fn=ci_fn,
    components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
    ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
    sources={"PersistentPGDReconLoss": src},
    sources_opt_state={"PersistentPGDReconLoss": init_sources_adam_state(src)},
    step=jnp.zeros((), jnp.int32),
)
loss_terms = build_loss_terms((
    FaithfulnessLossConfig(coeff=1e5),
    ImportanceMinimalityLossConfig(coeff=5e-6, pnorm=2.0, beta=0.2,
        p_anneal_start_frac=0.0, p_anneal_final_p=0.4, p_anneal_end_frac=1.0),
    ChunkwiseSubsetReconLossConfig(routing=UniformKSubsetRoutingConfig(), coeff=0.5,
        sites_per_chunk=7, n_samples=1),
    PersistentPGDReconLossConfig(coeff=0.5, scope=SCScope(),
        optimizer=AdamPGDConfig(beta1=0.5, beta2=0.99,
            lr_schedule=ScheduleConfig(start_val=0.01, warmup_pct=0.025)),
        n_warmup_steps=1),
), lm.site_names)
step = make_train_step(lm=lm, loss_terms=loss_terms, components_optimizer=opt_vu,
                       ci_fn_optimizer=opt_ci, total_steps=100, remat_recon_forwards=True, mesh=mesh)
b = 4
tokens = jax.device_put(jax.random.randint(jax.random.PRNGKey(9), (b, seq), 0, cfg.vocab_size),
                        jax.NamedSharding(mesh, jax.sharding.PartitionSpec("dp")))
print("=== compiling step (SPMD partitioner runs here; watch for Involuntary remat) ===")
compiled = step.lower(lm, state, tokens, jax.random.PRNGKey(100)).compile()
print("=== compiled OK ===")
