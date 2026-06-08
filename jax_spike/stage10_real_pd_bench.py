"""Stage 10: the REAL Llama-3.1-8B L18-MLP PD workload in JAX — throughput (tok/s/GPU).

Mirrors the controlled 4-way spec (see lore 2026-06-08--4way-controlled-topology-plan):
  * residual-start: forward runs the L18->L31 SUFFIX (14 Llama-8B blocks) + norm + lm_head,
    starting from a residual-stream input (random, benchmark scale).
  * decompose ONLY the L18 MLP gate/up/down (C=24576), weight-delta on; all other linears frozen.
  * CI fn = global_shared_transformer: per-site input acts (gate_in=up_in=rmsnorm(resid) 4096,
    down_in=silu(gate)*up 14336) concat 22528 -> Linear d_model 4096 -> 4 bidirectional RoPE
    blocks (64 heads, mlp 16384) -> Linear 3*24576 -> leaky-hard-sigmoid.
  * losses per step: faithfulness (weight MSE over 3 sites) + importance-minimality
    + StochasticReconLayerwise (3 masked suffix forwards, one site each) + PersistentPGD recon
    (1 masked suffix forward with the persistent broadcast source) ; plus a persistent-PGD
    source Adam update (1 suffix fwd+bwd). All recon = MSE on logits (KL-order cost).
  * bf16 params/compute; optax adamw (fp32 states) for V/U and the CI fn.

Single-pool GSPMD: params replicated, resid input + PGD source sharded over the 'dp' mesh axis
(--shard_params optionally shards V/U + CI head over 'dp' to fit if replication OOMs).

Same warm-up + blocked/dispatch timing discipline as stage9 (jax.block_until_ready before timing).
"""

import argparse
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
import vendored_jax.llama as llama_mod
from distributed_util import dp_mesh, init_distributed
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from vendored_jax.llama import (
    LlamaConfig,
    apply_rope,
    causal_sdpa,
    llama3_inv_freq,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
)

llama_mod.USE_FLASH_ATTENTION = True
jax.config.update("jax_default_matmul_precision", "tensorfloat32")  # match torch TF32 path

DT = jnp.bfloat16
SITES = ("gate", "up", "down")
COEFF = dict(faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5)
P_IMP = 0.4  # imp-min p-anneal final p


# ----------------------------- leaky-hard-sigmoid (lower) -----------------------------
@jax.custom_vjp
def lhs(x):
    return jnp.clip(x, 0.0, 1.0)


def _lhs_f(x):
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_b(x, g):
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lhs.defvjp(_lhs_f, _lhs_b)


# ----------------------------- target suffix (frozen) -----------------------------
class FrozenAttn(eqx.Module):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)

    def __call__(self, x, inv_freq):
        b, t, _ = x.shape
        q = (x @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.wk.T).reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.wv.T).reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        y = causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)
        return y @ self.wo.T


class FrozenMLP(eqx.Module):
    wg: jax.Array
    wu: jax.Array
    wd: jax.Array

    def __call__(self, x):
        return (jax.nn.silu(x @ self.wg.T) * (x @ self.wu.T)) @ self.wd.T


class FrozenBlock(eqx.Module):
    ln1: jax.Array
    ln2: jax.Array
    attn: FrozenAttn
    mlp: FrozenMLP
    eps: float = eqx.field(static=True)

    def __call__(self, x, inv_freq):
        x = x + self.attn(rms_norm(x, self.ln1, self.eps), inv_freq)
        x = x + self.mlp(rms_norm(x, self.ln2, self.eps))
        return x


# ----------------------------- decomposed L18 MLP -----------------------------
# Trainable components live in DecompVU; the frozen L18-MLP target weights live in Target
# (Wg/Wu/Wd) so the frozen suffix is a single pytree passed to the jit'd step as a runtime
# argument — NOT captured as a 7GB HLO constant (which made compilation pathologically slow).
class DecompVU(eqx.Module):
    Vg: jax.Array  # (d_in, C)
    Ug: jax.Array  # (C, d_out)
    Vu: jax.Array
    Uu: jax.Array
    Vd: jax.Array
    Ud: jax.Array


def weight_deltas(vu: DecompVU, Wg, Wu, Wd):
    return {
        "gate": Wg - (vu.Vg @ vu.Ug).T,
        "up": Wu - (vu.Vu @ vu.Uu).T,
        "down": Wd - (vu.Vd @ vu.Ud).T,
    }


def _proj(x, V, U, W, mask, delta_mask):
    """Masked weight-delta forward of one decomposed linear (matches torch LinearComponents)."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    if W is not None:  # weight-delta term
        delta = W - (V @ U).T  # (d_out,d_in)
        out = out + delta_mask * (x @ delta.T)
    return out


def decomp_mlp_forward(vu: DecompVU, Wg, Wu, Wd, x, masks, delta_masks):
    """x:(b,t,d) -> (b,t,d). masks/delta_masks: dict site->array or None per site."""
    gate = _proj(x, vu.Vg, vu.Ug, Wg, masks["gate"], delta_masks["gate"])
    up = _proj(x, vu.Vu, vu.Uu, Wu, masks["up"], delta_masks["up"])
    d_in = jax.nn.silu(gate) * up
    return _proj(d_in, vu.Vd, vu.Ud, Wd, masks["down"], delta_masks["down"])


def mlp_site_inputs(Wg, Wu, x):
    """Clean (unmasked) per-site inputs for the CI fn: gate_in, up_in (d), down_in (intermediate)."""
    gate = x @ Wg.T
    up = x @ Wu.T
    d_in = jax.nn.silu(gate) * up
    return x, x, d_in


class Target(eqx.Module):
    """Frozen suffix: L18 (frozen attn/lns + frozen MLP weights) + 13 frozen blocks, norm, lm_head."""

    l18_ln1: jax.Array
    l18_ln2: jax.Array
    l18_attn: FrozenAttn
    l18_Wg: jax.Array
    l18_Wu: jax.Array
    l18_Wd: jax.Array
    rest: list  # 13 FrozenBlock (L19..L31)
    norm: jax.Array
    lm_head: jax.Array
    inv_freq: jax.Array = eqx.field()
    eps: float = eqx.field(static=True)


def suffix_logits(tgt: Target, vu: DecompVU, resid, masks, delta_masks):
    x = resid + tgt.l18_attn(rms_norm(resid, tgt.l18_ln1, tgt.eps), tgt.inv_freq)
    mlp_in = rms_norm(x, tgt.l18_ln2, tgt.eps)
    x = x + decomp_mlp_forward(vu, tgt.l18_Wg, tgt.l18_Wu, tgt.l18_Wd, mlp_in, masks, delta_masks)
    for blk in tgt.rest:
        x = blk(x, tgt.inv_freq)
    x = rms_norm(x, tgt.norm, tgt.eps)
    return x @ tgt.lm_head.T


def l18_resid_to_mlp_input(tgt: Target, resid):
    x = resid + tgt.l18_attn(rms_norm(resid, tgt.l18_ln1, tgt.eps), tgt.inv_freq)
    return rms_norm(x, tgt.l18_ln2, tgt.eps)


# ----------------------------- CI fn: global_shared_transformer -----------------------------
class CIBlock(eqx.Module):
    ln1: jax.Array
    ln2: jax.Array
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x, inv_freq):
        b, t, d = x.shape
        h = rms_norm(x, self.ln1, self.eps)
        q = (h @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (h @ self.wk.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (h @ self.wv.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        # bidirectional flash attention (no causal mask) — avoids materializing (b,h,t,t) scores
        qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        y = jax.nn.dot_product_attention(qt, kt, vt, is_causal=False)
        y = y.reshape(b, t, d)
        x = x + y @ self.wo.T
        h = rms_norm(x, self.ln2, self.eps)
        return x + (jax.nn.gelu(h @ self.w1) @ self.w2)


class CIFn(eqx.Module):
    in_proj: jax.Array  # (total_in, d_model)
    blocks: list  # CIBlock
    out_head: jax.Array  # (d_model, total_c)
    inv_freq: jax.Array = eqx.field()
    C: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, site_inputs):
        # site_inputs: (gate_in, up_in, down_in) each (b,t,*); rms-norm + concat
        normed = [rms_norm(s, jnp.ones((s.shape[-1],), DT), self.eps) for s in site_inputs]
        x = jax.nn.relu(jnp.concatenate(normed, axis=-1) @ self.in_proj)
        for blk in self.blocks:
            x = blk(x, self.inv_freq)
        flat = x @ self.out_head  # (b,t,3C)
        return {s: lhs(flat[..., i * self.C : (i + 1) * self.C]) for i, s in enumerate(SITES)}


# ----------------------------- init -----------------------------
def init_target(cfg: LlamaConfig, C: int, key) -> tuple[Target, DecompVU]:
    ks = iter(random.split(key, 256))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    sc = d**-0.5

    def n(shape, s=sc):
        return (random.normal(next(ks), shape) * s).astype(DT)

    def fattn():
        return FrozenAttn(
            n((qd, d)),
            n((kvd, d)),
            n((kvd, d)),
            n((d, qd)),
            cfg.n_head,
            cfg.n_kv_head,
            cfg.head_dim,
            cfg.n_rep,
        )

    def fmlp():
        return FrozenMLP(n((di, d)), n((di, d)), n((d, di)))

    inv_freq = llama3_inv_freq(cfg)
    vu = DecompVU(
        Vg=n((d, C)),
        Ug=n((C, di), C**-0.5),
        Vu=n((d, C)),
        Uu=n((C, di), C**-0.5),
        Vd=n((di, C)),
        Ud=n((C, d), C**-0.5),
    )
    rest = [
        FrozenBlock(jnp.ones((d,), DT), jnp.ones((d,), DT), fattn(), fmlp(), cfg.rms_norm_eps)
        for _ in range(cfg.n_layer - 1)
    ]
    tgt = Target(
        l18_ln1=jnp.ones((d,), DT),
        l18_ln2=jnp.ones((d,), DT),
        l18_attn=fattn(),
        l18_Wg=n((di, d)),
        l18_Wu=n((di, d)),
        l18_Wd=n((d, di)),
        rest=rest,
        norm=jnp.ones((d,), DT),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=inv_freq,
        eps=cfg.rms_norm_eps,
    )
    return tgt, vu


def init_ci(d_model, n_blocks, n_heads, mlp_hidden, total_in, C, key) -> CIFn:
    ks = iter(random.split(key, 256))
    hd = d_model // n_heads

    def n(shape, s):
        return (random.normal(next(ks), shape) * s).astype(DT)

    def block():
        return CIBlock(
            ln1=jnp.ones((d_model,), DT),
            ln2=jnp.ones((d_model,), DT),
            wq=n((d_model, d_model), d_model**-0.5),
            wk=n((d_model, d_model), d_model**-0.5),
            wv=n((d_model, d_model), d_model**-0.5),
            wo=n((d_model, d_model), d_model**-0.5),
            w1=n((d_model, mlp_hidden), d_model**-0.5),
            w2=n((mlp_hidden, d_model), mlp_hidden**-0.5),
            n_head=n_heads,
            head_dim=hd,
            eps=1e-5,
        )

    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    return CIFn(
        in_proj=n((total_in, d_model), total_in**-0.5),
        blocks=[block() for _ in range(n_blocks)],
        out_head=n((d_model, len(SITES) * C), d_model**-0.5),
        inv_freq=inv_freq,
        C=C,
        eps=1e-5,
    )


# ----------------------------- training step -----------------------------
class State(NamedTuple):
    trainable: tuple  # (l18_mlp_VU, ci_fn)
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    source: dict  # broadcast PGD sources: site -> (1,T,C)
    step: jax.Array


def recon(a, b):
    return jnp.mean((a - b) ** 2)


def make_step(opt_vu, opt_ci, lr_pgd, n_pgd):
    @jax.jit
    def step(state: State, frozen: Target, resid, key):
        dm = {s: jnp.ones((1, 1, 1), DT) for s in SITES}  # delta_mask = 1 (broadcast)
        nomask = {s: None for s in SITES}
        Wg, Wu, Wd = frozen.l18_Wg, frozen.l18_Wu, frozen.l18_Wd

        def loss_fn(trainable):
            vu, ci_fn = trainable
            clean = jax.lax.stop_gradient(suffix_logits(frozen, vu, resid, nomask, dm))
            mlp_in = l18_resid_to_mlp_input(frozen, resid)
            site_in = mlp_site_inputs(Wg, Wu, mlp_in)
            ci = ci_fn(site_in)

            # faithfulness (weight MSE over 3 sites)
            wd = weight_deltas(vu, Wg, Wu, Wd)
            l_faith = sum((d**2).sum() for d in wd.values()) / sum(d.size for d in wd.values())
            # importance-minimality
            l_imp = jnp.mean(jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** P_IMP) for v in ci.values()]))

            # stochastic recon, layerwise: one masked site per forward
            l_stoch = jnp.array(0.0)
            for i, s in enumerate(SITES):
                u = random.uniform(random.fold_in(key, i), ci[s].shape, dtype=DT)
                m = ci[s] + (1 - ci[s]) * u
                masks = {**nomask, s: m}
                l_stoch = l_stoch + recon(suffix_logits(frozen, vu, resid, masks, dm), clean)
            l_stoch = l_stoch / len(SITES)

            # persistent PGD recon: all sites masked by ci*sigmoid(source) (broadcast source)
            src = jax.lax.stop_gradient(state.source)
            ppgd_masks = {s: ci[s] * jax.nn.sigmoid(src[s]) for s in SITES}
            l_ppgd = recon(suffix_logits(frozen, vu, resid, ppgd_masks, dm), clean)

            tot = (
                COEFF["faith"] * l_faith
                + COEFF["imp"] * l_imp
                + COEFF["stoch"] * l_stoch
                + COEFF["ppgd"] * l_ppgd
            )
            return tot, (clean, ci)

        (tot, (clean, ci)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.trainable
        )
        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.trainable[0])
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.trainable[1])
        new_vu = eqx.apply_updates(state.trainable[0], upd_vu)
        new_ci = eqx.apply_updates(state.trainable[1], upd_ci)

        # persistent PGD source update (n_pgd inner steps; steady-state n_pgd=1)
        vu_det = jax.lax.stop_gradient(state.trainable[0])
        ci_det = jax.lax.stop_gradient(ci)

        def adv(src):
            masks = {s: ci_det[s] * jax.nn.sigmoid(src[s]) for s in SITES}
            return recon(suffix_logits(frozen, vu_det, resid, masks, dm), clean)

        def body(src, _):
            g = jax.grad(adv)(src)
            return jax.tree.map(lambda s, gg: s + lr_pgd * gg, src, g), None

        new_src, _ = jax.lax.scan(body, state.source, None, length=n_pgd)
        new_src = jax.lax.stop_gradient(new_src)
        return State((new_vu, new_ci), os_vu, os_ci, new_src, state.step + 1), tot

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--per_gpu_batch", type=int, default=8)
    ap.add_argument("--C", type=int, default=24576)
    ap.add_argument(
        "--n_warmup", type=int, default=1, help="persistent-PGD inner steps per train step"
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--shard_params", action="store_true", help="shard V/U + CI head over dp axis")
    # dims default to real Llama-3.1-8B suffix; overridable for CPU smoke tests
    ap.add_argument("--n_embd", type=int, default=4096)
    ap.add_argument("--n_intermediate", type=int, default=14336)
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--suffix_layers", type=int, default=14)
    ap.add_argument("--n_head", type=int, default=32)
    ap.add_argument("--n_kv_head", type=int, default=8)
    ap.add_argument("--ci_d_model", type=int, default=4096)
    ap.add_argument("--ci_blocks", type=int, default=4)
    ap.add_argument("--ci_heads", type=int, default=64)
    ap.add_argument("--ci_mlp", type=int, default=16384)
    args = ap.parse_args()

    init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0
    gbatch = args.per_gpu_batch * ndev

    cfg = LlamaConfig(
        vocab_size=args.vocab,
        n_layer=args.suffix_layers,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        n_intermediate=args.n_intermediate,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )
    total_in = cfg.n_embd + cfg.n_embd + cfg.n_intermediate  # gate_in + up_in + down_in
    if is0:
        print(
            f"[p0] STAGE10 real L18-MLP PD | {ndev} GPU | gbatch={gbatch} seq={args.seq} "
            f"C={args.C} suffix_layers={cfg.n_layer} n_pgd={args.n_warmup} shard={args.shard_params}"
        )

    tgt, vu0 = init_target(cfg, args.C, random.PRNGKey(0))
    ci_fn = init_ci(
        args.ci_d_model,
        args.ci_blocks,
        args.ci_heads,
        args.ci_mlp,
        total_in,
        args.C,
        random.PRNGKey(1),
    )

    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    repl = NamedSharding(mesh, P())
    shard_dp = NamedSharding(mesh, P("dp"))

    def put(x, s):
        return jax.device_put(x, s)

    tgt = jax.tree.map(lambda a: put(a, repl) if eqx.is_array(a) else a, tgt)
    vu0 = jax.tree.map(lambda a: put(a, repl) if eqx.is_array(a) else a, vu0)
    ci_fn = jax.tree.map(lambda a: put(a, repl) if eqx.is_array(a) else a, ci_fn)

    state = State(
        trainable=(vu0, ci_fn),
        opt_vu=opt_vu.init(eqx.filter(vu0, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source={s: jnp.zeros((1, args.seq, args.C), DT) for s in SITES},
        step=jnp.array(0),
    )

    resid_full = (random.normal(random.PRNGKey(42), (gbatch, args.seq, cfg.n_embd)) * 0.5).astype(
        DT
    )
    resid = jax.device_put(resid_full, shard_dp)
    step = make_step(opt_vu, opt_ci, lr_pgd=0.01, n_pgd=args.n_warmup)

    for _ in range(2):
        state, tot = step(state, tgt, resid, random.PRNGKey(7))
        jax.block_until_ready((state.source, tot))

    per = []
    for s in range(args.steps):
        t = time.time()
        state, tot = step(state, tgt, resid, random.PRNGKey(1000 + s))
        jax.block_until_ready((state.source, tot))
        per.append(time.time() - t)
    blocked = sum(per) / len(per)

    t = time.time()
    for s in range(args.steps):
        state, tot = step(state, tgt, resid, random.PRNGKey(2000 + s))
    dispatch = (time.time() - t) / args.steps
    jax.block_until_ready((state.source, tot))

    if is0:
        toks = gbatch * args.seq
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | dispatch {dispatch * 1e3:.1f} ms/step "
            f"| {'HOST-BOUND' if dispatch > 0.7 * blocked else 'device-bound'}"
        )
        print(
            f"[p0] {toks / blocked:,.0f} tok/s | {toks / blocked / ndev:,.0f} tok/s/GPU "
            f"| final loss {float(tot):.4f}"
        )
        print(f"[p0] STAGE10 ({ndev} GPU): OK")

    jax.experimental.multihost_utils.sync_global_devices("stage10_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
