import sys, glob, yaml
import numpy as np, jax, jax.numpy as jnp, pyarrow.parquet as pq
from pathlib import Path
from jax import random
from param_decomp.sharding import dp_mesh
from param_decomp.run_state import build_optimizers, init_train_state
from param_decomp.checkpoint import make_checkpoint_manager, restore_latest
from param_decomp.train import COMPUTE_DT, cast_floating
from param_decomp.losses import kl_per_position
from param_decomp.eval import next_token_cross_entropy
from param_decomp.recon import uniform_k_subset_routes
from param_decomp.hidden_acts_eval import _all_false_routes
from param_decomp_lab.experiments.lm.config import build_from_schema
from param_decomp_lab.experiments.lm.load_run import build_target

ROOT = Path("/mnt/delicate-frog/artifacts/mechanisms/param-decomp/runs")
SRC = Path("param_decomp/configs/smoothl0_investigation/rdsc-frac-c2e-4-400k.yaml")
rids = sys.argv[1:]
NB, B, THR, PGD_STEPS, PGD_SS = 8, 32, 0.01, 20, 0.1

schema = yaml.safe_load(open(SRC)); schema["run_id"] = "p-00000000"
cfg0 = build_from_schema(schema); mesh = dp_mesh()
lm, frozen, prefix, prf, vocab = build_target(cfg0, mesh)
frozen = cast_floating(frozen, COMPUTE_DT)  # eval runs bf16; rms_norm output dtype follows the weight dtype
ov, oc, _ = build_optimizers(cfg0.pd); ik, sk = random.split(random.PRNGKey(cfg0.pd.seed))
ref = init_train_state(cfg0.pd, lm, cfg0.ci_fn, cfg0.data, ov, oc, ik, sk, mesh)
SN = lm.site_names; CC = {s.name: s.C for s in lm.sites}

sl = cfg0.data.seq_len
shard = sorted(glob.glob(str(cfg0.data.dir / "*.parquet")))[0]
arr = np.array(pq.read_table(shard, columns=["input_ids"]).column("input_ids").to_pylist()[: NB * B], dtype=np.int32)[:, :sl]
batches = arr.reshape(NB, B, sl)
harvest = jax.jit(lambda t: prf(prefix, t))

# metric keys: name -> (kind) ; kind in {kl, ceu, mse}
@jax.jit
def metrics(components, ci_fn, frozen, residual, token_ids, key):
    comp = cast_floating(components, COMPUTE_DT); cifn = cast_floating(ci_fn, COMPUTE_DT)
    residual = residual.astype(COMPUTE_DT)  # eval runs bf16 (autocast); attn kernel rejects fp32 Q
    clean = lm.clean_output(frozen, residual)
    ci = cifn(lm.site_inputs(frozen, residual)).lower
    cit = {s: jnp.where(ci[s] > THR, ci[s], jnp.zeros_like(ci[s])) for s in SN}
    leading = residual.shape[:-1]
    zd = {s: jnp.zeros(leading, COMPUTE_DT) for s in SN}        # delta masks, (B,T)
    zmask = {s: jnp.zeros_like(ci[s]) for s in SN}              # zero CI mask, (B,T,C)
    ones = {s: jnp.ones_like(ci[s]) for s in SN}
    mout = lambda m, d: lm.masked_output(frozen, comp, residual, m, d, None, SN, True)
    sout = lambda m, d, f: lm.masked_site_outputs(frozen, comp, residual, m, d, None, SN, f)
    target_ce = next_token_cross_entropy(clean, token_ids)
    ce_zero = next_token_cross_entropy(mout(zmask, zd), token_ids)
    ha_clean = lm.masked_site_outputs(frozen, comp, residual, ones, zd, _all_false_routes(SN, leading), SN, False)
    def mse(masked):
        num = sum(jnp.sum((masked[s].astype(jnp.float32) - ha_clean[s].astype(jnp.float32)) ** 2) for s in SN)
        return num / sum(ha_clean[s].size for s in SN)
    out = {}
    for tag, base in (("conv", ci), ("thr", cit)):
        lci = mout(base, zd)
        out[f"ci_kl/{tag}"] = kl_per_position(lci, clean)
        out[f"ci_ceu/{tag}"] = (next_token_cross_entropy(lci, token_ids) - target_ce) / (ce_zero - target_ce)
        ks, kd = random.split(random.fold_in(key, 1))
        stoch = {s: base[s] + (1 - base[s]) * random.uniform(random.fold_in(ks, i), base[s].shape, COMPUTE_DT) for i, s in enumerate(SN)}
        sdel = {s: random.uniform(random.fold_in(kd, i), leading, COMPUTE_DT) for i, s in enumerate(SN)}
        out[f"stoch_kl/{tag}"] = kl_per_position(mout(stoch, sdel), clean)
        routes = uniform_k_subset_routes(random.fold_in(key, 2), tuple(SN), leading)
        subm = {s: jnp.where(routes[s][..., None], stoch[s], ones[s]) for s in SN}
        subd = {s: jnp.where(routes[s], sdel[s], zd[s]) for s in SN}
        out[f"subset_kl/{tag}"] = kl_per_position(mout(subm, subd), clean)
        def kl_at(src, base=base):
            m = {s: (base[s] + (1 - base[s]) * src[s][..., :-1]).astype(COMPUTE_DT) for s in SN}
            d = {s: src[s][..., -1].astype(COMPUTE_DT) for s in SN}
            return kl_per_position(mout(m, d), clean)
        def ascend(src, _):
            g = jax.grad(kl_at)(src)
            return {s: jnp.clip(src[s] + PGD_SS * jnp.sign(g[s]), 0.0, 1.0) for s in SN}, None
        init = {s: random.uniform(random.fold_in(random.fold_in(key, 3), i), (1, 1, CC[s] + 1), jnp.float32) for i, s in enumerate(SN)}
        fin, _ = jax.lax.scan(ascend, init, None, length=PGD_STEPS)
        out[f"pgd_kl/{tag}"] = kl_at(fin)
        out[f"ci_ha_mse/{tag}"] = mse(sout(base, zd, False))
        out[f"stoch_ha_mse/{tag}"] = mse(sout(stoch, sdel, True))
    return out

METRICS = ["ci_kl", "ci_ceu", "stoch_kl", "subset_kl", "pgd_kl", "ci_ha_mse", "stoch_ha_mse"]
print("RESULTS run_id " + " ".join(f"{m}.conv {m}.thr" for m in METRICS))
key = random.PRNGKey(0)
for rid in rids:
    mgr = make_checkpoint_manager(ROOT / rid / "ckpts", cfg0.cadence.keep_last_n_checkpoints)
    state, step = restore_latest(mgr, ref)
    acc = {}
    for bi in range(NB):
        t = jnp.asarray(batches[bi]); r = harvest(t)
        o = metrics(state.components, state.ci_fn, frozen, r, t, random.fold_in(key, bi))
        for k, v in o.items():
            acc.setdefault(k, []).append(float(v))
    vals = {k: float(np.mean(v)) for k, v in acc.items()}
    cells = " ".join(f"{vals[m+'/conv']:.6g} {vals[m+'/thr']:.6g}" for m in METRICS)
    print(f"RESULT {rid} {cells}", flush=True)
