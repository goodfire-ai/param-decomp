"""Generate the FIXED, framework-agnostic fixtures for the cross-framework PD
equivalence harness (`jax_equivalence.py`).

Everything is drawn ONCE here with numpy and serialized to `fixtures.npz`, so both
the (now-frozen) torch reference and the JAX impl consume byte-identical inputs and
draw NO RNG of their own. That makes each loss term a deterministic function of the fixtures — the only
way two frameworks can disagree is a genuine math difference, which is exactly what we
want the harness to surface.

The model is embed-internal (it takes token ids and embeds them itself), so the fixtures
carry `tokens` + `embed` instead of injecting a residual. `embed[tokens]` reproduces the
`resid` array bit-exactly (tokens index one distinct embed row per position, that row set
to the residual value — a pure integer gather, no arithmetic), so the JAX token-fed
forward and the residual-fed `torch_reference.json` golden see the identical residual
entering layer 0. `resid` is kept in the npz for the (residual-fed) torch oracle.

Design choices that make the cross-check fp32-tight rather than approximate:

  * **Attention is zeroed** (wq=wk=wv=wo=0) in every layer, so the frozen attn
    contributes nothing and `post_attn == resid`. RoPE / SDPA never run, so there is no
    torch-vs-JAX attention-kernel drift to muddy the comparison. The model collapses to
    `embed[tokens] → (rms_norm → masked MLP) → ... → final rms_norm → lm_head`: plain
    matmuls + rms_norm, which both frameworks compute identically in fp32. The decomposed
    MLP — the thing under test — is the only nontrivial part.
  * **All masks are pre-drawn**: per-site component masks `u` (for stoch), per-site
    weight-delta masks, per-position uniform-k-subset routing (per chunk), and the PPGD
    sources (with the trailing weight-delta channel). Neither framework samples anything.
  * **fp32 everywhere** (`DTYPE`), so we compare at ~1e-5, not bf16's ~1%.

Config: a tiny full model that still exercises the per-chunk loop — `n_decomp_layers`
decomposed MLP layers (each a chunk of its 3 gate/up/down sites) + a frozen tail block,
then final norm + lm_head.
"""

from pathlib import Path

import numpy as np

SEED = 1234
DTYPE = np.float32

# Tiny model dims.
VOCAB = 48
N_EMBD = 16
N_INTERMEDIATE = 32
C = 4
N_DECOMP_LAYERS = 3  # 3 chunks of 3 sites each = the per-chunk loop under test
N_TAIL = 1  # one fully-frozen tail block above the decomposed layers
EPS = 1e-5

B = 2
T = 5

KINDS = ("gate", "up", "down")

# Imp-min knobs (production llama8b_l18 config).
IMP_P = 0.4
IMP_BETA = 0.2
IMP_EPS = 1e-12

OUT = Path(__file__).resolve().parent / "fixtures.npz"


def main() -> None:
    rng = np.random.default_rng(SEED)

    def randn(*shape: int, scale: float = 1.0) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(DTYPE)

    d, di = N_EMBD, N_INTERMEDIATE
    arrays: dict[str, np.ndarray] = {}

    # Residual entering the first decomposed layer. The model is embed-internal, so this is
    # reproduced by `embed[tokens]` (built at the end); kept in the npz for the torch oracle.
    arrays["resid"] = randn(B, T, d, scale=0.5)

    # Per decomposed layer: layernorms (ones), zeroed attn, MLP target weights, V/U.
    for i in range(N_DECOMP_LAYERS):
        arrays[f"ln1_{i}"] = np.ones((d,), DTYPE)
        arrays[f"ln2_{i}"] = np.ones((d,), DTYPE)
        # attn weights zeroed -> attn contributes nothing (post_attn == resid).
        arrays[f"Wg_{i}"] = randn(di, d, scale=d**-0.5)
        arrays[f"Wu_{i}"] = randn(di, d, scale=d**-0.5)
        arrays[f"Wd_{i}"] = randn(d, di, scale=di**-0.5)
        # V/U: small, so V@U != W and the weight-delta is nontrivial (exercises faith +
        # the delta channel).
        arrays[f"Vg_{i}"] = randn(d, C, scale=d**-0.5)
        arrays[f"Ug_{i}"] = randn(C, di, scale=C**-0.5)
        arrays[f"Vu_{i}"] = randn(d, C, scale=d**-0.5)
        arrays[f"Uu_{i}"] = randn(C, di, scale=C**-0.5)
        arrays[f"Vd_{i}"] = randn(di, C, scale=di**-0.5)
        arrays[f"Ud_{i}"] = randn(C, d, scale=C**-0.5)

    # Tail block(s): fully frozen MLP (no decomposition), zeroed attn.
    for j in range(N_TAIL):
        arrays[f"tail_ln1_{j}"] = np.ones((d,), DTYPE)
        arrays[f"tail_ln2_{j}"] = np.ones((d,), DTYPE)
        arrays[f"tail_Wg_{j}"] = randn(di, d, scale=d**-0.5)
        arrays[f"tail_Wu_{j}"] = randn(di, d, scale=d**-0.5)
        arrays[f"tail_Wd_{j}"] = randn(d, di, scale=di**-0.5)

    arrays["norm"] = np.ones((d,), DTYPE)
    arrays["lm_head"] = randn(VOCAB, d, scale=0.1)

    # CI values (lower + upper), per (kind): (B, T, L, C). Pre-squashed in [0,1]-ish.
    # We supply BOTH leaky variants directly so the harness does not depend on the CI fn;
    # the values are arbitrary but fixed. lower in [0,1]; upper allowed slightly >1.
    for k in KINDS:
        lower = rng.uniform(0.0, 1.0, (B, T, N_DECOMP_LAYERS, C)).astype(DTYPE)
        upper = rng.uniform(0.0, 1.1, (B, T, N_DECOMP_LAYERS, C)).astype(DTYPE)
        arrays[f"ci_lower_{k}"] = lower
        arrays[f"ci_upper_{k}"] = upper

    # Stoch fixtures: per site (kind, layer) a component-mask source u and a weight-delta
    # mask, both (B, T, L) for delta and (B, T, L, C) for u; per chunk a per-position
    # uniform-k-subset routing over the chunk's 3 sites: (B, T) int rank per site + k.
    for k in KINDS:
        arrays[f"stoch_u_{k}"] = rng.uniform(0.0, 1.0, (B, T, N_DECOMP_LAYERS, C)).astype(DTYPE)
        arrays[f"stoch_delta_{k}"] = rng.uniform(0.0, 1.0, (B, T, N_DECOMP_LAYERS)).astype(DTYPE)
    # Routing per chunk: a boolean (B, T) per site. Pre-materialized so both frameworks
    # consume identical routing (no sampling). chunk i routes only its own 3 sites.
    for i in range(N_DECOMP_LAYERS):
        # uniform-k-subset over 3 sites: k ~ [1,3], a random k-subset True.
        k_draw = rng.integers(1, len(KINDS) + 1, size=(B, T))
        ranks = rng.random((len(KINDS), B, T)).argsort(axis=0)  # random rank per site per pos
        for j, k in enumerate(KINDS):
            # ranks[j] is (B,T); k_draw is (B,T); compare -> (B,T) bool route for this site.
            arrays[f"route_chunk{i}_{k}"] = (ranks[j] < k_draw).astype(np.bool_)

    # PPGD sources per kind: (1, T, L, C+1), broadcast over batch, trailing delta channel.
    for k in KINDS:
        arrays[f"ppgd_source_{k}"] = rng.uniform(0.0, 1.0, (1, T, N_DECOMP_LAYERS, C + 1)).astype(
            DTYPE
        )

    # Embed-internal token contract: `embed[tokens]` reproduces `resid` bit-exactly. One
    # distinct token per position indexes a dedicated embed row set to that position's
    # residual; the gather is exact (no arithmetic), so the token-fed JAX forward sees the
    # same residual as the residual-fed torch golden. Drawn from `resid` (no RNG), so every
    # other fixture array — and thus the frozen golden — is unchanged.
    n_positions = B * T
    assert n_positions <= VOCAB, (
        f"need VOCAB >= B*T for one token per position, {VOCAB} < {n_positions}"
    )
    arrays["tokens"] = np.arange(n_positions, dtype=np.int32).reshape(B, T)
    embed = np.zeros((VOCAB, d), DTYPE)
    embed[:n_positions] = arrays["resid"].reshape(n_positions, d)
    arrays["embed"] = embed

    scalars = dict(
        VOCAB=VOCAB, N_EMBD=N_EMBD, N_INTERMEDIATE=N_INTERMEDIATE, C=C,
        N_DECOMP_LAYERS=N_DECOMP_LAYERS, N_TAIL=N_TAIL, EPS=EPS, B=B, T=T,
        IMP_P=IMP_P, IMP_BETA=IMP_BETA, IMP_EPS=IMP_EPS,
    )  # fmt: skip
    for name, val in scalars.items():
        arrays[f"_scalar_{name}"] = np.array(val)

    np.savez(OUT, **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    print(f"wrote {OUT} ({len(arrays)} arrays, n_decomp_layers={N_DECOMP_LAYERS}, "
          f"sites={N_DECOMP_LAYERS * 3})")  # fmt: skip


if __name__ == "__main__":
    main()
