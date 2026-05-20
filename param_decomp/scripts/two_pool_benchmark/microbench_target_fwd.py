"""Focused microbench for `TinyTransformer` forward at maxA shapes.

Why: in the maxA two-pool profile, `a/1_target_and_ci_fwd` is 61ms and ~57ms
of that is the target forward itself — about 10× off theoretical peak for a
~5 TFLOPs forward on H200 (TF32 ~990 TFLOPs/s → ~5ms). This script tries to
figure out where the time goes and what cheap interventions help.

Configurations measured (all at batch=66, seq=1024, fp32, TF32 matmul):

  - baseline      raw TinyTransformer.forward
  - per_block     same, but each TinyBlock is wrapped and timed individually
  - bf16          forward under torch.autocast(bf16)
  - compiled      torch.compile(target, mode="reduce-overhead")
  - compiled_bf16 both

Plus a static profile of the model's parameter counts and theoretical FLOPs
to anchor the numbers in something physical.

Run:
    sbatch param_decomp/scripts/two_pool_benchmark/microbench_target_fwd.sbatch
"""

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportOperatorIssue=false, reportMissingTypeArgument=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownLambdaType=false

import statistics
import time
from collections.abc import Callable

import torch

from param_decomp.scripts.two_pool_benchmark._tiny_model import TinyTransformer

VOCAB = 8192
D_MODEL = 768
N_HEADS = 12
D_MLP = 3072
N_BLOCKS = 6
BATCH = 66
SEQ = 1024

N_WARMUP = 5
N_PROFILE = 10


def time_phase(
    fn: Callable[[], object], name: str, *, n: int = N_PROFILE, warmup: int = N_WARMUP
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    avg = statistics.mean(ts)
    print(f"  {name:50s} avg={avg:7.2f}ms  min={min(ts):7.2f}ms  max={max(ts):7.2f}ms  (n={n})")
    return avg


def theoretical_flops() -> float:
    """Forward FLOPs for the TinyTransformer at b=66, s=1024.

    Per block:
      attn projections q/k/v/o: 4 × (2 × b × s × d × d) = 8 b s d²
      sdpa:                     2 × b × s² × d (rough)
      mlp (gate, up, down):     3 × (2 × b × s × d × d_mlp) = 6 b s d d_mlp
    Plus embed + unembed.
    """
    b, s, d, d_mlp = BATCH, SEQ, D_MODEL, D_MLP
    attn = 8 * b * s * d * d
    sdpa = 2 * b * s * s * d  # rough — assumes causal halves it but we ignore
    mlp = 6 * b * s * d * d_mlp
    per_block = attn + sdpa + mlp
    embed = b * s * d  # lookup is essentially free
    unembed = 2 * b * s * d * VOCAB
    return per_block * N_BLOCKS + embed + unembed


def main() -> None:
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    target = TinyTransformer(VOCAB, D_MODEL, N_BLOCKS, N_HEADS, D_MLP).to(device)
    target.requires_grad_(False)
    batch = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)

    n_params = sum(p.numel() for p in target.parameters())
    tflops = theoretical_flops() / 1e12
    # H200 peak TF32: ~989 TFLOPs/s
    peak_ms = tflops / 989 * 1000

    print(
        f"batch={BATCH} seq={SEQ} d={D_MODEL} d_mlp={D_MLP} n_blocks={N_BLOCKS} "
        f"params={n_params / 1e6:.1f}M\n"
        f"theoretical fp/TF32 FLOPs: {tflops:.2f} TFLOPs/forward "
        f"(~{peak_ms:.2f}ms at 989 TFLOPs/s peak)\n"
    )

    # --- baseline ---
    print("=== baseline ===")
    baseline_ms = time_phase(lambda: target(batch), "TinyTransformer.forward")

    # --- per-block breakdown ---
    print("\n=== per-block (raw, each block measured independently) ===")
    # Run forward up to before each block, then time just that block on the running x.
    # We pre-compute the input to each block once, then time the block alone.
    x = target.embed(batch)
    block_xs = [x.detach().clone()]
    for block in target.blocks:
        x = block(x)
        block_xs.append(x.detach().clone())
    # block i's input is block_xs[i]
    for i, block in enumerate(target.blocks):
        x_in = block_xs[i].clone()
        time_phase(lambda b=block, xi=x_in: b(xi), f"block[{i}].forward")

    # Components inside one block.
    print("\n=== sub-ops of block[0] (most representative) ===")
    block0 = target.blocks[0]
    x_in = block_xs[0].clone()
    import torch.nn.functional as F

    time_phase(
        lambda: F.rms_norm(x_in, (x_in.shape[-1],)),
        "rms_norm(x)",
    )
    normed = F.rms_norm(x_in, (x_in.shape[-1],))
    time_phase(
        lambda: block0.attn(normed),
        "attn(rms_norm(x)) — full sublayer",
    )
    attn_out = x_in + block0.attn(F.rms_norm(x_in, (x_in.shape[-1],)))
    time_phase(
        lambda: block0.mlp(F.rms_norm(attn_out, (attn_out.shape[-1],))),
        "mlp(rms_norm(x)) — full sublayer",
    )

    # --- attn deep-dive ---
    print("\n=== inside attn (block[0], on already-normed input) ===")
    attn = block0.attn
    H, D_head = attn.n_heads, attn.head_dim
    time_phase(lambda: attn.q_proj(normed), "  q_proj(x)")
    time_phase(lambda: attn.k_proj(normed), "  k_proj(x)")
    time_phase(lambda: attn.v_proj(normed), "  v_proj(x)")

    def shape_qkv():
        q = attn.q_proj(normed).view(BATCH, SEQ, H, D_head).transpose(1, 2)
        k = attn.k_proj(normed).view(BATCH, SEQ, H, D_head).transpose(1, 2)
        v = attn.v_proj(normed).view(BATCH, SEQ, H, D_head).transpose(1, 2)
        return q, k, v

    q, k, v = shape_qkv()
    time_phase(shape_qkv, "  q_proj + k_proj + v_proj + reshape")
    time_phase(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
        "  sdpa core (default backend, fp32)",
    )

    # Force each SDPA backend explicitly to see which the default picks.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = (
        ("math", SDPBackend.MATH),
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    )

    def make_sdpa_runner(backend, q_, k_, v_):
        def run():
            with sdpa_kernel([backend]):
                return F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)

        return run

    for backend_name, backend in backends:
        # First probe: does this backend work for these shapes/dtypes at all?
        try:
            with sdpa_kernel([backend]):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        except (RuntimeError, NotImplementedError) as e:
            print(f"  sdpa fp32 backend={backend_name:10s} NOT AVAILABLE  ({str(e)[:60]}...)")
            continue
        time_phase(
            make_sdpa_runner(backend, q, k, v),
            f"  sdpa fp32 forced={backend_name:10s}",
            n=5,
            warmup=2,
        )

    # Same forced-backend test under bf16 — this is where flash/cudnn become available.
    print("\n  -- forced backends under bf16 q/k/v --")
    q_bf, k_bf, v_bf = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    for backend_name, backend in backends:
        try:
            with sdpa_kernel([backend]):
                _ = F.scaled_dot_product_attention(q_bf, k_bf, v_bf, is_causal=True)
        except (RuntimeError, NotImplementedError) as e:
            print(f"  sdpa bf16 backend={backend_name:10s} NOT AVAILABLE  ({str(e)[:60]}...)")
            continue
        time_phase(
            make_sdpa_runner(backend, q_bf, k_bf, v_bf),
            f"  sdpa bf16 forced={backend_name:10s}",
            n=5,
            warmup=2,
        )

    sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def finish_attn():
        out = sdpa_out.transpose(1, 2).reshape(BATCH, SEQ, H * D_head)
        return attn.o_proj(out)

    time_phase(finish_attn, "  transpose+reshape + o_proj")

    # And the same under bf16 autocast for SDPA only — answers "is it just SDPA that needs bf16?"
    print("\n=== inside attn under bf16 autocast ===")

    def attn_bf16():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return attn(normed)

    time_phase(attn_bf16, "  attn (bf16 autocast)")

    def sdpa_bf16():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    time_phase(sdpa_bf16, "  sdpa (bf16 autocast)")

    # --- bf16 autocast ---
    print("\n=== bf16 autocast ===")

    def fwd_bf16():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return target(batch)

    bf16_ms = time_phase(fwd_bf16, "TinyTransformer.forward (bf16 autocast)")

    # --- torch.compile ---
    print("\n=== torch.compile (mode='reduce-overhead') ===")
    compiled = torch.compile(target, mode="reduce-overhead", dynamic=False)
    # warm-up runs trigger compile
    print("  (warming up compiled — first few calls trigger compile, slower than steady state)")
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled(batch)
        torch.cuda.synchronize()
        print(f"  compile warmup [{i}]: {(time.perf_counter() - t0) * 1000:.2f}ms")
    compiled_ms = time_phase(lambda: compiled(batch), "TinyTransformer.forward (compiled)")

    # --- compiled + bf16 ---
    print("\n=== compiled + bf16 autocast ===")

    def fwd_compiled_bf16():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return compiled(batch)

    # warm-up
    for _ in range(3):
        fwd_compiled_bf16()
    compiled_bf16_ms = time_phase(fwd_compiled_bf16, "compiled + bf16 autocast")

    # --- summary ---
    print(
        f"\nsummary (vs theoretical peak {peak_ms:.2f}ms):\n"
        f"  baseline       {baseline_ms:7.2f}ms   ({baseline_ms / peak_ms:5.1f}× peak)\n"
        f"  bf16           {bf16_ms:7.2f}ms   ({bf16_ms / peak_ms:5.1f}× peak)   ({bf16_ms / baseline_ms * 100:5.1f}% of baseline)\n"
        f"  compiled       {compiled_ms:7.2f}ms   ({compiled_ms / peak_ms:5.1f}× peak)   ({compiled_ms / baseline_ms * 100:5.1f}% of baseline)\n"
        f"  compiled+bf16  {compiled_bf16_ms:7.2f}ms   ({compiled_bf16_ms / peak_ms:5.1f}× peak)   ({compiled_bf16_ms / baseline_ms * 100:5.1f}% of baseline)\n"
    )


if __name__ == "__main__":
    main()
