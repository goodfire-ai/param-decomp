"""Microbench Qwen3-0.6B-Base target forward at varying batch/seq.

Quick: load the model in bf16 (since we already validated that's the right
target precision on H200), time forward at a few shapes, and report what
fits + how fast. Picks the b/s for the upcoming 2-pool Qwen bench.

Single GPU. Frozen target, no_grad, no caching — pure target_fwd time only.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false

import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
N_WARMUP = 3
N_PROFILE = 5

# (batch, seq) — start small, grow until OOM-ish.
SHAPES = [
    (8, 1024),
    (16, 1024),
    (20, 1024),
    (32, 1024),
    (48, 1024),
    (64, 1024),
    (8, 2048),
    (16, 2048),
]


def main() -> None:
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    print(
        f"model: {MODEL_ID}\n"
        f"  hidden_size={cfg.hidden_size}  intermediate_size={cfg.intermediate_size}\n"
        f"  num_hidden_layers={cfg.num_hidden_layers}  num_attention_heads={cfg.num_attention_heads}\n"
        f"  num_kv_heads={cfg.num_key_value_heads}  vocab_size={cfg.vocab_size}\n"
    )

    # Load in bf16 directly (target is frozen, no fp32 precision needed).
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M  dtype: {next(model.parameters()).dtype}\n")

    # Decomposable site enumeration — what the 2-pool bench will use.
    sites: list[str] = []
    site_tags = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for name, mod in model.named_modules():
        if (
            isinstance(mod, torch.nn.Linear)
            and any(tag in name for tag in site_tags)
            and "lm_head" not in name
        ):
            sites.append(name)
    print(f"decomposable sites: {len(sites)}  (first/last: {sites[0]} / {sites[-1]})\n")

    print(f"{'batch':>6} {'seq':>6} {'mem_GB':>8} {'avg_ms':>9} {'min_ms':>9} {'tok/s/GPU':>12}")
    for b, s in SHAPES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = torch.randint(0, cfg.vocab_size, (b, s), device=device)

        try:
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    _ = model(batch).logits
                torch.cuda.synchronize()
                ts = []
                for _ in range(N_PROFILE):
                    t0 = time.perf_counter()
                    _ = model(batch).logits
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000.0)
        except torch.cuda.OutOfMemoryError:
            print(f"{b:>6} {s:>6}    OOM")
            continue
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        avg = statistics.mean(ts)
        tok_per_sec = (b * s) / (avg / 1000)
        print(f"{b:>6} {s:>6} {peak_gb:>8.2f} {avg:>9.2f} {min(ts):>9.2f} {tok_per_sec:>12.0f}")


if __name__ == "__main__":
    main()
